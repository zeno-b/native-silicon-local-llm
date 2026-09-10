"""Procedural memory: skills the agent loads on demand and improves in use.

A 3B model in a 4096-token window cannot hold domain knowledge, and it cannot be
handed all of it either: the tool lane already spends ~1500 tokens on the tool
protocol. Skills are the answer to both halves. A skill is a short markdown
document describing HOW to do one kind of task, and only its one-line
description sits in the prompt. The body enters context solely when the model
asks for it by name (`load_skill`). That is the progressive-disclosure pattern:
the catalogue costs ~15 tokens per skill, the procedure costs nothing until it
is the procedure in question.

Storage is one directory per skill holding a SKILL.md with YAML frontmatter,
which is the agentskills.io layout, so a skill written here is portable and a
skill written elsewhere can be dropped in. Frontmatter keys beyond `name` and
`description` are ours (version, usage counters) and are ignored by anything
that only understands the standard.

Nothing here talks to a model. Authoring and refinement live in the caller, so
this module stays synchronous, testable, and safe to import from a thread.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .core import *  # noqa: F401,F403


# A skill name arrives from the MODEL (load_skill("...")), so it is untrusted
# input that gets turned into a filesystem path. Restrict it to a slug and
# resolve inside the skills directory, or `load_skill("../../.env")` reads
# whatever it likes.
SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

# Frontmatter is parsed by hand rather than with PyYAML: the dependency is
# optional in this project, the schema is five scalar keys, and a skill file
# that fails to parse must degrade to "skipped" rather than break a start.
_FRONTMATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?", re.S)


def slugify(name: str) -> str:
    """A safe directory name for a skill, or "" if nothing usable survives."""
    text = re.sub(r"[^a-z0-9]+", "-", str(name or "").strip().lower()).strip("-")
    return text[:64] if SLUG.fullmatch(text or "") else ""


@dataclass
class Skill:
    """One procedural-memory document.

    `body` is loaded lazily: list() and catalogue() leave it empty, because the
    whole point is that reading the catalogue does not pay for the procedures.
    """

    name: str
    description: str
    path: Path
    version: int = 1
    uses: int = 0
    wins: int = 0
    losses: int = 0
    updated: str = ""
    body: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def reliability(self) -> float | None:
        """Share of rated uses that went well, or None when never rated.

        None and 1.0 are different things: a brand-new skill has not earned
        trust yet, and presenting it as perfect is how a bad skill spreads.
        """
        rated = self.wins + self.losses
        return (self.wins / rated) if rated else None

    def summary(self) -> str:
        """The one-line catalogue entry. This is what the prompt pays for."""
        line = f"- {self.name}: {self.description}"
        score = self.reliability
        if score is not None and self.wins + self.losses >= 3 and score < 0.5:
            # Say so rather than silently offering something that keeps failing.
            line += " (unreliable so far)"
        return line

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "uses": self.uses,
            "wins": self.wins,
            "losses": self.losses,
            "updated": self.updated,
            "tags": list(self.tags),
            "reliability": self.reliability,
            "chars": len(self.body) if self.body else None,
        }


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """(frontmatter mapping, body). A file with no frontmatter is all body."""
    match = _FRONTMATTER.match(text or "")
    if not match:
        return {}, (text or "").strip()
    meta: dict = {}
    for line in match.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        meta[key.strip().lower()] = value.strip().strip("'\"")
    return meta, text[match.end():].strip()


def _render_frontmatter(skill: Skill) -> str:
    """Serialise the metadata. Values are escaped by being kept to one line."""
    tags = ", ".join(skill.tags)
    return (
        "---\n"
        f"name: {skill.name}\n"
        f"description: {_one_line(skill.description)}\n"
        f"version: {skill.version}\n"
        f"uses: {skill.uses}\n"
        f"wins: {skill.wins}\n"
        f"losses: {skill.losses}\n"
        f"updated: {skill.updated}\n"
        + (f"tags: {tags}\n" if tags else "")
        + "---\n\n"
    )


def _one_line(text: str) -> str:
    """Collapse to a single line: frontmatter here is line-oriented."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _int(meta: dict, key: str, default: int = 0) -> int:
    try:
        return int(str(meta.get(key, default)).strip())
    except Exception:
        return default


class SkillLibrary:
    """The on-disk skill set, plus the counters that let it improve.

    Deliberately re-reads the directory on every call rather than caching. The
    library is a handful of small files, a cache would go stale the moment a
    skill is edited by hand or by a subagent, and "the file on disk is the
    truth" is worth more here than the microseconds.
    """

    def __init__(self, root: Path | str, max_skills: int = 200):
        self.root = Path(root)
        self.max_skills = max(1, int(max_skills))

    # ---------------------------------------------------------------- paths --
    def _dir_for(self, name: str) -> Path | None:
        """The directory for a skill name, or None if the name is not safe.

        Resolves and then checks containment, so a slug that somehow survives
        sanitisation still cannot point outside the library.
        """
        slug = slugify(name)
        if not slug:
            return None
        candidate = (self.root / slug).resolve()
        root = self.root.resolve()
        if candidate != root and root not in candidate.parents:
            return None
        return candidate

    def retired_dir(self) -> Path:
        return self.root / ".retired"

    # ----------------------------------------------------------------- read --
    def _load_file(self, path: Path, with_body: bool) -> Skill | None:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        meta, body = _parse_frontmatter(text)
        name = slugify(meta.get("name") or path.parent.name)
        if not name:
            return None
        description = _one_line(meta.get("description") or "")
        if not description:
            # A skill with no description cannot be offered in the catalogue at
            # all -- the description IS the progressive-disclosure surface -- so
            # fall back to its first prose line rather than dropping it.
            first = next((ln.strip(" #") for ln in body.splitlines() if ln.strip()), "")
            description = _one_line(first)[:200] or "(no description)"
        return Skill(
            name=name,
            description=description,
            path=path,
            version=_int(meta, "version", 1),
            uses=_int(meta, "uses"),
            wins=_int(meta, "wins"),
            losses=_int(meta, "losses"),
            updated=_one_line(meta.get("updated") or ""),
            body=body if with_body else "",
            tags=[t.strip() for t in str(meta.get("tags") or "").split(",") if t.strip()],
        )

    def list(self, with_body: bool = False) -> list[Skill]:
        """Every usable skill, most-used first, then alphabetical."""
        if not self.root.exists():
            return []
        found: list[Skill] = []
        for entry in sorted(self.root.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            skill_file = entry / "SKILL.md"
            if not skill_file.is_file():
                continue
            skill = self._load_file(skill_file, with_body)
            if skill is not None:
                found.append(skill)
        found.sort(key=lambda s: (-s.uses, s.name))
        return found

    def get(self, name: str) -> Skill | None:
        """One skill WITH its body. The only call that pays for a procedure."""
        directory = self._dir_for(name)
        if directory is None:
            return None
        skill_file = directory / "SKILL.md"
        if not skill_file.is_file():
            return None
        return self._load_file(skill_file, with_body=True)

    def search(self, query: str, limit: int = 5) -> list[Skill]:
        """Rank skills against a query by word overlap on name + description.

        Deliberately not embeddings: the corpus is tens of one-line
        descriptions, the tokens are the user's own words, and a lexical score
        needs no model, no index and no warm-up on an 8GB machine.
        """
        words = {w for w in re.findall(r"[a-z0-9]+", (query or "").lower()) if len(w) > 2}
        if not words:
            return self.list()[:limit]
        scored: list[tuple[float, Skill]] = []
        for skill in self.list():
            haystack = f"{skill.name} {skill.description} {' '.join(skill.tags)}".lower()
            hay_words = set(re.findall(r"[a-z0-9]+", haystack))
            overlap = len(words & hay_words)
            if not overlap:
                continue
            # Usage is a weak tiebreak, not a ranking signal: a popular skill
            # should not outrank a better match.
            scored.append((overlap + min(skill.uses, 20) / 100.0, skill))
        scored.sort(key=lambda pair: (-pair[0], pair[1].name))
        return [skill for _, skill in scored[:limit]]

    def best_match(self, query: str, min_overlap: int = 2) -> Skill | None:
        """The one skill clearly about this message, or None.

        Thresholded on purpose. A small model will not call load_skill on its
        own, so the caller loads the match automatically -- which means a loose
        match injects an irrelevant procedure into a prompt that has 4096 tokens
        to spend. Requiring several significant words in common makes a false
        positive much less likely than a missed one, and a miss only costs the
        model the chance to ask for the skill itself.
        """
        words = {w for w in re.findall(r"[a-z0-9]+", (query or "").lower()) if len(w) > 2}
        if len(words) < min_overlap:
            return None
        best: tuple[int, Skill] | None = None
        for skill in self.list():
            haystack = f"{skill.name} {skill.description} {' '.join(skill.tags)}".lower()
            overlap = len(words & set(re.findall(r"[a-z0-9]+", haystack)))
            if overlap >= min_overlap and (best is None or overlap > best[0]):
                best = (overlap, skill)
        return best[1] if best else None

    def catalogue(self, limit: int = 12) -> str:
        """The lines that go INTO the prompt. Bounded, and STABLE.

        Ordered by name, and the first `limit` by name are the ones shown --
        deliberately not by relevance to the current message, and deliberately
        not by usage. Both of those would make the system prompt differ from
        one turn to the next, and on this hardware the prefix cache is worth
        far more than a better-chosen list: an unstable prefix means a full
        re-prefill of the whole prompt every single turn. Skills past the cap
        are still reachable through search_skills, so nothing is lost, and
        `overflowing()` lets the prompt say so.

        Returns "" when there is nothing to say, so a fresh install adds not one
        token to the prompt.
        """
        skills = sorted(self.list(), key=lambda s: s.name)
        if not skills:
            return ""
        return "\n".join(skill.summary() for skill in skills[:max(1, limit)])

    def overflowing(self, limit: int) -> int:
        """How many skills the catalogue had to leave out."""
        return max(0, len(self.list()) - max(1, limit))

    # ---------------------------------------------------------------- write --
    def _write(self, skill: Skill) -> bool:
        """Write SKILL.md atomically. Returns False on any filesystem trouble."""
        directory = self._dir_for(skill.name)
        if directory is None:
            return False
        try:
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / "SKILL.md"
            tmp = directory / "SKILL.md.tmp"
            tmp.write_text(_render_frontmatter(skill) + skill.body.strip() + "\n",
                           encoding="utf-8")
            tmp.replace(target)
            skill.path = target
            return True
        except OSError as exc:
            log(f"Could not write skill {skill.name!r}: {exc}", logging.WARNING)
            return False

    def save(self, name: str, description: str, body: str,
             tags: list[str] | None = None) -> Skill | None:
        """Create or replace a skill. Returns the stored skill, or None.

        Replacing bumps the version and KEEPS the counters: the point of a
        self-improving skill is that its track record survives being rewritten,
        otherwise every refinement resets the evidence about whether it works.
        """
        slug = slugify(name)
        if not slug or not str(body or "").strip():
            return None
        existing = self.get(slug)
        if existing is None and len(self.list()) >= self.max_skills:
            log(f"Skill library is at its {self.max_skills}-skill cap; "
                f"not adding {slug!r}.", logging.WARNING)
            return None
        skill = Skill(
            name=slug,
            description=_one_line(description) or (existing.description if existing else slug),
            path=self.root / slug / "SKILL.md",
            version=(existing.version + 1) if existing else 1,
            uses=existing.uses if existing else 0,
            wins=existing.wins if existing else 0,
            losses=existing.losses if existing else 0,
            updated=iso(utc_now()),
            body=str(body).strip(),
            tags=list(tags or (existing.tags if existing else [])),
        )
        return skill if self._write(skill) else None

    def refine(self, name: str, note: str) -> Skill | None:
        """Append one learned note to a skill and bump its version.

        This is the "improves in use" half of the loop, and it is deliberately
        additive rather than a rewrite: a rewrite asks a 3B model to reproduce a
        working document from memory, which is how a good skill gets worse. A
        bounded list of dated notes cannot lose what already worked.
        """
        skill = self.get(name)
        note = _one_line(note)
        if skill is None or not note:
            return None
        marker = "## Learned in use"
        if marker not in skill.body:
            skill.body = skill.body.rstrip() + f"\n\n{marker}\n"
        if note.lower() in skill.body.lower():
            return skill              # already recorded; do not grow the file
        lines = [ln for ln in skill.body.splitlines()]
        # Keep the notes section bounded, oldest dropped first, so a long-lived
        # skill cannot grow until it no longer fits the context it exists for.
        start = lines.index(marker) + 1
        notes = [ln for ln in lines[start:] if ln.strip().startswith("- ")]
        notes.append(f"- {note}")
        notes = notes[-8:]
        skill.body = "\n".join(lines[:start] + notes)
        skill.version += 1
        skill.updated = iso(utc_now())
        return skill if self._write(skill) else None

    # ------------------------------------------------------------- counters --
    def _bump(self, name: str, **deltas: int) -> Skill | None:
        skill = self.get(name)
        if skill is None:
            return None
        for key, delta in deltas.items():
            setattr(skill, key, max(0, getattr(skill, key, 0) + delta))
        return skill if self._write(skill) else None

    def record_use(self, name: str) -> Skill | None:
        """Count a load. Separate from wins/losses: most uses are never rated."""
        return self._bump(name, uses=1)

    def record_outcome(self, name: str, ok: bool) -> Skill | None:
        return self._bump(name, wins=1) if ok else self._bump(name, losses=1)

    def retire(self, name: str, reason: str = "") -> bool:
        """Move a skill out of the catalogue without destroying it.

        Deleting would throw away the evidence of why it was bad, and a skill
        the model wrote is exactly the thing a human may want to read after it
        misfired. Retired skills are ignored by list() (dot-directory) and can
        be moved back by hand.
        """
        directory = self._dir_for(name)
        if directory is None or not directory.is_dir():
            return False
        try:
            target_root = self.retired_dir()
            target_root.mkdir(parents=True, exist_ok=True)
            target = target_root / f"{directory.name}-{int(time.time())}"
            directory.replace(target)
            if reason:
                (target / "RETIRED.txt").write_text(
                    f"{iso(utc_now())}  {reason}\n", encoding="utf-8")
            log(f"Retired skill {name!r}: {reason or 'no reason given'}", logging.INFO)
            return True
        except OSError as exc:
            log(f"Could not retire skill {name!r}: {exc}", logging.WARNING)
            return False

    def should_retire(self, skill: Skill, min_rated: int, max_loss_rate: float) -> bool:
        """True for a skill with enough evidence that it is doing harm.

        Requires a minimum number of RATED uses, so one bad day cannot retire a
        skill and a never-rated skill is never retired.
        """
        rated = skill.wins + skill.losses
        if rated < max(1, min_rated):
            return False
        return (skill.losses / rated) > max_loss_rate

    def stats(self) -> dict:
        skills = self.list()
        return {
            "count": len(skills),
            "uses": sum(s.uses for s in skills),
            "wins": sum(s.wins for s in skills),
            "losses": sum(s.losses for s in skills),
            "root": str(self.root),
            "retired": sum(1 for _ in self.retired_dir().glob("*"))
            if self.retired_dir().exists() else 0,
        }


__all__ = [
    'Skill',
    'SkillLibrary',
    'slugify',
]
