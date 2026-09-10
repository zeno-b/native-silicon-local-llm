"""LoRA retraining from collected feedback.

Split out of the original single-file deploy.py.

The recipe is derived, not fixed. A constant iteration count at batch size 1 is
either ~19 epochs over the 16-example minimum (memorisation) or well under one
epoch once tool traces are included (rows never seen), depending only on how
much unrelated data happens to be in the corpus. Everything here — iteration
count, the tool/feedback ratio, the rehearsal share, the held-out split — is
computed from the corpus that actually got exported, and the run is judged
against that held-out split before the adapter is promoted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from .core import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .database import *  # noqa: F401,F403
from .model_server import *  # noqa: F401,F403
from .websearch import *  # noqa: F401,F403
from .sysutil import *  # noqa: F401,F403


# mlx-lm prints one of these per evaluation point. The first is measured before
# any weight update, so losses[0] is the pre-training baseline for this run.
_VAL_LOSS_RE = re.compile(r"Iter\s+(\d+)\s*:\s*Val loss\s+([0-9]*\.?[0-9]+)", re.IGNORECASE)

# fnmatch form of sysutil's checkpoint pattern, for shutil.copytree's ignore.
_CHECKPOINT_GLOB = "[0-9][0-9][0-9][0-9][0-9][0-9][0-9]_adapters.safetensors"

# Written to data/sft/replay.jsonl the first time a retrain needs rehearsal data
# and none exists. Deliberately short, generic and domain-neutral: the job of a
# rehearsal set is to hold the model's general instruction-following in place
# while a few dozen feedback rows pull it somewhere specific, so these are about
# tone, format and honesty rather than about any subject. Edit or replace the
# file freely — anything valid in train.jsonl is valid here.
DEFAULT_REPLAY_EXAMPLES: list[tuple[str, str]] = [
    ("Reverse a string in Python.",
     "```python\ns = \"hello\"\nprint(s[::-1])\n```\nSlicing with a step of -1 is the idiomatic form."),
    ("What does HTTP status 429 mean?",
     "Too Many Requests: the client has sent more requests than the server's rate limit allows. "
     "The response usually carries a Retry-After header saying how long to wait."),
    ("Write a bash one-liner that counts lines in every .py file under the current directory.",
     "```bash\nfind . -name '*.py' -exec wc -l {} +\n```"),
    ("Explain what a race condition is, briefly.",
     "Two or more operations touch shared state concurrently and the result depends on which one "
     "happens to run first. It is a bug because the ordering is not guaranteed, so the program is "
     "correct on some runs and wrong on others."),
    ("What is the capital of Portugal?", "Lisbon."),
    ("Is `is` the same as `==` in Python?",
     "No. `==` compares values; `is` compares identity, meaning whether both names refer to the same "
     "object. Use `==` for equality and reserve `is` for None and other singletons."),
    ("What is the time complexity of binary search?",
     "O(log n) on a sorted sequence, because each comparison halves the remaining range."),
    ("Give me a regex that matches an ISO date like 2024-05-31.",
     "```\n^\\d{4}-\\d{2}-\\d{2}$\n```\nIt matches the shape only; it will accept 2024-13-45, so validate "
     "the ranges separately if that matters."),
    ("What does the `git rebase` command do?",
     "It replays your commits on top of another base commit, producing new commits with new hashes. "
     "It gives a linear history but rewrites it, so avoid rebasing anything already pushed and shared."),
    ("How do I read a file line by line in Python?",
     "```python\nwith open(\"file.txt\", encoding=\"utf-8\") as f:\n    for line in f:\n        print(line.rstrip(\"\\n\"))\n```\n"
     "Iterating the file object streams it, so this works on files larger than memory."),
    ("What is the difference between TCP and UDP?",
     "TCP is connection-oriented and guarantees ordered, retransmitted delivery. UDP is connectionless "
     "and does neither, which makes it cheaper and lower-latency. Use TCP when losing a byte is a bug "
     "and UDP when losing a packet is preferable to waiting for it."),
    ("What's the population of the city I live in?",
     "I don't know where you live — that isn't something I have access to. Tell me the city and I can "
     "give you the figure I have, with the caveat that population data goes stale."),
    ("Summarise what a hash function does in one sentence.",
     "It maps input of any size onto a fixed-size value deterministically, so the same input always "
     "gives the same output and a different input almost always gives a different one."),
    ("What does `chmod 755` set?",
     "Read, write and execute for the owner; read and execute for the group and for everyone else. "
     "The digits are owner/group/other, each a sum of 4 (read), 2 (write) and 1 (execute)."),
    ("Write a SQL query for the five most recent rows in a table called events.",
     "```sql\nSELECT * FROM events ORDER BY created_at DESC LIMIT 5;\n```"),
    ("What is idempotency in an API?",
     "An idempotent request has the same effect whether it is sent once or many times. PUT and DELETE "
     "are idempotent by definition; POST usually is not, which is why retrying a POST can double-charge."),
    ("Explain the difference between a list and a tuple in Python.",
     "A list is mutable and a tuple is not. That makes tuples hashable, so they can be dictionary keys "
     "and set members, and it makes them a reasonable choice for a fixed-length record."),
    ("What does the acronym API stand for?", "Application Programming Interface."),
    ("How do I find which process is using port 8080 on macOS?",
     "```bash\nlsof -nP -iTCP:8080 -sTCP:LISTEN\n```"),
    ("What is a memory leak?",
     "Memory that a program has allocated and no longer uses but never releases, so its footprint grows "
     "over time. In garbage-collected languages it usually means a reference is still held somewhere, "
     "often in a cache or a listener list that nothing ever clears."),
    ("Convert 45 degrees Celsius to Fahrenheit.",
     "113 °F. The conversion is F = C × 9/5 + 32, so 45 × 1.8 = 81, plus 32."),
    ("What is the point of a code review?",
     "Catching defects, spreading knowledge of the change beyond its author, and keeping the codebase "
     "consistent. The first is the one people cite and often the least valuable of the three."),
    ("What does `git stash` do?",
     "It saves your uncommitted changes onto a stack and restores a clean working tree. `git stash pop` "
     "reapplies the most recent entry and removes it from the stack."),
    ("Who won the 2031 World Cup?",
     "I don't know — that is past what I can reliably speak to, and guessing a winner would be inventing "
     "one. Check a current source."),
]


class RetrainManager:
    """Manages LoRA retraining with adapter backup, val-loss gating and rollback."""

    def __init__(self, db: Database, model_manager: ModelServerManager, config: Config):
        self.db = db
        self.model_manager = model_manager
        self.config = config
        self.lock = threading.Lock()
        self.status = {"running": False, "message": "idle"}
        self._lora_help = help_cmd("mlx_lm.lora")
        # Breakdown of the last export, so run() can gate on the human-approved
        # count rather than on the padded total.
        self.last_export: dict = {}

    # ------------------------------------------------------------------ #
    # Adapter lifecycle
    # ------------------------------------------------------------------ #
    def _backup_adapter(self) -> Path | None:
        """Backup current adapter before retraining."""
        if not adapter_ready():
            return None
        ADAPTER_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = ADAPTER_BACKUP_DIR / backup_id
        # Two retrains inside the same second would collide on the timestamp.
        suffix = 1
        while backup_path.exists():
            backup_path = ADAPTER_BACKUP_DIR / f"{backup_id}_{suffix}"
            suffix += 1
        # Periodic checkpoints are reproducible intermediate state, not something
        # worth a permanent copy each run — without this filter every backup
        # carries the previous run's checkpoints and the directory grows
        # quadratically in retrain count.
        shutil.copytree(ADAPTER_DIR, backup_path,
                        ignore=shutil.ignore_patterns(_CHECKPOINT_GLOB))
        log(f"Adapter backed up to {backup_path}")
        prune_adapter_backups(self.config.train_max_backups)
        return backup_path

    def _rollback_adapter(self, backup_path: Path | None) -> None:
        """Restore the pre-training adapter after a failed or regressive run."""
        if backup_path is None or not backup_path.exists():
            # No backup means there was no adapter before this run. Leaving the
            # half-trained one in place would silently serve it, so clear it.
            if ADAPTER_DIR.exists():
                try:
                    shutil.rmtree(ADAPTER_DIR)
                    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
                    log("Discarded the new adapter; falling back to the base model.",
                        logging.WARNING)
                except OSError as exc:
                    log(f"Could not discard the new adapter: {exc}", logging.ERROR)
            return
        try:
            if ADAPTER_DIR.exists():
                shutil.rmtree(ADAPTER_DIR)
            shutil.copytree(backup_path, ADAPTER_DIR)
            log(f"Rolled back adapter from {backup_path}", logging.WARNING)
        except Exception as exc:
            log(f"Adapter rollback failed: {exc}", logging.ERROR)

    # ------------------------------------------------------------------ #
    # Recipe
    # ------------------------------------------------------------------ #
    def plan(self, example_count: int) -> dict:
        """Derive the iteration count and evaluation cadence from the corpus.

        At batch size 1 a fixed iteration count means the number of epochs is
        set by however much unrelated data is in the corpus, which is the wrong
        thing to hold constant. Hold epochs constant instead.
        """
        cfg = self.config
        batch = max(1, cfg.train_batch_size)
        if cfg.train_iters:
            iters = cfg.train_iters             # explicit pin, already clamped
        else:
            steps_per_epoch = max(1, -(-example_count // batch))   # ceil
            iters = int(round(steps_per_epoch * cfg.train_epochs))
        iters = min(cfg.train_max_iters, max(cfg.train_min_iters, iters))
        # Five evaluation points is enough to see the curve turn without paying
        # for a full validation pass every few steps. Checkpoints are pinned to
        # the same cadence so every evaluated iteration has a file to promote.
        every = max(1, iters // 5)
        return {"iters": iters, "batch_size": batch, "eval_every": every,
                "save_every": every, "epochs_effective": round(iters * batch / max(1, example_count), 2)}

    def _write_lora_config(self) -> Path | None:
        """Write the LoRA shape mlx-lm only accepts through a config file."""
        if "--config" not in self._lora_help:
            return None
        cfg = self.config
        path = SFT_DIR / "lora_config.yaml"
        # Hand-written rather than via pyyaml: three scalars under one key needs
        # no dependency, and mlx-lm's own example config is this shape.
        path.write_text(
            "# Generated by the retrain loop. Edits are overwritten on the next run;\n"
            "# change TRAIN_LORA_RANK / TRAIN_LORA_SCALE / TRAIN_LORA_DROPOUT instead.\n"
            "lora_parameters:\n"
            f"  rank: {cfg.train_lora_rank}\n"
            f"  scale: {float(cfg.train_lora_scale)}\n"
            f"  dropout: {float(cfg.train_lora_dropout)}\n",
            encoding="utf-8")
        return path

    def _build_cmd(self, plan: dict | None = None) -> list[str]:
        plan = plan or self.plan(max(1, self.last_export.get("train", 1)))
        cmd = [sys.executable, "-m", "mlx_lm.lora"]
        if not add_if_supported(cmd, self._lora_help, ["--model", "--hf-path", "--mlx-path"], self.config.model):
            cmd.extend(["--model", self.config.model])
        if not add_if_supported(cmd, self._lora_help, ["--train"]):
            cmd.append("--train")
        if not add_if_supported(cmd, self._lora_help, ["--data"], str(SFT_DIR)):
            cmd.extend(["--data", str(SFT_DIR)])
        if not add_if_supported(cmd, self._lora_help, ["--adapter-path", "--adapter"], str(ADAPTER_DIR)):
            cmd.extend(["--adapter-path", str(ADAPTER_DIR)])

        add_if_supported(cmd, self._lora_help, ["--iters", "--iterations"], str(plan["iters"]))
        add_if_supported(cmd, self._lora_help, ["--batch-size"], str(plan["batch_size"]))
        # Fine-tune method: lora (default, 8GB-friendly), dora, or full. Newer
        # mlx-lm exposes --fine-tune-type; on builds that do not, the flag is
        # skipped and it trains LoRA, which is the safe default anyway.
        ft = (self.config.train_fine_tune_type or "lora").lower()
        if ft in ("lora", "dora", "full"):
            add_if_supported(cmd, self._lora_help, ["--fine-tune-type", "--train-type"], ft)
        # For full fine-tuning the user sets TRAIN_NUM_LAYERS=-1 to tune all
        # layers; mlx-lm reads -1 as "all". LoRA keeps the top-N default.
        add_if_supported(cmd, self._lora_help, ["--num-layers"], str(self.config.train_num_layers))
        add_if_supported(cmd, self._lora_help, ["--learning-rate", "-lr"], self.config.train_lr)
        add_if_supported(cmd, self._lora_help, ["--grad-checkpoint", "--gradient-checkpoint"])
        add_if_supported(cmd, self._lora_help, ["--max-seq-length", "--seq-length", "--max-seq-len"], self.config.train_seq_len)
        # Evaluation and checkpointing on the same cadence: the val-loss gate
        # below needs a checkpoint file for every iteration it scored.
        add_if_supported(cmd, self._lora_help, ["--steps-per-eval"], str(plan["eval_every"]))
        add_if_supported(cmd, self._lora_help, ["--save-every", "--steps-per-save"], str(plan["save_every"]))
        # -1 is mlx-lm's "use the whole validation set", which is what we want on
        # a set this small: a sampled subset makes the gate noisy.
        add_if_supported(cmd, self._lora_help, ["--val-batches"], "-1")
        lora_config = self._write_lora_config()
        if lora_config is not None:
            cmd.extend(["--config", str(lora_config)])
        return cmd

    # ------------------------------------------------------------------ #
    # Corpus
    # ------------------------------------------------------------------ #
    def _example(self, user_prompt: str, assistant: str, system: str) -> dict:
        return {"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant},
        ]}

    def _fits(self, example: dict) -> bool:
        """Whether an example survives the sequence window intact.

        mlx-lm truncates an over-long sequence rather than dropping it, so an
        example that does not fit does not merely go to waste: it teaches the
        model to stop mid-answer. Estimated, not tokenized — pulling a tokenizer
        into the web process would mean a second copy of the model in RAM.
        """
        if not self.config.train_drop_over_length:
            return True
        try:
            budget = int(str(self.config.train_seq_len).strip())
        except (TypeError, ValueError):
            return True
        return messages_tokens(example["messages"]) <= budget

    def _warn_if_window_is_mostly_prompt(self, system: str) -> None:
        """Flag a sequence window the system prompt has already eaten.

        Every example carries the full system prompt, so the window has to hold
        it plus a question plus a complete answer. The shipped prompt is ~270
        tokens; against the old 512 default that left room for roughly a
        paragraph, and mlx-lm spends the overflow by cutting the answer.
        """
        try:
            budget = int(str(self.config.train_seq_len).strip())
        except (TypeError, ValueError):
            return
        overhead = messages_tokens([{"role": "system", "content": system}])
        if overhead * 3 > budget:
            log(f"The system prompt is ~{overhead} tokens of a TRAIN_SEQ_LEN of "
                f"{budget}, leaving ~{max(0, budget - overhead)} for the question "
                f"and the whole answer. Raise TRAIN_SEQ_LEN or shorten SYSTEM_PROMPT, "
                f"or most answers will be dropped as over-length.", logging.WARNING)

    def export_tool_traces(self, limit: int | None = None) -> list[dict]:
        """Turn logged tool calls into supervised examples of the call format.

        Small models fail at format adherence far more than at reasoning, and
        format adherence is what the loop depends on: a run dies when the model
        emits {"tool": "search"} instead of this app's schema, not when it
        misremembers a date. These rows are the app's own schema, in the app's
        own wording, which is exactly the supervision that is missing.

        They are also the model's own output being fed back as ground truth, so
        two filters apply. `error IS NULL` alone means "did not raise", not "was
        correct"; with train_tool_quality="rated" the conversation must also
        contain an answer a human approved for training. And the caller caps the
        count relative to the human rows, because unfiltered self-distillation
        compounds whatever tool-selection bias the model already has.
        """
        if not self.config.train_on_tool_calls:
            return []
        if limit is None:
            limit = self.config.train_tool_examples
        if limit <= 0:
            return []

        # The newest user message at or before the call is the question that
        # provoked it.
        question_join = (
            "LEFT JOIN messages m "
            "  ON m.conversation_id = t.conversation_id AND m.role = 'user' "
            "  AND m.id = (SELECT MAX(id) FROM messages m2 "
            "              WHERE m2.conversation_id = t.conversation_id "
            "                AND m2.role = 'user' AND m2.created_at <= t.created_at) "
        )
        where = ["t.error IS NULL", "t.conversation_id IS NOT NULL",
                 "t.result IS NOT NULL", "TRIM(t.result) != ''"]
        if self.config.train_tool_quality == "rated":
            # feedback has no conversation id (session_id is a per-click uuid),
            # so the join is on the question text: some user turn in this
            # conversation was answered well enough to be approved, and no turn
            # in it was rated down.
            where.append(
                "EXISTS (SELECT 1 FROM messages mq JOIN feedback f "
                "        ON f.user_prompt = mq.content "
                "        WHERE mq.conversation_id = t.conversation_id "
                "          AND mq.role = 'user' "
                "          AND f.approved_for_training = 1 AND f.rating > 0)")
            where.append(
                "NOT EXISTS (SELECT 1 FROM messages mb JOIN feedback fb "
                "            ON fb.user_prompt = mb.content "
                "            WHERE mb.conversation_id = t.conversation_id "
                "              AND mb.role = 'user' AND fb.rating < 0)")

        rows = self.db.execute(
            "SELECT t.name, t.args, m.content AS question "
            "FROM tool_calls t " + question_join +
            "WHERE " + " AND ".join(where) +
            " ORDER BY t.id DESC LIMIT ?",
            (limit,),
        ).fetchall()

        examples: list[dict] = []
        seen: set[tuple] = set()
        for row in rows:
            question = (row["question"] or "").strip()
            if not question:
                continue
            try:
                args = json.loads(row["args"] or "{}")
            except Exception:
                continue
            call = json.dumps({"tool": row["name"], "args": args},
                              ensure_ascii=False, sort_keys=True)
            key = (question, call)
            if key in seen:
                continue
            seen.add(key)
            example = self._example(question, call, TOOL_TRAINING_PREAMBLE)
            if self._fits(example):
                examples.append(example)
        return examples

    def _seed_replay_file(self) -> None:
        """Write a starter rehearsal set if the user has not supplied one."""
        if REPLAY_FILE.exists():
            return
        REPLAY_FILE.parent.mkdir(parents=True, exist_ok=True)
        system = self.config.system_prompt_with_identity
        lines = [json.dumps(self._example(q, a, system), ensure_ascii=False)
                 for q, a in DEFAULT_REPLAY_EXAMPLES]
        REPLAY_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log(f"Seeded a starter rehearsal set at {REPLAY_FILE} "
            f"({len(lines)} examples). Edit it to match your own general use.")

    def load_replay(self, wanted: int, rng: random.Random) -> list[dict]:
        """Sample rehearsal examples to mix into the training set.

        Rehearsal is the standard defence against catastrophic forgetting: a
        share of general examples carried through every fine-tune so the adapter
        cannot collapse onto a few dozen feedback rows. It is sampled and then
        shuffled into the corpus rather than appended, because a block of it at
        the end is a second mini-finetune, not rehearsal.
        """
        if wanted <= 0:
            return []
        self._seed_replay_file()
        try:
            raw = REPLAY_FILE.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            log(f"Rehearsal set unreadable ({exc}); training without it. "
                f"The adapter is more likely to forget general behaviour.", logging.WARNING)
            return []

        pool: list[dict] = []
        for line in raw:
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except ValueError:
                continue
            if isinstance(example, dict) and isinstance(example.get("messages"), list) \
                    and self._fits(example):
                pool.append(example)
        if not pool:
            log(f"No usable rehearsal examples in {REPLAY_FILE}; training without them.",
                logging.WARNING)
            return []
        if wanted <= len(pool):
            return rng.sample(pool, wanted)
        # Fewer on file than the ratio asks for: repeat the pool rather than
        # silently under-weighting the anchor.
        out = list(pool)
        while len(out) < wanted:
            out.extend(rng.sample(pool, min(len(pool), wanted - len(out))))
        return out[:wanted]

    def export_feedback(self) -> tuple[int, list[int]]:
        """Write train/valid JSONL. Returns (example count, contributing feedback ids).

        The count is the whole mixed corpus; self.last_export carries the
        breakdown, and it is the human-approved figure in there — not this total
        — that the minimum-examples gate is applied to.
        """
        rows = self.db.execute("""
            SELECT id, user_prompt, assistant_response, corrected_response, rating
            FROM feedback WHERE approved_for_training = 1
            ORDER BY id
        """).fetchall()

        cfg = self.config
        system = cfg.system_prompt_with_identity
        self._warn_if_window_is_mostly_prompt(system)
        feedback_examples: list[dict] = []
        exported_ids: list[int] = []
        dropped_long = 0
        seen = set()
        for row in rows:
            user_prompt = (row["user_prompt"] or "").strip()
            assistant_response = (row["corrected_response"] or row["assistant_response"] or "").strip()
            if not user_prompt or not assistant_response:
                continue
            key = (user_prompt, assistant_response)
            if key in seen:
                # Duplicate content still counts as consumed, or it retriggers forever.
                exported_ids.append(row["id"])
                continue
            seen.add(key)
            example = self._example(user_prompt, assistant_response, system)
            if not self._fits(example):
                # Not marked as consumed: raising TRAIN_SEQ_LEN should bring it
                # back rather than have it silently skipped forever.
                dropped_long += 1
                continue
            exported_ids.append(row["id"])
            feedback_examples.append(example)

        # Tool traces are capped as a multiple of the human rows, so the signal
        # the whole feedback UI exists to collect cannot be a rounding error in
        # the gradient.
        tool_cap = cfg.train_tool_examples
        if cfg.train_tool_ratio > 0:
            tool_cap = min(tool_cap, int(len(feedback_examples) * cfg.train_tool_ratio))
        tool_examples = self.export_tool_traces(tool_cap) if tool_cap > 0 else []
        if tool_examples:
            log(f"Adding {len(tool_examples)} tool-call examples "
                f"(cap {tool_cap}, quality={cfg.train_tool_quality}).")

        examples = feedback_examples + tool_examples
        if not examples:
            self.last_export = {"feedback": 0, "tool": 0, "replay": 0, "total": 0,
                                "train": 0, "valid": 0, "dropped_long": dropped_long}
            if dropped_long:
                log(f"{dropped_long} approved example(s) exceed TRAIN_SEQ_LEN="
                    f"{cfg.train_seq_len} and were dropped rather than truncated.",
                    logging.WARNING)
            return 0, []

        # A private Random keeps the split reproducible without reseeding the
        # process-wide generator, which every other caller of random shares.
        rng = random.Random(42)

        # replay_ratio is a share of the FINAL mixed set, so solve for it rather
        # than taking the share of the new data.
        replay_examples: list[dict] = []
        if cfg.train_replay_ratio > 0:
            wanted = int(round(len(examples) * cfg.train_replay_ratio
                               / max(1e-6, 1.0 - cfg.train_replay_ratio)))
            replay_examples = self.load_replay(wanted, rng)
            if replay_examples:
                log(f"Mixing in {len(replay_examples)} rehearsal examples "
                    f"({cfg.train_replay_ratio:.0%} of the final set).")

        examples.extend(replay_examples)
        rng.shuffle(examples)

        # Hold out a real slice. Below the point where a split can be honest,
        # take one example out rather than putting the same row in both files:
        # validating on a row that was trained on measures nothing.
        holdout = max(1, int(round(len(examples) * cfg.train_val_split)))
        holdout = min(holdout, max(0, len(examples) - 1))
        split = len(examples) - holdout
        train_examples = examples[:split]
        valid_examples = examples[split:]

        SFT_DIR.mkdir(parents=True, exist_ok=True)
        for name, chunk in (("train.jsonl", train_examples),
                            ("valid.jsonl", valid_examples),
                            # mlx-lm's --test reads this; same held-out rows, so
                            # a manual `mlx_lm.lora --test` reproduces the gate.
                            ("test.jsonl", valid_examples)):
            text = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in chunk)
            (SFT_DIR / name).write_text(text, encoding="utf-8")

        if dropped_long:
            log(f"{dropped_long} approved example(s) exceed TRAIN_SEQ_LEN="
                f"{cfg.train_seq_len} and were dropped rather than truncated.",
                logging.WARNING)

        self.last_export = {
            "feedback": len(feedback_examples),
            "tool": len(tool_examples),
            "replay": len(replay_examples),
            "total": len(examples),
            "train": len(train_examples),
            "valid": len(valid_examples),
            "dropped_long": dropped_long,
        }
        return len(examples), exported_ids

    # ------------------------------------------------------------------ #
    # Judging the run
    # ------------------------------------------------------------------ #
    @staticmethod
    def parse_val_losses(text: str) -> list[tuple[int, float]]:
        """(iteration, validation loss) for every evaluation point in a run's log."""
        return [(int(it), float(loss)) for it, loss in _VAL_LOSS_RE.findall(text)]

    def _promote_best(self, losses: list[tuple[int, float]]) -> tuple[int, float] | None:
        """Make the best-scoring checkpoint the live adapter. Returns what was promoted."""
        if not self.config.train_promote_best or len(losses) < 2:
            return None
        # Ordered by loss then iteration, so a tie resolves to the earlier
        # checkpoint — the less-trained of two equals is the safer one.
        best_loss, best_iter = min((loss, it) for it, loss in losses[1:])
        final_iter = losses[-1][0]
        if best_iter == final_iter:
            return None                      # the last one already is the best
        checkpoints = dict(adapter_checkpoints(ADAPTER_DIR))
        source = checkpoints.get(best_iter)
        if source is None:
            log(f"Best validation loss was at iteration {best_iter} but no "
                f"checkpoint was written for it; keeping the final adapter.",
                logging.WARNING)
            return None
        try:
            shutil.copyfile(source, ADAPTER_DIR / "adapters.safetensors")
        except OSError as exc:
            log(f"Could not promote checkpoint {source}: {exc}", logging.WARNING)
            return None
        log(f"Promoted the iteration-{best_iter} checkpoint "
            f"(val loss {best_loss:.4f}) over the final one.")
        return best_iter, best_loss

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #
    def run(self, trigger: str = "manual") -> None:
        if not self.lock.acquire(blocking=False):
            return

        backup_path: Path | None = None
        promoted = False
        try:
            self.status = {"running": True, "message": f"Retraining started from {trigger}"}
            cfg = self.config

            # An empty help string means the trainer could not be interrogated:
            # every optional flag would then be silently dropped and mlx-lm would
            # run on its own defaults (1000 iters, lr 1e-5, seq 2048), which is
            # not the recipe anyone configured. Refuse rather than train blind.
            if not self._lora_help:
                self._lora_help = help_cmd("mlx_lm.lora")     # may have been installed since
            if not self._lora_help:
                self.status = {"running": False, "message":
                    "Cannot read `mlx_lm.lora --help`, so the training recipe "
                    "(iterations, learning rate, sequence length) would be ignored. "
                    "Install or repair mlx-lm and try again."}
                return

            count, exported_ids = self.export_feedback()
            stats = self.last_export
            approved = stats.get("feedback", 0)
            if count == 0:
                extra = ""
                if stats.get("dropped_long"):
                    extra = (f" {stats['dropped_long']} example(s) were dropped for "
                             f"exceeding TRAIN_SEQ_LEN={cfg.train_seq_len}.")
                self.status = {"running": False,
                               "message": "No approved feedback available for training." + extra}
                return
            if approved < cfg.train_min_examples:
                # Fine-tuning on a handful of examples overfits and degrades the
                # model everywhere else (catastrophic forgetting). Refuse rather
                # than ship a worse adapter; the threshold is configurable. The
                # count checked here is human-approved feedback only: tool traces
                # and rehearsal rows pad the corpus but are not the signal, and
                # counting them would let the gate pass on zero real feedback.
                self.status = {"running": False, "message":
                    f"Only {approved} approved example(s); need at least "
                    f"{cfg.train_min_examples} to train safely. "
                    "Collect more feedback (or lower TRAIN_MIN_EXAMPLES)."}
                return

            plan = self.plan(stats["train"] or count)
            previous = adapter_meta(ADAPTER_DIR)
            prompt_sha = hashlib.sha256(
                cfg.system_prompt_with_identity.encode("utf-8")).hexdigest()[:16]
            if previous.get("system_prompt_sha") and previous["system_prompt_sha"] != prompt_sha:
                # The examples are conditioned on the system prompt they were
                # exported with. Changing it after the fact leaves the adapter
                # tuned for a prompt the server no longer sends.
                log("The system prompt has changed since the last adapter was "
                    "trained; this run re-conditions on the current one.", logging.WARNING)

            backup_path = self._backup_adapter()
            # Not cleared before now: the backup above is what makes discarding
            # the previous run's files safe. Leaving them would let a failed run
            # leave a mix of two runs' checkpoints behind.
            clear_adapter_checkpoints(ADAPTER_DIR)

            self.status["message"] = (
                f"Exported {count} examples "
                f"({stats['feedback']} feedback, {stats['tool']} tool, {stats['replay']} rehearsal); "
                f"{plan['iters']} iterations. Stopping model server.")
            self.model_manager.stop()

            cmd = self._build_cmd(plan)
            train_log = LOG_DIR / "train.log"
            LOG_DIR.mkdir(parents=True, exist_ok=True)

            self.status["message"] = (f"Training LoRA adapter "
                                      f"({plan['iters']} iterations over {stats['train']} examples)...")

            with open(train_log, "a", encoding="utf-8") as lf:
                lf.write(f"\n\n{datetime.now(timezone.utc).isoformat()} Training command:\n{' '.join(cmd)}\n")
                lf.flush()
                # Where this run's output starts, so the val-loss gate below reads
                # only this run and not every run since the log was created.
                offset = lf.tell()
                try:
                    proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                          timeout=cfg.train_timeout)
                except subprocess.TimeoutExpired:
                    raise RuntimeError(
                        f"Training exceeded TRAIN_TIMEOUT ({cfg.train_timeout:.0f}s) and was "
                        f"killed. See logs/train.log.") from None
                if proc.returncode != 0:
                    raise RuntimeError(f"Training failed with exit code {proc.returncode}. See logs/train.log.")

            with open(train_log, "r", encoding="utf-8", errors="replace") as lf:
                lf.seek(offset)
                run_log = lf.read()
            losses = self.parse_val_losses(run_log)

            # mlx-lm measures the first validation loss before any weight update,
            # so losses[0] is this model's pre-training score on the held-out
            # rows and the comparison needs no extra evaluation pass.
            baseline = losses[0][1] if losses else None
            best = self._promote_best(losses)
            final = best[1] if best else (losses[-1][1] if losses else None)

            if cfg.train_val_check and len(losses) >= 2:
                ceiling = baseline * (1.0 + cfg.train_val_tolerance)
                if final > ceiling:
                    raise RuntimeError(
                        f"Held-out loss got worse ({baseline:.4f} -> {final:.4f}); "
                        f"the adapter was discarded. Collect more feedback, or lower "
                        f"TRAIN_EPOCHS / TRAIN_LR.")
            elif cfg.train_val_check:
                log("No validation losses were reported by the trainer, so the run "
                    "could not be checked for regression; the adapter was promoted "
                    "unverified.", logging.WARNING)

            clear_adapter_checkpoints(ADAPTER_DIR)
            write_adapter_base(ADAPTER_DIR, cfg.model)
            write_adapter_meta(ADAPTER_DIR, {
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "trigger": trigger,
                "model": cfg.model,
                "system_prompt_sha": prompt_sha,
                "counts": stats,
                "plan": plan,
                "learning_rate": cfg.train_lr,
                "seq_len": cfg.train_seq_len,
                "fine_tune_type": cfg.train_fine_tune_type,
                "num_layers": cfg.train_num_layers,
                "lora": {"rank": cfg.train_lora_rank, "scale": cfg.train_lora_scale,
                         "dropout": cfg.train_lora_dropout},
                "val_loss_baseline": baseline,
                "val_loss_final": final,
                "promoted_iteration": best[0] if best else plan["iters"],
            })
            promoted = True
            # Only now: a run that was rolled back must leave its rows untrained
            # so a later, larger run picks them up again.
            self.db.mark_trained(exported_ids)

            self.status["message"] = "Training complete. Restarting model server."
            self.model_manager.restart()

            delta = ""
            if baseline is not None and final is not None:
                delta = f" Held-out loss {baseline:.4f} -> {final:.4f}."
            self.status = {"running": False, "message":
                           f"Retraining complete on {count} examples "
                           f"({stats['feedback']} feedback, {stats['tool']} tool, "
                           f"{stats['replay']} rehearsal).{delta}"}

        except Exception as exc:
            log(f"Retrain failed: {exc}", logging.ERROR)
            if not promoted:
                self._rollback_adapter(backup_path)
            self.status = {"running": False, "message": f"Retrain error: {exc}"}
            try:
                self.model_manager.start()
            except Exception as restart_exc:
                self.status["message"] += f" Restart error: {restart_exc}"
        finally:
            self.lock.release()


# Re-exported explicitly: the original file was one flat namespace, so private
# helpers (leading underscore) must cross module boundaries too.
__all__ = [
    'DEFAULT_REPLAY_EXAMPLES',
    'RetrainManager',
]
