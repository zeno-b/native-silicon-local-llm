"""LoRA retraining from collected feedback.

Split out of the original single-file deploy.py; behaviour is unchanged.
"""

from __future__ import annotations

import json
import logging
import random
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


class RetrainManager:
    """Manages LoRA retraining with adapter backup/rollback."""

    def __init__(self, db: Database, model_manager: ModelServerManager, config: Config):
        self.db = db
        self.model_manager = model_manager
        self.config = config
        self.lock = threading.Lock()
        self.status = {"running": False, "message": "idle"}
        self._lora_help = help_cmd("mlx_lm.lora")

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
        shutil.copytree(ADAPTER_DIR, backup_path)
        log(f"Adapter backed up to {backup_path}")
        return backup_path

    def _rollback_adapter(self, backup_path: Path | None) -> None:
        """Restore the pre-training adapter after a failed run."""
        if backup_path is None or not backup_path.exists():
            return
        try:
            if ADAPTER_DIR.exists():
                shutil.rmtree(ADAPTER_DIR)
            shutil.copytree(backup_path, ADAPTER_DIR)
            log(f"Rolled back adapter from {backup_path}", logging.WARNING)
        except Exception as exc:
            log(f"Adapter rollback failed: {exc}", logging.ERROR)

    def _build_cmd(self) -> list[str]:
        cmd = [sys.executable, "-m", "mlx_lm.lora"]
        if not add_if_supported(cmd, self._lora_help, ["--model", "--hf-path", "--mlx-path"], self.config.model):
            cmd.extend(["--model", self.config.model])
        if not add_if_supported(cmd, self._lora_help, ["--train"]):
            cmd.append("--train")
        if not add_if_supported(cmd, self._lora_help, ["--data"], str(SFT_DIR)):
            cmd.extend(["--data", str(SFT_DIR)])
        if not add_if_supported(cmd, self._lora_help, ["--adapter-path", "--adapter"], str(ADAPTER_DIR)):
            cmd.extend(["--adapter-path", str(ADAPTER_DIR)])

        add_if_supported(cmd, self._lora_help, ["--iters", "--iterations"], str(self.config.train_iters))
        add_if_supported(cmd, self._lora_help, ["--batch-size"], "1")
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
        return cmd

    def export_tool_traces(self) -> list[dict]:
        """Turn logged tool calls into supervised examples of the call format.

        Small models fail at format adherence far more than at reasoning, and
        format adherence is what the loop depends on: a run dies when the model
        emits {"tool": "search"} instead of this app's schema, not when it
        misremembers a date. These rows are the app's own schema, in the app's
        own wording, which is exactly the supervision that is missing.
        """
        if not self.config.train_on_tool_calls:
            return []
        rows = self.db.execute(
            "SELECT t.name, t.args, t.error, m.content AS question "
            "FROM tool_calls t LEFT JOIN messages m "
            "  ON m.conversation_id = t.conversation_id AND m.role = 'user' "
            "  AND m.id = (SELECT MAX(id) FROM messages m2 "
            "              WHERE m2.conversation_id = t.conversation_id "
            "                AND m2.role = 'user' AND m2.created_at <= t.created_at) "
            "WHERE t.error IS NULL AND t.conversation_id IS NOT NULL "
            "ORDER BY t.id DESC LIMIT ?",
            (self.config.train_tool_examples,),
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
            examples.append({
                "messages": [
                    {"role": "system", "content": TOOL_TRAINING_PREAMBLE},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": call},
                ]
            })
        return examples

    def export_feedback(self) -> tuple[int, list[int]]:
        """Write train/valid JSONL. Returns (example count, contributing feedback ids)."""
        rows = self.db.execute("""
            SELECT id, user_prompt, assistant_response, corrected_response, rating
            FROM feedback WHERE approved_for_training = 1
            ORDER BY id
        """).fetchall()

        examples = []
        exported_ids: list[int] = []
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
            exported_ids.append(row["id"])
            examples.append({
                "messages": [
                    {"role": "system", "content": self.config.system_prompt_with_identity},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response},
                ]
            })

        tool_examples = self.export_tool_traces()
        if tool_examples:
            log(f"Adding {len(tool_examples)} tool-call examples to the training set.")
            examples.extend(tool_examples)

        if not examples:
            return 0, []

        # A private Random keeps the split reproducible without reseeding the
        # process-wide generator, which every other caller of random shares.
        rng = random.Random(42)
        rng.shuffle(examples)

        split = max(1, int(len(examples) * 0.9))
        train_examples = examples[:split]
        valid_examples = examples[split:] or examples[:1]

        train_path = SFT_DIR / "train.jsonl"
        valid_path = SFT_DIR / "valid.jsonl"

        train_text = "\n".join(json.dumps(x, ensure_ascii=False) for x in train_examples) + "\n"
        valid_text = "\n".join(json.dumps(x, ensure_ascii=False) for x in valid_examples) + "\n"

        train_path.write_text(train_text, encoding="utf-8")
        valid_path.write_text(valid_text, encoding="utf-8")

        return len(examples), exported_ids

    def run(self, trigger: str = "manual") -> None:
        if not self.lock.acquire(blocking=False):
            return

        backup_path: Path | None = None
        try:
            self.status = {"running": True, "message": f"Retraining started from {trigger}"}

            count, exported_ids = self.export_feedback()
            if count == 0:
                self.status = {"running": False, "message": "No approved feedback available for training."}
                return
            if count < self.config.train_min_examples:
                # Fine-tuning on a handful of examples overfits and degrades the
                # model everywhere else (catastrophic forgetting). Refuse rather
                # than ship a worse adapter; the threshold is configurable.
                self.status = {"running": False, "message":
                    f"Only {count} approved examples; need at least "
                    f"{self.config.train_min_examples} to train safely. "
                    "Collect more feedback (or lower TRAIN_MIN_EXAMPLES)."}
                return

            backup_path = self._backup_adapter()

            self.status["message"] = f"Exported {count} examples. Stopping model server."
            self.model_manager.stop()

            cmd = self._build_cmd()
            train_log = LOG_DIR / "train.log"

            self.status["message"] = "Training LoRA adapter..."

            with open(train_log, "a", encoding="utf-8") as lf:
                lf.write(f"\n\n{datetime.now(timezone.utc).isoformat()} Training command:\n{' '.join(cmd)}\n")
                lf.flush()

                proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
                if proc.returncode != 0:
                    raise RuntimeError(f"Training failed with exit code {proc.returncode}. See logs/train.log.")

            write_adapter_base(ADAPTER_DIR, self.config.model)
            self.db.mark_trained(exported_ids)

            self.status["message"] = "Training complete. Restarting model server."
            self.model_manager.restart()

            self.status = {"running": False, "message": f"Retraining complete on {count} examples."}

        except Exception as exc:
            log(f"Retrain failed: {exc}", logging.ERROR)
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
    'RetrainManager',
]
