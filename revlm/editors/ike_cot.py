"""IKE_COT: IKE with rationale sentences in corpus.

Extends IKE by also storing each rationale sentence as a separate entry.
"""

import re
from typing import Any

from .ike import IKE


class IKE_COT(IKE):
    """IKE extended with rationale sentences."""

    def edit(self, config, tokens=None, batch_history=None, edit_ds=None, train_ds=None):
        """Add edits to corpus, including rationale sentences."""
        if edit_ds is None:
            return self.model

        data = getattr(edit_ds, "data", [])
        n_added = 0
        n_sentences = 0

        for ex in data:
            uid = ex.get("uid") or (ex.get("image"), ex.get("question"))
            prompt = ex.get("prompt_orig") or ex.get("prompt", "")
            target = ex.get("gold", {}).get("label", "")

            # 1. Add main (prompt, target) entry
            if prompt and target and self._add_edit(prompt, target, uid):
                n_added += 1

            # 2. Add rationale sentences (sentence, "") entries
            rationale = ex.get("cot") or ex.get("rationale", "")
            if rationale:
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', rationale.strip()) if s.strip()]
                for i, sent in enumerate(sentences):
                    sent_uid = f"{uid}_sent_{i}"
                    if self._add_edit(sent, "", sent_uid):
                        n_sentences += 1

        print(f"[IKE_COT] +{n_added} edits, +{n_sentences} sentences, corpus={len(self.corpus_sentences)}", flush=True)
        return self.model
