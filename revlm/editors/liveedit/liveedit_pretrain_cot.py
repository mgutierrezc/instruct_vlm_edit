# -*- coding: utf-8 -*-
"""
LiveEdit Pretraining with COT - Train meta-learners with sentence-level experts.

Extends LiveEditPretrain to also train on rationale sentences, similar to
how LiveEdit_COT extends LiveEdit.
"""

import re
import torch
from typing import Dict, List, Optional
from PIL import Image

from .liveedit_pretrain import LiveEditPretrain


class LiveEditPretrainCOT(LiveEditPretrain):
    """
    LiveEdit pretraining with COT sentence experts.
    
    During pretraining, each edit also trains on its rationale sentences:
    - Main edit: question → answer
    - Sentence 1: sentence → sentence (autoencoding)
    - Sentence 2: sentence → sentence
    - ...
    """
    
    def __init__(self, config, model):
        super().__init__(config, model)
        self.wrapper = model
    
    def pretrain_step_cot(
        self,
        edit_tokens: Dict[str, torch.Tensor],
        gen_tokens_list: List[Dict[str, torch.Tensor]],
        loc_tokens: Optional[Dict[str, torch.Tensor]],
        image: str,
        cot: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Pretrain step with COT sentence experts.
        
        Args:
            edit_tokens: Tokenized edit example
            gen_tokens_list: Tokenized generality samples
            loc_tokens: Tokenized locality sample
            image: Image path for sentence batches
            cot: Chain-of-thought rationale string
        
        Returns:
            Dict with losses
        """
        # 1. Standard pretrain step on edit + gen + loc
        losses = self.pretrain_step(edit_tokens, gen_tokens_list, loc_tokens)
        
        # 2. If no COT, return standard losses
        if not cot or not image:
            return losses
        
        # 3. Train on each sentence in COT
        sentences = self._split_sentences(cot)
        sentence_losses = []
        
        for sent in sentences:
            if len(sent.split()) < 3:  # Skip very short sentences
                continue
            
            # Create sentence → sentence batch
            try:
                img = Image.open(image).convert("RGB") if isinstance(image, str) else image
                sent_tokens = self.wrapper.prepare_training_batch({
                    "images": [img],
                    "prompts": [sent],
                    "golds": [{"label": sent, "label_train": sent}],
                    "idxs": [0],
                })
            except Exception as e:
                print(f"[liveedit_pretrain_cot] Skip sentence: {e}")
                continue
            
            # Get mid-layer reps and generate LoRA
            with torch.no_grad():
                mid_reps = self._get_mid_layer_reps(sent_tokens)
            
            if mid_reps is None:
                continue
            
            mid_reps_f32 = mid_reps.float()
            moe_c = self.moegen_c(mid_reps_f32).squeeze(0)
            moe_r = self.moegen_r(mid_reps_f32).squeeze(0)
            
            self.is_training_edit = True
            self.current_expert = (moe_c, moe_r)
            
            sent_loss = self._compute_ce_loss(sent_tokens)
            sentence_losses.append(sent_loss)
            
            self.is_training_edit = False
            self.current_expert = None
        
        # 4. Add sentence losses to total
        if sentence_losses:
            loss_sent = torch.stack(sentence_losses).mean()
            losses["loss_sent"] = loss_sent
            losses["loss"] = losses["loss"] + loss_sent
        else:
            losses["loss_sent"] = torch.tensor(0.0, device=self.device)
        
        return losses
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
