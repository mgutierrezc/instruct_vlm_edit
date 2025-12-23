from .ft import Finetune
from .ft_retrain import Finetune_retrain
from .mend import MEND
from .mend_pretrain import MEND_Pretrain
from .grace import GRACE
from .balancedit import BalancEdit
from .balancedit_kv import BalancEditKV
from .ike import IKE
from .ike_cot import IKE_COT
# from .ike_clip_v1 import IKE_CLIP
from .ike_clip import IKE_CLIP
from .ike_tuple import IKE_TUPLE
from .ike_proto import IKE_PROTO
from .reasonedit import ReasonEdit


def get_editor(config, model):
    """
    Factory function to get editor based on config.
    
    Args:
        config: Configuration object with editor settings
        model: Model to edit
        device: Device to use
        
    Returns:
        Editor instance
    """
    device = config.device
    editor_name = getattr(
        config.editor, "_name", config.editor if hasattr(config, "editor") else None
    )
    
    if editor_name == "ft":
        editor = Finetune(config, model)
    elif editor_name == "ft_ewc":
        editor = Finetune_ewc(config, model)
    elif editor_name == "ft_retrain":
        editor = Finetune_retrain(config, model)
    elif editor_name == "mend":
        tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
        editor = MEND(config, model, tokenizer, device)
    elif editor_name == "mend_pretrain":
        tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
        checkpoint = getattr(config.editor, 'checkpoint_path', None)
        editor = MEND_Pretrain(config, model, tokenizer, device, checkpoint)
    elif editor_name == "grace":
        editor = GRACE(config, model)
    elif editor_name == "rome":
        editor = ROME(config, model)
    elif editor_name == "memory":
        editor = MemoryNetwork(config, model)
    elif editor_name == "defer":
        editor = Defer(config, model)
    elif editor_name == "balancedit":
        editor = BalancEdit(config, model)
    elif editor_name == "balancedit_kv":
        editor = BalancEditKV(config, model)
    elif editor_name == "ike":
        editor = IKE(config, model)
    elif editor_name == "ike_cot":
        editor = IKE_COT(config, model)
    elif editor_name == "ike_clip":
        editor = IKE_CLIP(config, model)
    elif editor_name == "ike_tuple":
        editor = IKE_TUPLE(config, model)
    elif editor_name == "ike_proto":
        editor = IKE_PROTO(config, model)
    elif editor_name == "reasonedit":
        editor = ReasonEdit(config, model)
    else:
        raise ValueError(f"Unknown editor: {editor_name}")
    
    return editor

