from .ft import Finetune
from .ft_ewc import Finetune_ewc
from .ft_retrain import Finetune_retrain
from .mend import MEND
from .grace import GRACE
from .rome_base import ROME
from .defer import Defer
from .memory import MemoryNetwork
from .balancedit import BalancEdit
from .ike import IKE


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
    editor_name = getattr(config.editor, "_name", config.editor if hasattr(config, "editor") else None)
    
    if editor_name == "ft":
        editor = Finetune(config, model)
    elif editor_name == "ft_ewc":
        editor = Finetune_ewc(config, model)
    elif editor_name == "ft_retrain":
        editor = Finetune_retrain(config, model)
    elif editor_name == "mend":
        tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else None
        editor = MEND(config, model, tokenizer, device)
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
    elif editor_name == "ike":
        editor = IKE(config, model)
    else:
        raise ValueError(f"Unknown editor: {editor_name}")
    
    return editor

