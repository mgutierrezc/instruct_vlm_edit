import torch
from PIL import Image as PILImage
from torchvision import transforms as T


class Augmenter:
    """Online augmentation for images, questions, and rationales."""

    def __init__(self, wrapper=None):
        self.wrapper = wrapper
        self.img_aug = T.Compose([
            T.RandomResizedCrop(size=(384, 384), scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        ])
        self._blank = PILImage.new("RGB", (364, 364), color="black")

    def image(self, img):
        """Apply random image augmentations."""
        if isinstance(img, str):
            img = PILImage.open(img).convert("RGB")
        elif hasattr(img, "convert"):
            img = img.convert("RGB")
        return self.img_aug(img)
        # return img

    def question(self, q):
        """Rephrase question using VLM."""
        if not self.wrapper or not q:
            return q
        prompt = f"Rephrase this question differently while keeping the same meaning:\n\n{q}\n\nRephrased:"
        try:
            out = self.wrapper.generate([self._blank], [prompt], max_new_tokens=64, temperature=0.7)[0]
            out = str(out).strip()
            return out if out else q
        except Exception:
            return q

    def rationale(self, sent):
        """Rephrase rationale sentence using VLM."""
        if not self.wrapper or not sent:
            return sent
        prompt = f"Rephrase this fact differently while keeping the same meaning:\n\n{sent}\n\nRephrased:"
        try:
            out = self.wrapper.generate([self._blank], [prompt], max_new_tokens=64, temperature=0.7)[0]
            out = str(out).strip()
            return out if out else sent
        except Exception:
            return sent


def get_inner_params(named_parameters, inner_names):
    """Get parameters by name"""
    param_dict = dict(named_parameters)
    return [(n, param_dict[n]) for n in inner_names if n in param_dict]


def param_subset(named_parameters, inner_names):
    """Get subset of parameters"""
    param_dict = dict(named_parameters)
    return [param_dict[n] for n in inner_names if n in param_dict]


def parent_module(model, pname):
    """Get parent module for a parameter name"""
    components = pname.split('.')
    parent = model
    for component in components[:-1]:
        if hasattr(parent, component):
            parent = getattr(parent, component)
        elif component.isdigit():
            parent = parent[int(component)]
        else:
            raise RuntimeError(f"Couldn't find child module {component}")
    if not hasattr(parent, components[-1]):
        raise RuntimeError(f"Couldn't find child module {components[-1]}")
    return parent


def brackets_to_periods(name):
    """Convert brackets to periods in parameter names"""
    return name.replace("[", ".").replace("]", "")


def linear_backward_hook(mod, grad_in, grad_out):
    """Hook for capturing gradients in MEND"""
    if not hasattr(mod, "weight"):
        return
    if hasattr(mod.weight, "__x__"):
        assert len(grad_out) == 1
        mod.weight.__delta__ = grad_out[0].detach()


def linear_forward_hook(mod, activations, output):
    """Hook for capturing activations in MEND"""
    assert len(activations) == 1
    mod.weight.__x__ = activations[0].detach()


def hook_model(model, pnames):
    """Add forward and backward hooks to model for MEND"""
    handles = []
    for pname in pnames:
        parent = parent_module(model, pname)
        handles.append(parent.register_forward_hook(linear_forward_hook))
        handles.append(parent.register_full_backward_hook(linear_backward_hook))
    model.handles = handles


def explore_layers(model, top_k=10):
    """Find suggested layer candidates for finetuning.
    
    This function searches for layers that are commonly used for finetuning,
    such as language model heads, attention projections, and MLP layers.
    
    Args:
        model: PyTorch model to explore
        top_k: Maximum number of suggestions to return
        
    Returns:
        list: List of parameter names that match finetuning keywords
    """
    all_names = [n for n, p in model.named_parameters()]
    keywords = ['lm_head', 'embed_out', 'output', 'classifier', 'head', 
                'self_attn.q_proj', 'self_attn.v_proj', 'self_attn.k_proj',
                'mlp.c_fc', 'mlp.c_proj', 'gate_proj', 'up_proj', 'down_proj']
    
    suggestions = []
    for name in all_names:
        for kw in keywords:
            if kw in name.lower():
                suggestions.append(name)
                break
    
    return suggestions[:top_k] if suggestions else all_names[:1]


def validate_and_correct_param_name(model, param_name, logger=None):
    """Validate parameter name exists, try to correct if not found.
    
    This is useful for handling variations in parameter names across different
    model versions or HuggingFace implementations.
    
    Args:
        model: PyTorch model to search
        param_name: Parameter name to validate (may include brackets)
        logger: Optional logger for warnings/info (if None, uses print)
        
    Returns:
        str: Valid parameter name (original if exists, corrected if found, fallback otherwise)
    """
    param_name = brackets_to_periods(param_name)
    model_params = dict(model.named_parameters())
    
    # Check if parameter exists
    if param_name in model_params:
        return param_name
    
    # Try fuzzy matching by keywords
    log_msg = f"Parameter '{param_name}' not found in model. Attempting to find match..."
    if logger:
        logger.warning(log_msg)
    else:
        print(f"WARNING: {log_msg}")
    
    layer_parts = param_name.split('.')
    key_parts = [p for p in layer_parts if p and not p.isdigit()]
    
    # Match by last 3 non-numeric parts
    if len(key_parts) >= 3:
        matches = [n for n in model_params.keys() 
                   if all(part.lower() in n.lower() for part in key_parts[-3:])]
        if matches:
            corrected = matches[0]
            log_msg = f"  Found corrected layer: {corrected}"
            if logger:
                logger.info(log_msg)
            else:
                print(log_msg)
            return corrected
    
    # Fallback: use first available parameter
    fallback = list(model_params.keys())[0]
    log_msg = f"  No match found. Using fallback: {fallback}"
    if logger:
        logger.warning(log_msg)
    else:
        print(f"WARNING: {log_msg}")
    return fallback


