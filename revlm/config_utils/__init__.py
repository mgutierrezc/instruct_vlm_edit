import os
import yaml
from .utils import *

def configure_args(args, config_path=None):
    """Load config.yaml, apply simple CLI overrides, return NestedConfig.
    Sticks to editor/model_name/inner_params/dataset_name.
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.yaml")

    cfg = load_yaml(config_path)

    # Extract sections (default empty dicts if missing)
    model = dict(cfg.get("model", {}))
    experiment = dict(cfg.get("experiment", {}))

    # Simple editor handling: take CLI editor if provided, else YAML
    cli_editor = getattr(args, "editor", None)
    editor = cfg.get("editor", None)
    if cli_editor is not None:
        editor = {"_name": cli_editor}
    elif isinstance(editor, str):
        editor = {"_name": editor}
    elif isinstance(editor, dict):
        editor = {"_name": editor.get("_name", editor.get("name", ""))}
    else:
        editor = {}

    # Optional: load and merge editor preset YAML by name
    editor_name = editor.get("_name")
    if editor_name:
        preset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "editor", f"{editor_name}.yaml")
        preset = load_yaml(preset_path)
        if preset:
            merged = {k: v for k, v in preset.items() if k != "_name"}
            editor = {"_name": editor_name, **merged}

    # Map short model name if provided and load model preset
    model_name_provided = getattr(args, "model_name", None)
    if model_name_provided:
        short_to_full = {
            "qwen3": "Qwen/Qwen3-VL-8B-Instruct",
            "llava": "llava-hf/llava-1.5-7b-hf",
            "blip": "Salesforce/instructblip-vicuna-7b",
        }
        key = str(model_name_provided).lower()
        resolved_name = short_to_full.get(key, model_name_provided)
        
        # Try to load model preset YAML (similar to editor presets)
        model_preset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "model", f"{key}.yaml")
        model_preset = load_yaml(model_preset_path)
        if model_preset:
            # Merge preset into model config (preset values override defaults)
            for k, v in model_preset.items():
                if k not in model or model[k] is None or model[k] == []:
                    model[k] = v
            # Ensure name is set to resolved full name
            model["name"] = resolved_name
        else:
            # No preset found, just set the name
            model["name"] = resolved_name

    # CLI overrides
    if getattr(args, "inner_params", None):
        model["inner_params"] = args.inner_params
    if getattr(args, "dataset_name", None):
        experiment["dataset_name"] = args.dataset_name

    # create result saving dir
    editor_tag = editor.get("_name") or "raw"
    model_tag = (model.get("name", "").split("/")[-1] or "model").replace(" ", "_")
    dataset_tag = (experiment.get("dataset_name", "dataset") or "dataset").replace(" ", "_")
    res_dir = os.path.join("results", editor_tag, model_tag, dataset_tag)
    os.makedirs(res_dir, exist_ok=True)

    nested = {
        "batch_size": cfg.get("batch_size", 1),
        "n_iter": cfg.get("n_iter", 100),
        "max_n_edits": cfg.get("max_n_edits", 5000),
        "seed": cfg.get("seed", 42),
        "device": cfg.get("device", "cuda"),
        "ckpt_dir": cfg.get("ckpt_dir", None),
        "dropout": cfg.get("dropout", None),
        "res_dir": res_dir,
        "model": model,
        "editor": editor,
        "experiment": experiment,
    }

    return NestedConfig(**to_ns(nested).__dict__)


def build_args_from_yaml(config_path: str) -> NestedConfig:
    """Return a NestedConfig from a YAML file (for notebooks)."""
    cfg = load_yaml(config_path)
    # Coerce editor string to nested
    if isinstance(cfg.get("editor"), str):
        cfg["editor"] = {"_name": cfg["editor"]}
    cfg.setdefault("model", {})
    cfg.setdefault("experiment", {})
    nested = {
        "batch_size": cfg.get("batch_size", 1),
        "n_iter": cfg.get("n_iter", 100),
        "max_n_edits": cfg.get("max_n_edits", 5000),
        "seed": cfg.get("seed", 42),
        "device": cfg.get("device", "cuda"),
        "ckpt_dir": cfg.get("ckpt_dir", None),
        "model": cfg.get("model", {}),
        "editor": cfg.get("editor", {}),
        "experiment": cfg.get("experiment", {}),
    }
    return NestedConfig(**to_ns(nested).__dict__)
