
def biased_run(hydra_config: DictConfig):
    from argparse import Namespace
    from pathlib import Path
    import random
    import time
    import shutil
    import torch
    import json
    import argparse

    from revlm.config_utils import configure_args 
    from revlm.run.edit import run_edit 

    # Repo setup
    repo_root = hydra_config.repo_root # "/home/sgw3fy/jobs/model_editing_jobs/repos/instruct_vlm_edit"
    os.chdir(repo_root)
    if repo_root not in sys.path:
        sys.path.append(repo_root)

    biases_path = hydra_config.biases_path
    biases_name = os.path.splitext(os.path.basename(biases_path))[0]
    subsample_path = hydra_config.subsample_path # "/home/sgw3fy/jobs/model_editing_jobs/repos/instruct_vlm_edit/aux_files/subsample_edits/aokvqa.json"
    namespace_config_path = hydra_config.namespace_config_path 

    # Map short model names to full HF model names used in results/pred/
    MODEL_NAME_MAP = {
        "qwen3": "Qwen3-VL-8B-Instruct",
        "qwen3_4b": "Qwen3-VL-4B-Instruct",
        "llava": "llava-1.5-7b-hf",
        "blip": "instructblip-vicuna-7b",
    }

    project_root = Path(repo_root)
    editor_name = hydra_config.editor_name # "ike_chain"
    subsample = hydra_config.subsample # 0
    subsample_edits = hydra_config.subsample_edits # 500
    model_name = hydra_config.model_name # "qwen3_4b"       # <-- change to match your model
    dataset_name = hydra_config.dataset_name # "aokvqa"        # <-- change to match your dataset
    task = hydra_config.task # "mc"             # <-- change to match your task (e.g., "mc", "qa", etc.)

    full_model_name = MODEL_NAME_MAP[model_name]
    src = project_root / "results" / "pred" / full_model_name / dataset_name / "mc_all.json"
    
    # change "results" to whatever we want to call our bias
    if biases_name == "":
        dst_dir = project_root / "bias_runs" / biases_name / "test" / f"editor_{editor_name}" / model_name / dataset_name
    else:
        dst_dir = project_root / "results" / "test" / f"editor_{editor_name}" / model_name / dataset_name

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "pred_mc.json"

    assert src.exists(), f"Source not found: {src}"
    shutil.copy2(src, dst)
    print(f"Copied {src}\n    -> {dst}")

    # Project root
    project_root = Path(repo_root)

    # Build test paths from args (no hardcoding)
    args = Namespace(
        config=namespace_config_path,
        editor=editor_name,
        model_name=model_name,
        dataset_name=dataset_name,
        task=task,
        batch_size=1,
        split="all",
        rationale=False,
        cot=False,
        subsample=subsample,
        subsample_edits=subsample_edits,
        overwrite=True
    )

    # Derive result prefix from args
    if biases_name == "":
        res_prefix = project_root / "bias_runs" / biases_name / "test" / f"editor_{args.editor}" / args.model_name / args.dataset_name
    else:
        res_prefix = project_root / "results" / "test" / f"editor_{args.editor}" / args.model_name / args.dataset_name

    res_prefix.mkdir(parents=True, exist_ok=True)
    (res_prefix / "pred_postedit").mkdir(parents=True, exist_ok=True)

    # Force all outputs into the test prefix (overwrite allowed)
    args.task_dir = str(res_prefix)
    args.edit_dir = str(res_prefix)
    args.pred_dir = str(res_prefix)
    args.pred_path = str(res_prefix / "pred_mc.json")

    args.pred_postedit_dir = str(res_prefix / f"pred_postedit")

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.suffix = "_cot" if args.rationale and args.cot else ("_rationale" if args.rationale else "")

    config = configure_args(args, config_path=args.config)
    config.subsample = args.subsample
    config.subsample_edits = args.subsample_edits
    config.rationale = args.rationale
    config.cot = args.cot
    config.pred_path = args.pred_path
    config.overwrite = args.overwrite
    config.task_dir = args.task_dir
    config.pred_dir = args.pred_dir
    config.pred_postedit_dir = args.pred_postedit_dir
    config.edit_dir = args.edit_dir

    config.plot_k_dist = True 
    run_edit(config, sequential=True, eval_every=20, subsample_path=subsample_path, biases_path=biases_path)

def locality_run(hydra_config: DictConfig):
    from argparse import Namespace
    from pathlib import Path
    import random
    import time
    import shutil
    import torch
    import json
    import argparse

    from revlm.config_utils import configure_args 
    from revlm.run.edit import run_edit_locality 

    # Repo setup
    repo_root = hydra_config.repo_root # "/home/sgw3fy/jobs/model_editing_jobs/repos/instruct_vlm_edit"
    os.chdir(repo_root)
    if repo_root not in sys.path:
        sys.path.append(repo_root)

    loc_sample_path = hydra_config.loc_sample_path
    loc_sample_name = os.path.splitext(os.path.basename(loc_sample_path))[0]
    namespace_config_path = hydra_config.namespace_config_path

    # Map short model names to full HF model names used in results/pred/
    MODEL_NAME_MAP = {
        "qwen3": "Qwen3-VL-8B-Instruct",
        "qwen3_4b": "Qwen3-VL-4B-Instruct",
        "llava": "llava-1.5-7b-hf",
        "blip": "instructblip-vicuna-7b",
    }

    project_root = Path(repo_root)
    editor_name = hydra_config.editor_name # "ike_chain"
    model_name = hydra_config.model_name # "qwen3_4b"       # <-- change to match your model
    dataset_name = hydra_config.dataset_name # "aokvqa"        # <-- change to match your dataset
    task = hydra_config.task # "mc"             # <-- change to match your task (e.g., "mc", "qa", etc.)
    q_index = hydra_config.q_index
    
    full_model_name = MODEL_NAME_MAP[model_name]
    src = project_root / "results" / "pred" / full_model_name / dataset_name / "mc_all.json"
    
    # change "results" to whatever we want to call our bias
    dst_dir = project_root / "loc_runs" / loc_sample_name / "test" / f"editor_{editor_name}" / model_name / dataset_name

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "pred_mc.json"

    assert src.exists(), f"Source not found: {src}"
    shutil.copy2(src, dst)
    print(f"Copied {src}\n    -> {dst}")

    # Project root
    project_root = Path(repo_root)

    # Build test paths from args (no hardcoding)
    args = Namespace(
        config=namespace_config_path,
        editor=editor_name,
        model_name=model_name,
        dataset_name=dataset_name,
        task=task,
        batch_size=1,
        split="all",
        rationale=False,
        cot=False,
        subsample=None,
        subsample_edits=None,
        overwrite=True,
        q_index=q_index
    )

    # Derive result prefix from args
    res_prefix = project_root / "loc_runs" / loc_sample_name / "test" / f"editor_{args.editor}" / args.model_name / args.dataset_name

    res_prefix.mkdir(parents=True, exist_ok=True)
    (res_prefix / "pred_postedit").mkdir(parents=True, exist_ok=True)

    # Force all outputs into the test prefix (overwrite allowed)
    args.task_dir = str(res_prefix)
    args.edit_dir = str(res_prefix)
    args.pred_dir = str(res_prefix)
    args.pred_path = str(res_prefix / "pred_mc.json")

    args.pred_postedit_dir = str(res_prefix / f"pred_postedit")

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.suffix = "_cot" if args.rationale and args.cot else ("_rationale" if args.rationale else "")

    config = configure_args(args, config_path=args.config)
    config.subsample = args.subsample
    config.subsample_edits = args.subsample_edits
    config.rationale = args.rationale
    config.cot = args.cot
    config.pred_path = args.pred_path
    config.overwrite = args.overwrite
    config.task_dir = args.task_dir
    config.pred_dir = args.pred_dir
    config.pred_postedit_dir = args.pred_postedit_dir
    config.edit_dir = args.edit_dir
    config.q_index = args.q_index
    print(f"current question index: {config.q_index}")

    config.plot_k_dist = True 
    run_edit_locality(config, sequential=False, eval_every=1, loc_sample_path=loc_sample_path)

def independent_run(hydra_config: DictConfig):
    from argparse import Namespace
    from pathlib import Path
    import random
    import time
    import shutil
    import torch
    import json
    import argparse

    from revlm.config_utils import configure_args 
    from revlm.run.edit import run_edit_indep

    # Repo setup
    repo_root = hydra_config.repo_root # "/home/sgw3fy/jobs/model_editing_jobs/repos/instruct_vlm_edit"
    output_path = hydra_config.output_path

    namespace_config_path = hydra_config.namespace_config_path

    # Map short model names to full HF model names used in results/pred/
    MODEL_NAME_MAP = {
        "qwen3": "Qwen3-VL-8B-Instruct",
        "qwen3_4b": "Qwen3-VL-4B-Instruct",
        "llava": "llava-1.5-7b-hf",
        "blip": "instructblip-vicuna-7b",
    }

    project_root = Path(repo_root)
    editor_name = hydra_config.editor_name # "ike_chain"
    model_name = hydra_config.model_name # "qwen3_4b"       # <-- change to match your model
    dataset_name = hydra_config.dataset_name # "aokvqa"        # leave it on aokvqa for now
    task = hydra_config.task # "mc"             # <-- change to match your task (e.g., "mc", "qa", etc.)
    indices = hydra_config.indices # qa_id indices

    wrong_edit_path = hydra_config.wrong_edit_path
    correct_edit_path = hydra_config.correct_edit_path
    eval_df_path = hydra_config.eval_df_path
    
    full_model_name = MODEL_NAME_MAP[model_name]
    
    # Build test paths from args (no hardcoding)
    args = Namespace(
        config=namespace_config_path,
        editor=editor_name,
        model_name=model_name,
        dataset_name=dataset_name,
        task=task,
        batch_size=1,
        split="all",
        rationale=False,
        cot=False,
        subsample=None,
        subsample_edits=None,
        overwrite=True,
        indices=indices,
        wrong_edit_path=wrong_edit_path,
        correct_edit_path=correct_edit_path,
        eval_df_path=eval_df_path,
        output_path=output_path
    )

    # Derive result prefix from args
    res_prefix = project_root / "indep_runs" / f"editor_{args.editor}" / args.model_name / args.dataset_name

    res_prefix.mkdir(parents=True, exist_ok=True)
    (res_prefix / "pred_postedit").mkdir(parents=True, exist_ok=True)

    # Force all outputs into the test prefix (overwrite allowed)
    args.task_dir = str(res_prefix)
    args.edit_dir = str(res_prefix)
    args.pred_dir = str(res_prefix)
    args.pred_path = str(res_prefix / "pred_mc.json")

    args.pred_postedit_dir = str(res_prefix / f"pred_postedit")

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.suffix = "_cot" if args.rationale and args.cot else ("_rationale" if args.rationale else "")

    config = configure_args(args, config_path=args.config)
    config.subsample = args.subsample
    config.subsample_edits = args.subsample_edits
    config.rationale = args.rationale
    config.cot = args.cot
    config.pred_path = args.pred_path
    config.overwrite = args.overwrite
    config.task_dir = args.task_dir
    config.pred_dir = args.pred_dir
    config.pred_postedit_dir = args.pred_postedit_dir
    config.edit_dir = args.edit_dir
    config.indices = args.indices
    
    config.eval_df_path = args.eval_df_path
    config.wrong_edit_path = args.wrong_edit_path
    config.correct_edit_path = args.correct_edit_path
    config.output_path = args.output_path
    print(f"current indices: {config.indices}")

    config.plot_k_dist = True 
    run_edit_indep(config)

def no_edit_runs(hydra_config: DictConfig):
    from argparse import Namespace
    from pathlib import Path
    import random
    import time
    import shutil
    import torch
    import json
    import argparse

    from revlm.config_utils import configure_args 
    from revlm.run.edit import run_edit_indep

    # Repo setup
    repo_root = hydra_config.repo_root # "/home/sgw3fy/jobs/model_editing_jobs/repos/instruct_vlm_edit"
    output_path = hydra_config.output_path

    namespace_config_path = hydra_config.namespace_config_path

    # Map short model names to full HF model names used in results/pred/
    MODEL_NAME_MAP = {
        "qwen3": "Qwen3-VL-8B-Instruct",
        "qwen3_4b": "Qwen3-VL-4B-Instruct",
        "llava": "llava-1.5-7b-hf",
        "blip": "instructblip-vicuna-7b",
    }

    project_root = Path(repo_root)
    editor_name = hydra_config.editor_name # "ike_chain"
    model_name = hydra_config.model_name # "qwen3_4b"       # <-- change to match your model
    dataset_name = hydra_config.dataset_name # "aokvqa"        # leave it on aokvqa for now
    task = hydra_config.task # "mc"             # <-- change to match your task (e.g., "mc", "qa", etc.)
    indices = hydra_config.indices # qa_id indices

    # output path
    output_path = hydra_config.output_path
    current_split = hydra_config.current_split # e.g. 0-499

    # 
    wrong_edit_path = hydra_config.wrong_edit_path
    correct_edit_path = hydra_config.correct_edit_path
    eval_df_path = hydra_config.eval_df_path
    
    full_model_name = MODEL_NAME_MAP[model_name]
    
    # Build test paths from args (no hardcoding)
    args = Namespace(
        config=namespace_config_path,
        editor=editor_name,
        model_name=model_name,
        dataset_name=dataset_name,
        task=task,
        batch_size=1,
        split="all",
        rationale=False,
        cot=False,
        subsample=None,
        subsample_edits=None,
        overwrite=True,
        indices=indices,
        wrong_edit_path=wrong_edit_path,
        correct_edit_path=correct_edit_path,
        eval_df_path=eval_df_path,
        output_path=output_path
    )

    # Derive result prefix from args
    res_prefix = project_root / "indep_runs" / f"editor_{args.editor}" / args.model_name / args.dataset_name

    res_prefix.mkdir(parents=True, exist_ok=True)
    (res_prefix / "pred_postedit").mkdir(parents=True, exist_ok=True)

    # Force all outputs into the test prefix (overwrite allowed)
    args.task_dir = str(res_prefix)
    args.edit_dir = str(res_prefix)
    args.pred_dir = str(res_prefix)
    args.pred_path = str(res_prefix / "pred_mc.json")

    args.pred_postedit_dir = str(res_prefix / f"pred_postedit")

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.suffix = "_cot" if args.rationale and args.cot else ("_rationale" if args.rationale else "")

    config = configure_args(args, config_path=args.config)
    config.subsample = args.subsample
    config.subsample_edits = args.subsample_edits
    config.rationale = args.rationale
    config.cot = args.cot
    config.pred_path = args.pred_path
    config.overwrite = args.overwrite
    config.task_dir = args.task_dir
    config.pred_dir = args.pred_dir
    config.pred_postedit_dir = args.pred_postedit_dir
    config.edit_dir = args.edit_dir
    config.indices = args.indices
    
    config.eval_df_path = args.eval_df_path
    config.wrong_edit_path = args.wrong_edit_path
    config.correct_edit_path = args.correct_edit_path
    config.output_path = args.output_path
    print(f"current indices: {config.indices}")

    config.plot_k_dist = True 
    run_edit_indep(config)

