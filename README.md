

# Requirements
```bash
git clone [current].git
conda create -n revlm python=3.10 -y
conda activate revlm
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
(optional) jupyter notebook
```bash
conda install jupyterlab ipykernel notebook -y
jupyter kernelspec uninstall revlm -y
python -m ipykernel install --user --name revlm --display-name "revlm"
jupyter kernelspec list
```


# Datasets

Download ReasonVQA images using `main_cpu.py` and the configuration file from `config/downloader/gld.yaml`

`python main_cpu.py -m hydra/launcher=submitit_slurm +downloader=gld`

- Change the `save_path` from the `gld.yaml` accordingly (NOTE: use `/scratch/` as we'll need around 1 TB for the images)
- Change `config/config_cpu.yaml` accordingly, mainly
    - `log_dir`
    - `account`

After the download, run `python main_cpu.py -m hydra/launcher=submitit_slurm +downloader=gld_file_explorer` to create a dataframe with the paths to all images, but first update `config/downloader/gld_file_explorer.yaml` accordingly

- `base_path`
- `output_path` (probably scratch)

Then run `notebooks/reasonvqa.ipynb` from beginning to end to obtain the sample we need in the appropriate formats. Update the paths accordingly.

The main inputs can be obtained from [here](https://reasonvqa.duongtr.com/download), mainly from the Download Directly - Annotations section
The core outputs from the notebook needed for a run are

- `rvqa_correct_edit_df.pkl`
- `rvqa_wrong_edit_df.pkl`
- `eval_df.pkl`

# Run evaluation

Run `python main_gpu.py -m hydra/launcher=submitit_slurm +indep_runs=rvqa_sample_qwen3_4b`

Update the following entries accordingly in `config/indep_runs/rvqa_sample_qwen3_4b.yaml`

- `output_path`
- `namespace_config_path`

- Change `config/config_gpu.yaml` accordingly, mainly
    - `log_dir`
    - `account`