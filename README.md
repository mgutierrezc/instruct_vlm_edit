

# Requirements
```bash
git clone [current].git
conda create -y -n revlm python=3.10.0
conda activate revlm
...
pip install -r requirements.txt
```
(optional) jupyter notebook
```bash
conda install jupyterlab ipykernel notebook -y
jupyter kernelspec remove revlm -y
python -m ipykernel install --user --name revlm --display-name "revlm"
```


# Datasets

To download images and reproduce the preprocessing of raw aokvqa and fvqa datasets, follow the steps [here](data_raw/README.md). 

Benchmark dataset RationaleVQA (based on AOKVQA, FVQA datasets) can be downloaded from [HuggingFace](https://huggingface.co/datasets/JJoy333/RationaleVQA) 


# Run

Update each .sbatch with your account/partition/GPU, CUDA/module load and project path.

1. prompting
```bash
cd ./jobs/eval
bash run.sh
```
2. finetune
```bash
cd ./jobs/ft
bash run.sh

cd ./jobs/ft_ewc
bash run.sh

cd ./jobs/ft_retrain
bash run.sh
```