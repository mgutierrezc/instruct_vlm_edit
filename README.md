

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

Download images following the steps [here](data_raw/README.md). 

Benchmark dataset RationaleVQA (based on AOKVQA, FVQA datasets) can be downloaded/called from [here](https://huggingface.co/datasets/JJoy333/RationaleVQA). 


# Run

- Update each .sbatch with your account/partition/GPU, CUDA/module load and project path. 

- Note that **ike_chain** is the other name of **reasonedit** in the original implementation. 

- Batch job examples
1. use ReasonEdit to edit 4 VLMs on AOKVQA
```bash
cd ./jobs/edit/aokvqa/ike_chain
bash run.sh
```
2. use GRACE to edit 4 VLMs on AOKVQA
```bash
cd ./jobs/edit/aokvqa/grace
bash run.sh
```



