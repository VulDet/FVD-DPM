# FVD-DPM: Fine-grained Vulnerability Detection via Conditional Diffusion Probabilistic Models
This is an official implementation of our paper "[FVD-DPM: Fine-grained Vulnerability Detection via Conditional Diffusion Probabilistic Models](https://www.usenix.org/system/files/usenixsecurity24-shao.pdf)" accepted at **USENIX Security '24**.
# Overview 
In this repository, you will find a Python implementation of our FVD-DPM. As described in our paper, FVD-DPM formalizes vulnerability detection as a diffusion-based graph-structured prediction problem. Firstly, it generates a new fine-grained code representation by extracting graph-level program slices (i.e., GrVCs) from the Code Joint Graph. Then, a conditional diffusion probabilistic model is employed to model the node label distribution in the program slices, predicting which nodes are vulnerable. FVD-DPM achieves both precise vulnerability identification (i.e., slice-level detection) and vulnerability localization (i.e., statement-level detection). 
# Setting up the environment
You can set up the environment by following commands.
```
conda create -n FVD-DPM python=3.10
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install tqdm
pip install pyyaml
pip install easydict
pip install torch-sparse
pip install torch-scatter==2.1.0
pip install torch-geometric==2.1.0 
```
# Training and Evaluation
```
python preprocess.py
```
This command is used to transform Graph-based Vulnerability Candidate slices (i.e., GrVCs) into embedding vectors. We generate the initial node embedding based on two node's attributes: `type` and `code`. 
```
python -m torch.distributed.run --nproc_per_node gpu_number main.py --dataset dataset_name
```
This command is used to train FVD-DPM model. The `gpu_number` represents the number of GPUs when training FVD-DPM. The `dataset_name` is the dataset name we use to train and evaluate FVD-DPM, such as NVD, SARD, OpenSSL, Libav, Linux. 
```
python -m torch.distributed.run --nproc_per_node gpu_number main.py --dataset dataset_name --do_train test
```
Execute this command to test FVD-DPM on the test set.
# Usage 
This repository is partially based on [DPM-SNC](https://github.com/hsjang0/DPM-SNC).
