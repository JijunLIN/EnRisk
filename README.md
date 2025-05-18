# Ensemble Risk: based on PLUTO
This is a PLUTO autonomous driving using total and local ensemble network to give driving risk from neural network.

See official repository at

**PLUTO: Push the Limit of Imitation Learning-based Planning for Autonomous Driving**,

[Jie Cheng](https://jchengai.github.io/), [Yingbing Chen](https://sites.google.com/view/chenyingbing-homepage), and [Qifeng Chen](https://cqf.io/)


<p align="left">
<a href="https://jchengai.github.io/pluto">
<img src="https://img.shields.io/badge/Project-Page-blue?style=flat">
</a>
<a href='https://arxiv.org/abs/2404.14327' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/arXiv-PDF-red?style=flat&logo=arXiv&logoColor=wihte' alt='arXiv PDF'>
</a>
</p>

## Things Changed
- src/models/ensemble: add several models and trainers using ensemble.
- add ensemble config
- add a script training the model
- modify planner to adapt the ensemble
- modify renderer that shows the risk in video
- and other small changes

## Setup Environment

Test with Ubuntu 20.04.5 LTS.

### Setup dataset

Setup the nuPlan dataset following the [offiical-doc](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html)

### Setup conda environment

```
conda create -n pluto python=3.9
conda activate pluto

# install nuplan-devkit
git clone https://github.com/motional/nuplan-devkit.git && cd nuplan-devkit
pip install -e .
pip install -r ./requirements.txt

# setup pluto
cd ..
git clone https://github.com/jchengai/pluto.git && cd pluto
sh ./script/setup_env.sh
```

## Feature Cache

Set Environment variables. Set "/your/data/path".

```
export NUPLAN_DATA_ROOT="/your/data/path/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/your/data/path/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="/your/data/path/nuplan/exp"
```

Preprocess the dataset to accelerate training. It is recommended to run a small sanity check to make sure everything is correctly setup. You may need to change `cache.cache_path` to suit your condition.

```
 python run_training.py \
    py_func=cache +training=train_pluto \
    scenario_builder=nuplan_mini \
    cache.cache_path=/your/data/path/nuplan/exp/sanity_check \
    cache.cleanup_cache=true \
    scenario_filter=training_scenarios_tiny \
    worker=sequential
```

Preprocess the mini set. You should modify the "training_scenarios_mini.yaml" to scale the trainset.

```
 python run_training.py \
    py_func=cache +training=train_pluto \
    scenario_builder=nuplan_mini \
    cache.cache_path=/your/data/path/nuplan/exp/mini \
    cache.cleanup_cache=true \
    scenario_filter=training_scenarios_mini \
    worker.threads_per_node=5
```

Then preprocess the whole nuPlan training set (this will take some time).

```
 export PYTHONPATH=$PYTHONPATH:$(pwd)

 python run_training.py \
    py_func=cache +training=train_pluto \
    scenario_builder=nuplan \
    cache.cache_path=/your/data/path/nuplan/exp/cache_pluto_1M \
    cache.cleanup_cache=true \
    scenario_filter=training_scenarios_1M \
    worker.threads_per_node=40
```

## Training
### BEFORE YOUR TRAINING
You should
- adjust *every* yaml in config.
- modify train_pluto_model.sh and run_pluto_planner.sh (environment variable, batch_size, CHALLENGE and so on)
- adjust DEVICE, BATCH SIZE, WORKER NUMBER according to your device

run 
```
wandb login
```
to visualize the training.
### Normal training
(The training part it not fully tested)

Same, it is recommended to run a sanity check first:

```
CUDA_VISIBLE_DEVICES=0 python run_training.py \
  py_func=train +training=train_pluto \
  worker=single_machine_thread_pool worker.max_workers=4 \
  scenario_builder=nuplan cache.cache_path=/media/jjlin/database/nuplan/exp/sanity_check cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=4 data_loader.params.num_workers=1
```

Training on the mini dataset:

```
sh ./script/train_pluto_model.sh train train_pluto 32 /media/jjlin/database/nuplan/exp/tinymini
```

Training on the full dataset (without CIL):

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_training.py \
  py_func=train +training=train_pluto \
  worker=single_machine_thread_pool worker.max_workers=32 \
  scenario_builder=nuplan cache.cache_path=/nuplan/exp/cache_pluto_1M cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=32 data_loader.params.num_workers=16 \
  lr=1e-3 epochs=25 warmup_epochs=3 weight_decay=0.0001 \
  wandb.mode=online wandb.project=nuplan wandb.name=pluto
```

- add option `model.use_hidden_proj=true +custom_trainer.use_contrast_loss=true` to enable CIL.

- you can remove wandb related configurations if your prefer tensorboard.

### Ensemble training
Total ensemble PLUTO.
```
sh ./script/train_pluto_model.sh train train_total_ensemble_pluto 20 /media/jjlin/database/nuplan/exp/tinytinymini

sh ./script/run_pluto_planner.sh total_ensemble_planner nuplan_mini mini_demo_scenario te_t.ckpt /home/jjlin/pluto_dev/result

```
Prediction ensemble PLUTO.
```
sh ./script/train_pluto_model.sh train train_prediction_ensemble_pluto 20 /media/jjlin/database/nuplan/exp/tinymini

sh ./script/run_pluto_planner.sh prediction_ensemble_planner nuplan_mini mini_demo_scenario pe_t.ckpt /home/jjlin/pluto_dev/result

sh ./script/run_pluto_planner.sh prediction_ensemble_planner nuplan_mini mini_demo_scenario pe_tt.ckpt /home/jjlin/pluto_dev/result
```
MoE PLUTO
```
sh ./script/train_pluto_model.sh train train_MoE_ensemble_pluto 20 /media/jjlin/database/nuplan/exp/tinymini

sh ./script/run_pluto_planner.sh prediction_MoE_planner nuplan_mini mini_demo_scenario pe_t.ckpt /home/jjlin/pluto_dev/result
```

## Checkpoint

Download and place the checkpoint in the `pluto/checkpoints` folder.

| Model            | Download |
| ---------------- | -------- |
| Pluto-1M-aux-cil | [OneDrive](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jchengai_connect_ust_hk/EaFpLwwHFYVKsPVLH2nW5nEBNbPS7gqqu_Rv2V1dzODO-Q?e=LAZQcI)    |
| Total-Ensemble-0.1mini|[Tsinghua]|
| Prediction-Ensemble-0.1mini|[Tsinghua]|

## Run Pluto-planner simulation

Run simulation for a random scenario in the nuPlan-mini split

```
sh ./script/run_pluto_planner.sh pluto_planner nuplan_mini mini_demo_scenario exp2.ckpt /home/jjlin/pluto_dev/result

sh ./script/run_pluto_planner.sh pluto_planner nuplan_mini mini_demo_scenario pluto_1M_aux_cil.ckpt /home/jjlin/pluto_dev/result
```

The rendered simulation video will be saved to the specified directory (need change `/dir_to_save_the_simulation_result_video`).

## To Do

- [ ] improve docs
- [x] total ensemble
- [x] local ensemble
- [x] visualization
- [ ] evaluate
- [ ] utils

## Citation


## appendix: docker

```
# 列出本机的所有 docker容器：
docker ps -a
# 列出本机的运行中的 docker容器：
docker ps
# 列出本机的所有 image 文件：
docker images
```
删除并创建容器
```
docker stop enrisk && docker rm enrisk
docker run --name enrisk -idt continuumio/miniconda3
```
进入容器、查看镜像
```
docker exec -it enrisk /bin/bash
docker images
```
复制conda环境和代码
```
docker cp /home/jjlin/anaconda3/envs/pluto enrisk:/opt/conda/envs
docker cp /home/jjlin/pluto_dev enrisk:/root
docker cp /home/jjlin/nuplan-devkit enrisk:/root
```

删除、创建镜像并保存镜像
```
docker rmi image_enrisk
docker commit -a 'author' -m 'instruction' enrisk image_enrisk
docker save -o image_enrisk.tar image_enrisk
```

宿主机
```
# 读取文件
docker load -i image_enrisk.tar
# 创建容器 挂载nuplan路径 需要修改
docker run --name enrisk -v /media/jjlin/database/nuplan:/media/jjlin/database/nuplan -idt image_enrisk 
# 进入容器
docker exec -it enrisk /bin/bash
# 初始配置
conda activate pluto

sed -i '1s|^.*$|#!/opt/conda/envs/pluto/bin/python|' /opt/conda/envs/pluto/bin/pip
sed -i '1s|^.*$|#!/opt/conda/envs/pluto/bin/python|' /opt/conda/envs/pluto/bin/pip3

cd root/nuplan-devkit
pip install -e .

cd ../pluto_dev


# 检查CUDA版本，H100可能是12.1，更改依赖包版本
```