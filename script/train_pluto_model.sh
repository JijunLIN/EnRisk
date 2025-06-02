cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"

export NUPLAN_DATA_ROOT="/media/jjlin/database1/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/media/jjlin/database1/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="/media/jjlin/database1/nuplan/exp"

FUNC=$1
TRAIN_JOB=$2
WORKERS_NUM=$3
CACHE_PATH=$4

CUDA_VISIBLE_DEVICES=0 python run_training.py \
  py_func=$FUNC +training=$TRAIN_JOB \
  worker=single_machine_thread_pool worker.max_workers=$WORKERS_NUM \
  scenario_builder=nuplan cache.cache_path=$CACHE_PATH cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=32 data_loader.params.num_workers=20