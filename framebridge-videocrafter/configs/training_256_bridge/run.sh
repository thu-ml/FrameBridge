# args
name="training_256_bridge"
config_file=configs/training_256_bridge/config.yaml
HOST_GPU_NUM=8 # change according to the number of GPUs

# save root dir for logs, checkpoints, tensorboard record, etc.
save_root=experiments

mkdir -p $save_root/$name

## run
python3 -m torch.distributed.launch \
--nproc_per_node=$HOST_GPU_NUM --nnodes=1 --master_addr=127.0.0.1 --master_port=12352 --node_rank=0 \
main/trainer.py \
--base $config_file \
--train \
--name $name \
--logdir $save_root \
--devices $HOST_GPU_NUM \
lightning.trainer.num_nodes=1