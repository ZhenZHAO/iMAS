tport=55476
ngpu=4

python -m torch.distributed.launch \
    --nproc_per_node=${ngpu} \
    --node_rank=0 \
    --master_port=${tport} \
    train_semi.py \
    --config=./exps/citys_semi372/config_semi.yaml --seed 2 --port ${tport}
# -------
    # --config=./exps/citys_semi186/config_semi.yaml --seed 2 --port ${tport}
    # --config=./exps/citys_semi372/config_semi.yaml --seed 2 --port ${tport}
    # --config=./exps/citys_semi744/config_semi.yaml --seed 2 --port ${tport}
    # --config=./exps/citys_semi1488/config_semi.yaml --seed 2 --port ${tport}
