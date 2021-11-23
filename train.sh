#!/bin/bash
op=MatMul
for((layer=2;layer<8;layer++))
do
for((n=3;n<8;n++))
do
lr_base=1
python3 train.py --raw_data=data/conv2d_rand_data_analyze.csv --data_col_start=0 --data_col_end=2 --hidden_neurals=${n} --hidden_layers=${layer} --label_col=3 --batch_size=64 --train_epochs=5 --default_pc_exp=1 --loss_log=./loss_log.txt --load=0 --lr=0.0001 --checkpoint_dir=./train_checkpoints/
for((i=0;i<5;i++))
do
lr_base=0${lr_base}
lr=0.${lr_base}
echo new round lr = $lr
python3 train.py --raw_data=data/conv2d_rand_data_analyze.csv --data_col_start=0 --data_col_end=2 --hidden_neurals=${n} --hidden_layers=${layer} --label_col=3 --batch_size=64 --train_epochs=5 --default_pc_exp=1 --loss_log=./loss_log.txt --load=0 --lr=0.0001 --checkpoint_dir=./train_checkpoints/
done
mv result.txt results/${op}_l${layer}_n${n}_result.txt
done
done