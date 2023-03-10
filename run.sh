python main.py \
--name save_seed1_cos_T100 \
--arch resnet18 \
--datasets CIFAR100 \
--train_batch_size 128 \
--epochs 200 \
--lr 0.1 \
--momentum 0.9 \
--weight_decay 5e-4 \
--lr_schedule cos \
--schedule 80 120 160 \
--gamma 0.2 \
--gpu 1 \
--random_seed 1 \
--T_max 100 \
--save_path