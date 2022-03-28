cd $1

python train.py \
    --arch resnet18 \
    --dataset cifar10 \
    --train-steps 40000 \
    --batch-size 128 \
    --optim sgd \
    --lr 0.1 \
    --lr-decay-rate 0.1 \
    --lr-decay-freq 16000 \
    --weight-decay 5e-4 \
    --momentum 0.9 \
    --pgd-radius 0 \
    --pgd-steps 10 \
    --pgd-step-size 0 \
    --pgd-random-start \
    --report-freq 1000 \
    --save-freq 100000 \
    --data-dir ./data \
    --save-dir ./exp_data/cifar10/train/clr/r0 \
    --save-name train
