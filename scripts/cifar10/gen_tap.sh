cd $1

python generate_tap.py \
    --arch resnet18 \
    --dataset cifar10 \
    --targeted \
    --batch-size 128 \
    --optim sgd \
    --lr 0.1 \
    --lr-decay-rate 0.1 \
    --lr-decay-freq 2000 \
    --pgd-radius 8 \
    --pgd-steps 250 \
    --pgd-step-size 0.064 \
    --pgd-random-start \
    --samp-num 1 \
    --resume \
    --resume-path ./exp_data/cifar10/train/clr/r0/train-fin-model.pkl \
    --data-dir ./data \
    --save-dir ./exp_data/cifar10/noise/tap8 \
    --save-name tap
