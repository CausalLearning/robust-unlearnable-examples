cd $1

python generate_robust_em.py \
    --arch resnet18 \
    --dataset cifar10 \
    --train-steps 5000 \
    --batch-size 128 \
    --optim sgd \
    --lr 0.1 \
    --lr-decay-rate 0.1 \
    --lr-decay-freq 2000 \
    --weight-decay 5e-4 \
    --momentum 0.9 \
    --pgd-radius 8 \
    --pgd-steps 10 \
    --pgd-step-size 1.6 \
    --pgd-random-start \
    --atk-pgd-radius 4 \
    --atk-pgd-steps 10 \
    --atk-pgd-step-size 0.8 \
    --atk-pgd-random-start \
    --samp-num 5 \
    --report-freq 1000 \
    --save-freq 1000 \
    --data-dir ./data \
    --save-dir ./exp_data/cifar10/noise/rem8-4 \
    --save-name rem
