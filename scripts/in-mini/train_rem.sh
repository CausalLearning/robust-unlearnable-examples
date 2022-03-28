cd $1
imagenet_path=$2

python train.py \
    --arch resnet18 \
    --dataset imagenet-mini \
    --train-steps 40000 \
    --batch-size 128 \
    --optim sgd \
    --lr 0.1 \
    --lr-decay-rate 0.1 \
    --lr-decay-freq 16000 \
    --weight-decay 5e-4 \
    --momentum 0.9 \
    --pgd-radius 4 \
    --pgd-steps 8 \
    --pgd-step-size 1 \
    --pgd-random-start \
    --report-freq 1000 \
    --save-freq 100000 \
    --parallel \
    --data-dir ${imagenet_path} \
    --noise-path ./exp_data/in-mini/noise/rem8-4/rem-fin-def-noise.pkl \
    --save-dir ./exp_data/in-mini/train/rem8-4/r4 \
    --save-name train
