cd $1
imagenet_path=$2

python generate_em.py \
    --arch resnet18 \
    --dataset imagenet-mini \
    --train-steps 3000 \
    --batch-size 128 \
    --optim sgd \
    --lr 0.1 \
    --lr-decay-rate 0.1 \
    --lr-decay-freq 1200 \
    --weight-decay 5e-4 \
    --momentum 0.9 \
    --pgd-radius 8 \
    --pgd-steps 7 \
    --pgd-step-size 2 \
    --pgd-random-start \
    --perturb-freq 1 \
    --report-freq 500 \
    --save-freq 500 \
    --parallel \
    --data-dir ${imagenet_path} \
    --save-dir ./exp_data/in-mini/noise/em8 \
    --save-name em
