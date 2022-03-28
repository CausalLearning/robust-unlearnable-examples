cd $1
imagenet_path=$2

python generate_tap.py \
    --arch resnet18 \
    --dataset imagenet-mini \
    --targeted \
    --batch-size 128 \
    --optim sgd \
    --lr 0.1 \
    --lr-decay-rate 0.1 \
    --lr-decay-freq 2000 \
    --pgd-radius 8 \
    --pgd-steps 100 \
    --pgd-step-size 0.16 \
    --pgd-random-start \
    --samp-num 1 \
    --parallel \
    --resume \
    --resume-path ./exp_data/in-mini/train/clr/r0/train-fin-model.pkl \
    --data-dir ${imagenet_path} \
    --save-dir ./exp_data/in-mini/noise/tap8 \
    --save-name tap
