cd $1

for i in {2..8..2}
do
python train.py \
    --data-mode mix \
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
    --pgd-radius 4 \
    --pgd-steps 10 \
    --pgd-step-size 0.8 \
    --pgd-random-start \
    --report-freq 1000 \
    --save-freq 100000 \
    --poi-idx-path ./data/indices/cifar10/idx-${i}0.pkl \
    --noise-path ./exp_data/cifar10/noise/rem8-4/rem-fin-def-noise.pkl \
    --data-dir ./data \
    --save-dir ./exp_data/cifar10/mixing/rem8-4/${i}0/r4 \
    --save-name train
done
