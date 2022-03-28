# Robust Unlearnable Examples: Protecting Data Against Adversarial Learning

This is the official repository for ICLR 2022 paper ["Robust Unlearnable Examples: Protecting Data Against Adversarial Learning"](https://openreview.net/forum?id=baUQQPwQiAg) by Shaopeng Fu, Fengxiang He, Yang Liu, Li Shen and Dacheng Tao.

## Requirements

- Python 3.8
- PyTorch 1.8.1
- Torchvision 0.9.1
- OpenCV 4.5.5

#### Install dependencies using pip

```shell
pip install -r requirements.txt
```

#### Install dependencies using Anaconda

It is recommended to create your experiment environment with [Anaconda3](https://www.anaconda.com/).

```shell
conda install pytorch=1.8.1 torchvision=0.9.1 cudatoolkit=10.2 -c pytorch
conda install -c conda-forge opencv=4.5.5
```

## Quick Start

We give an example of creating robust unlearnable examples from CIFAR-10 dataset. More experiment examples can be found in [./scripts](./scripts).

#### Generate robust error-minimizing noise for CIFAR-10 dataset

```bash
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
```

#### Perform adversarial training on robust unlearnable examples

```bash
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
    --pgd-radius 4 \
    --pgd-steps 10 \
    --pgd-step-size 0.8 \
    --pgd-random-start \
    --report-freq 1000 \
    --save-freq 100000 \
    --noise-path ./exp_data/cifar10/noise/rem8-4/rem-fin-def-noise.pkl \
    --data-dir ./data \
    --save-dir ./exp_data/cifar10/train/rem8-4/r4 \
    --save-name train
```

## Citation

```
@inproceedings{fu2022robust,
  title={Robust Unlearnable Examples: Protecting Data Against Adversarial Learning},
  author={Shaopeng Fu and Fengxiang He and Yang Liu and Li Shen and Dacheng Tao},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```

## Acknowledgment

- Unlearnable examples: [https://github.com/HanxunH/Unlearnable-Examples](https://github.com/HanxunH/Unlearnable-Examples)
- Adversarial poisons: [https://github.com/lhfowl/adversarial_poisons](https://github.com/lhfowl/adversarial_poisons)
- Neural tangent generalization attacks: [https://github.com/lionelmessi6410/ntga](https://github.com/lionelmessi6410/ntga)