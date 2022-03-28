import pickle
import os
import sys
import logging
import numpy as np
import torch
import cv2
import torchvision
import torchvision.transforms as transforms

import models
from . import data
from . import imagenet_utils


class AverageMeter():
    def __init__(self):
        self.cnt = 0
        self.sum = 0
        self.mean = 0

    def update(self, val, cnt):
        self.cnt += cnt
        self.sum += val * cnt
        self.mean = self.sum / self.cnt

    def average(self):
        return self.mean
    
    def total(self):
        return self.sum


def add_log(log, key, value):
    if key not in log.keys():
        log[key] = []
    log[key].append(value)


def get_transforms(dataset, train=True, is_tensor=True):
    if dataset == 'imagenet' or dataset == 'imagenet-mini':
        return imagenet_utils.get_transforms(dataset, train, is_tensor)

    if train:
        if dataset == 'cifar10' or dataset == 'cifar100':
            comp1 = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4), ]
        elif dataset == 'tiny-imagenet':
            comp1 = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, 8), ]
        else:
            raise NotImplementedError
    else:
        comp1 = []

    if is_tensor:
        comp2 = [
            torchvision.transforms.Normalize((255*0.5, 255*0.5, 255*0.5), (255., 255., 255.))]
    else:
        comp2 = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.))]

    trans = transforms.Compose( [*comp1, *comp2] )

    if is_tensor: trans = data.ElementWiseTransform(trans)

    return trans


def get_filter(fitr):
    if fitr == 'averaging':
        return lambda x: cv2.blur(x, (3,3))
    elif fitr == 'gaussian':
        return lambda x: cv2.GaussianBlur(x, (3,3), 0)
    elif fitr == 'median':
        return lambda x: cv2.medianBlur(x, 3)
    elif fitr == 'bilateral':
        return lambda x: cv2.bilateralFilter(x, 9, 75, 75)

    raise ValueError


def get_dataset(dataset, root='./data', train=True, fitr=None):
    if dataset == 'imagenet' or dataset == 'imagenet-mini':
        return imagenet_utils.get_dataset(dataset, root, train)

    transform = get_transforms(dataset, train=train, is_tensor=False)
    lp_fitr   = None if fitr is None else get_filter(fitr)

    if dataset == 'cifar10':
        target_set = data.datasetCIFAR10(root=root, train=train, transform=transform)
        x, y = target_set.data, target_set.targets
    elif dataset == 'cifar100':
        target_set = data.datasetCIFAR100(root=root, train=train, transform=transform)
        x, y = target_set.data, target_set.targets
    elif dataset == 'tiny-imagenet':
        target_set = data.datasetTinyImageNet(root=root, train=train, transform=transform)
        x, y = target_set.x, target_set.y
    else:
        raise NotImplementedError('dataset {} is not supported'.format(dataset))

    return data.Dataset(x, y, transform, lp_fitr)


def get_indexed_loader(dataset, batch_size, root='./data', train=True):
    if dataset == 'imagenet' or dataset == 'imagenet-mini':
        return imagenet_utils.get_indexed_loader(dataset, batch_size, root, train)

    target_set = get_dataset(dataset, root=root, train=train)

    if train:
        target_set = data.IndexedDataset(x=target_set.x, y=target_set.y, transform=target_set.transform)
    else:
        target_set = data.Dataset(x=target_set.x, y=target_set.y, transform=target_set.transform)

    if train:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=False, drop_last=False)

    return loader


def get_indexed_tensor_loader(dataset, batch_size, root='./data', train=True):
    if dataset == 'imagenet' or dataset == 'imagenet-mini':
        return imagenet_utils.get_indexed_tensor_loader(dataset, batch_size, root, train)

    target_set = get_dataset(dataset, root=root, train=train)
    target_set = data.IndexedTensorDataset(x=target_set.x, y=target_set.y)

    if train:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=False, drop_last=False)

    return loader


def get_poisoned_loader(
        dataset, batch_size, root='./data', train=True,
        noise_path=None, noise_rate=1.0, poisoned_indices_path=None, fitr=None):

    if dataset == 'imagenet' or dataset == 'imagenet-mini':
        return imagenet_utils.get_poisoned_loader(
                dataset, batch_size, root, train, noise_path, noise_rate, poisoned_indices_path, fitr)

    target_set = get_dataset(dataset, root=root, train=train, fitr=fitr)

    if noise_path is not None:
        with open(noise_path, 'rb') as f:
            raw_noise = pickle.load(f)

        assert isinstance(raw_noise, np.ndarray)
        assert raw_noise.dtype == np.int8

        raw_noise = raw_noise.astype(np.int16)

        noise = np.zeros_like(raw_noise)

        if poisoned_indices_path is not None:
            with open(poisoned_indices_path, 'rb') as f:
                indices = pickle.load(f)
        else:
            indices = np.random.permutation(len(noise))[:int(len(noise)*noise_rate)]

        noise[indices] += raw_noise[indices]

        ''' restore noise (NCWH) for raw images (NHWC) '''
        noise = np.transpose(noise, [0,2,3,1])

        ''' add noise to images (uint8, 0~255) '''
        imgs = target_set.x.astype(np.int16) + noise
        imgs = imgs.clip(0,255).astype(np.uint8)
        target_set.x = imgs

    target_set = data.Dataset(x=target_set.x, y=target_set.y, transform=target_set.transform, fitr=target_set.fitr)

    if train:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=False, drop_last=False)

    return loader


def get_clear_loader(
        dataset, batch_size, root='./data', train=True,
        noise_rate=1.0, poisoned_indices_path=None, fitr=None):

    if dataset == 'imagenet' or dataset == 'imagenet-mini':
        return imagenet_utils.get_clear_loader(
                dataset, batch_size, root, train, noise_rate, poisoned_indices_path)

    target_set = get_dataset(dataset, root=root, train=train, fitr=fitr)
    data_nums = len(target_set)

    if poisoned_indices_path is not None:
        with open(poisoned_indices_path, 'rb') as f:
            poi_indices = pickle.load(f)
        indices = np.array( list( set(range(data_nums)) - set(poi_indices) ) )

    else:
        indices = np.random.permutation(range(data_nums))[: int( data_nums * (1-noise_rate) )]

    ''' select clear examples '''
    target_set.x = target_set.x[indices]
    target_set.y = np.array(target_set.y)[indices]

    target_set = data.Dataset(x=target_set.x, y=target_set.y, transform=target_set.transform, fitr=target_set.fitr)

    if train:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        loader = data.Loader(target_set, batch_size=batch_size, shuffle=False, drop_last=False)

    return loader


def get_arch(arch, dataset):
    if dataset == 'cifar10':
        in_dims, out_dims = 3, 10
    elif dataset == 'cifar100':
        in_dims, out_dims = 3, 100
    elif dataset == 'tiny-imagenet':
        in_dims, out_dims = 3, 200
    elif dataset == 'imagenet':
        in_dims, out_dims = 3, 1000
    elif dataset == 'imagenet-mini':
        in_dims, out_dims = 3, 100
    else:
        raise NotImplementedError('dataset {} is not supported'.format(dataset))

    if arch == 'resnet18':
        return models.resnet18(in_dims, out_dims)

    elif arch == 'resnet50':
        return models.resnet50(in_dims, out_dims)

    elif arch == 'wrn-34-10':
        return models.wrn34_10(in_dims, out_dims)

    elif arch == 'vgg11-bn':
        if dataset == 'imagenet' or dataset == 'imagenet-mini':
            raise NotImplementedError
        return models.vgg11_bn(in_dims, out_dims)

    elif arch == 'vgg16-bn':
        if dataset == 'imagenet' or dataset == 'imagenet-mini':
            raise NotImplementedError
        return models.vgg16_bn(in_dims, out_dims)

    elif arch == 'vgg19-bn':
        return models.vgg19_bn(in_dims, out_dims)

    elif arch == 'densenet-121':
        return models.densenet121(num_classes=out_dims)

    else:
        raise NotImplementedError('architecture {} is not supported'.format(arch))


def get_optim(optim, params, lr=0.1, weight_decay=1e-4, momentum=0.9):
    if optim == 'sgd':
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optim == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    raise NotImplementedError('optimizer {} is not supported'.format(optim))


def generic_init(args):
    if os.path.exists(args.save_dir) == False:
        os.makedirs(args.save_dir)

    fmt = '%(asctime)s %(name)s:%(levelname)s:  %(message)s'
    formatter = logging.Formatter(
        fmt, datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(
        '{}/{}_log.txt'.format(args.save_dir, args.save_name), mode='w')
    fh.setFormatter(formatter)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=fmt, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.addHandler(fh)

    logger.info('Arguments')
    for arg in vars(args):
        logger.info('    {:<22}        {}'.format(arg+':', getattr(args,arg)) )
    logger.info('')

    return logger


def evaluate(model, criterion, loader, cpu):
    acc = AverageMeter()
    loss = AverageMeter()

    model.eval()
    for x, y in loader:
        if not cpu: x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            _y = model(x)
            ac = (_y.argmax(dim=1) == y).sum().item() / len(x)
            lo = criterion(_y,y).item()
        acc.update(ac, len(x))
        loss.update(lo, len(x))

    return acc.average(), loss.average()


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def get_model_state(model):
    # if isinstance(model, torch.nn.parallel.DistributedDataParallel):
    if isinstance(model, torch.nn.DataParallel):
        model_state = model_state_to_cpu(model.module.state_dict())
    else:
        model_state = model.state_dict()
    return model_state
