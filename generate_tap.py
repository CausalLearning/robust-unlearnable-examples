import os
import pickle
import argparse
import numpy as np
import torch

import utils
import attacks


class TargetedIndexedDataset():
    def __init__(self, dataset, classes):
        self.dataset = dataset
        self.classes = classes

    def __getitem__(self, idx):
        x, y, ii = self.dataset[idx]
        y += 1
        if y >= self.classes: y -= self.classes

        return x, y, ii

    def __len__(self):
        return len(self.dataset)


def get_args():
    parser = argparse.ArgumentParser()
    utils.add_shared_args(parser)

    parser.add_argument('--adv-type', type=str, default='robust-pgd',
                        choices=['robust-pgd', 'diff-aug-pgd'])
    parser.add_argument('--targeted', action='store_true')

    parser.add_argument('--samp-num', type=int, default=1,
                        help='set the number of samples for calculating expectations')

    parser.add_argument('--resume', action='store_true',
                        help='set resume')
    parser.add_argument('--resume-path', type=str, default=None,
                        help='set where to resume the model')

    return parser.parse_args()


def regenerate_def_noise(def_noise, model, criterion, loader, defender, cpu, logger):
    cnt = 0
    for x, y, ii in loader:
        cnt += 1
        logger.info('progress [{}/{}]'.format(cnt, len(loader)) )

        if not cpu: x, y = x.cuda(), y.cuda()
        delta = defender.perturb(model, criterion, x, y)
        def_noise[ii] = delta.cpu().numpy()


def save_checkpoint(save_dir, save_name, model, optim, log, def_noise=None):
    torch.save({
        'model_state_dict': utils.get_model_state(model),
        'optim_state_dict': optim.state_dict(),
        }, os.path.join(save_dir, '{}-model.pkl'.format(save_name)))
    with open(os.path.join(save_dir, '{}-log.pkl'.format(save_name)), 'wb') as f:
        pickle.dump(log, f)
    if def_noise is not None:
        def_noise = (def_noise * 255).round()
        assert (def_noise.max()<=127 and def_noise.min()>=-128)
        def_noise = def_noise.astype(np.int8)
        with open(os.path.join(save_dir, '{}-def-noise.pkl'.format(save_name)), 'wb') as f:
            pickle.dump(def_noise, f)


def main(args, logger):
    ''' init model / optim / loss func '''
    model = utils.get_arch(args.arch, args.dataset)
    optim = utils.get_optim(
        args.optim, model.parameters(),
        lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    criterion = torch.nn.CrossEntropyLoss()

    ''' get Tensor train loader '''
    train_loader = utils.get_indexed_tensor_loader(
        args.dataset, batch_size=args.batch_size, root=args.data_dir, train=True)

    dataset = train_loader.loader.dataset
    ascending = True
    if args.targeted:
        if args.dataset == 'cifar10': classes = 10
        elif args.dataset == 'cifar100': classes = 100
        elif args.dataset == 'imagenet-mini': classes = 100
        else: raise ValueError
        dataset = TargetedIndexedDataset(dataset, classes)
        ascending = False

    train_loader = utils.Loader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    ''' get train transforms '''
    train_trans = utils.get_transforms(
        args.dataset, train=True, is_tensor=True)

    ''' get (original) test loader '''
    test_loader = utils.get_indexed_loader(
        args.dataset, batch_size=args.batch_size, root=args.data_dir, train=False)

    if args.adv_type == 'robust-pgd':
        defender = attacks.RobustPGDAttacker(
            samp_num     = args.samp_num,
            trans        = train_trans,
            radius       = args.pgd_radius,
            steps        = args.pgd_steps,
            step_size    = args.pgd_step_size,
            random_start = args.pgd_random_start,
            ascending    = ascending,
        )
    elif args.adv_type == 'diff-aug-pgd':
        defender = attacks.DiffAugPGDAttacker(
            samp_num     = args.samp_num,
            trans        = train_trans,
            radius       = args.pgd_radius,
            steps        = args.pgd_steps,
            step_size    = args.pgd_step_size,
            random_start = args.pgd_random_start,
            ascending    = ascending,
        )
    else: raise ValueError

    ''' initialize the defensive noise (for unlearnable examples) '''
    data_nums = len( train_loader.loader.dataset )
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        def_noise = np.zeros([data_nums, 3, 32, 32], dtype=np.float16)
    elif args.dataset == 'tiny-imagenet':
        def_noise = np.zeros([data_nums, 3, 64, 64], dtype=np.float16)
    elif args.dataset == 'imagenet-mini':
        def_noise = np.zeros([data_nums, 3, 256, 256], dtype=np.float16)
    else:
        raise NotImplementedError

    start_step = 0
    log = dict()

    if not args.cpu:
        model.cuda()
        criterion = criterion.cuda()

    if args.resume:
        state_dict = torch.load( os.path.join(args.resume_path) )
        model.load_state_dict( state_dict['model_state_dict'] )
        optim.load_state_dict( state_dict['optim_state_dict'] )
        del state_dict

    if args.parallel:
        model = torch.nn.DataParallel(model)

    logger.info('Noise generation started')

    regenerate_def_noise(
        def_noise, model, criterion, train_loader, defender, args.cpu, logger)

    logger.info('Noise generation finished')

    save_checkpoint(args.save_dir, '{}-fin'.format(args.save_name), model, optim, log, def_noise)

    return


if __name__ == '__main__':
    args = get_args()
    logger = utils.generic_init(args)

    logger.info('EXP: robust minimax pgd perturbation')
    try:
        main(args, logger)
    except Exception as e:
        logger.exception('Unexpected exception! %s', e)
