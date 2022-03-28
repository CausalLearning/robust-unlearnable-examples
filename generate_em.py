import os
import pickle
import argparse
import numpy as np
import torch

import utils
import attacks


def get_args():
    parser = argparse.ArgumentParser()
    utils.add_shared_args(parser)

    parser.add_argument('--perturb-freq', type=int, default=1,
                        help='set the perturbation frequency')
    parser.add_argument('--report-freq', type=int, default=500,
                        help='set the report frequency')
    parser.add_argument('--save-freq', type=int, default=5000,
                        help='set the checkpoint saving frequency')

    return parser.parse_args()


def regenerate_def_noise(def_noise, model, criterion, loader, defender, cpu):
    for x, y, ii in loader:
        if not cpu: x, y = x.cuda(), y.cuda()
        def_x = defender.perturb(model, criterion, x, y)
        def_noise[ii] = (def_x - x).cpu().numpy()


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
    ''' init model / optim / dataloader / loss func '''
    model = utils.get_arch(args.arch, args.dataset)
    optim = utils.get_optim(
        args.optim, model.parameters(),
        lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    train_loader = utils.get_indexed_loader(
        args.dataset, batch_size=args.batch_size, root=args.data_dir, train=True)
    test_loader = utils.get_indexed_loader(
        args.dataset, batch_size=args.batch_size, root=args.data_dir, train=False)
    criterion = torch.nn.CrossEntropyLoss()

    defender = attacks.PGDAttacker(
        radius = args.pgd_radius,
        steps = args.pgd_steps,
        step_size = args.pgd_step_size,
        random_start = args.pgd_random_start,
        norm_type = args.pgd_norm_type,
        ascending = False,
    )

    if not args.cpu:
        model.cuda()
        criterion = criterion.cuda()

    if args.parallel:
        model = torch.nn.DataParallel(model)

    log = dict()

    ''' initialize the defensive noise (for unlearnable examples) '''
    data_nums = len( train_loader.loader.dataset )
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        def_noise = np.zeros([data_nums, 3, 32, 32], dtype=np.float16)
    elif args.dataset == 'tiny-imagenet':
        def_noise = np.zeros([data_nums, 3, 64, 64], dtype=np.float16)
    elif args.dataset == 'imagenet-mini':
        def_noise = np.zeros([data_nums, 3, 224, 224], dtype=np.float16)
    else:
        raise NotImplementedError

    for step in range(args.train_steps):
        lr = args.lr * (args.lr_decay_rate ** (step // args.lr_decay_freq))
        for group in optim.param_groups:
            group['lr'] = lr

        x, y, ii = next(train_loader)
        if not args.cpu:
            x, y = x.cuda(), y.cuda()

        if (step+1) % args.perturb_freq == 0:
            def_x = defender.perturb(model, criterion, x, y)
            def_noise[ii] = (def_x - x).cpu().numpy()

        if args.cpu:
            def_x = x + torch.tensor(def_noise[ii])
        else:
            def_x = x + torch.tensor(def_noise[ii]).cuda()
        def_x.clamp_(-0.5, 0.5)

        model.train()
        _y = model(def_x)
        def_acc = (_y.argmax(dim=1) == y).sum().item() / len(x)
        def_loss = criterion(_y, y)
        optim.zero_grad()
        def_loss.backward()
        optim.step()

        utils.add_log(log, 'def_acc', def_acc)
        utils.add_log(log, 'def_loss', def_loss.item())

        if (step+1) % args.save_freq == 0:
            save_checkpoint(
                args.save_dir, '{}-ckpt-{}'.format(args.save_name, step+1),
                model, optim, log, def_noise)

        if (step+1) % args.report_freq == 0:
            test_acc, test_loss = utils.evaluate(model, criterion, test_loader, args.cpu)
            utils.add_log(log, 'test_acc', test_acc)
            utils.add_log(log, 'test_loss', test_loss)

            logger.info('step [{}/{}]:'.format(step+1, args.train_steps))
            logger.info('def_acc {:.2%} \t def_loss {:.3e}'
                        .format( def_acc, def_loss.item() ))
            logger.info('test_acc  {:.2%} \t test_loss  {:.3e}'
                        .format( test_acc, test_loss ))
            logger.info('')

    regenerate_def_noise(
        def_noise, model, criterion, train_loader, defender, args.cpu)

    save_checkpoint(args.save_dir, '{}-fin'.format(args.save_name), model, optim, log, def_noise)

    return


if __name__ == '__main__':
    args = get_args()
    logger = utils.generic_init(args)

    try:
        main(args, logger)
    except Exception as e:
        logger.exception('Unexpected exception! %s', e)
