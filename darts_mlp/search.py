""" Search cell """
import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from config import SearchConfig
import utils
from models.search_mlp import SearchController
from architect import Architect
from visualize import plot
import math
import numpy as np
import scipy
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from tqdm.auto import tqdm
import functools


config = SearchConfig()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)


def main():
    logger.info("Logger is set - training start")

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    # get data with meta info
    #print(config.dataset_name)  

    dataloaders = utils.get_data(config.dataset_name, batch_size=config.batch_size, num_workers=config.num_workers)
    

    net_crit = nn.MSELoss().to(device)
    model = SearchController(in_size=32, out_size=32, n_count_features=dataloaders['train'][1], n_classes=1, 
                                n_layers=8, criterion=net_crit, n_nodes=2, device_ids=config.gpus)
    model = model.to(device)

    # weights optimizer
    w_optim = torch.optim.SGD(model.weights(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=config.w_weight_decay)
    # alphas optimizer
    alpha_optim = torch.optim.Adam(model.alphas(), config.alpha_lr, betas=(0.5, 0.999),
                                   weight_decay=config.alpha_weight_decay)

    # split data to train/validation


    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.epochs, eta_min=config.w_lr_min)
    architect = Architect(model, config.w_momentum, config.w_weight_decay)

    # training loop
    best_top1 = 0.
    for epoch in range(config.epochs):
        print(epoch)
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]


        # model.print_alphas(logger)

        # training
        train(dataloaders['train'][0], dataloaders['val'], model, architect, w_optim, alpha_optim, lr, epoch)

        # validation
        cur_step = (epoch+1) * len(dataloaders['train'][0])
        top1 = validate(dataloaders['test'], model, epoch, cur_step)

        # log
        # genotype
        genotype = model.genotype()
        logger.info("genotype = {}".format(genotype))

        # genotype as a image
        plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch+1))
        caption = "Epoch {}".format(epoch+1)
        plot(genotype.normal, plot_path + "-normal", caption)
        plot(genotype.reduce, plot_path + "-reduce", caption)

        # save
        if best_top1 < top1:
            best_top1 = top1
            best_genotype = genotype
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, config.path, is_best)
        print("")

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Best Genotype = {}".format(best_genotype))


def train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)

    model.train()

    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
        N = trn_X.size(0)

        # phase 2. architect step (alpha)
        alpha_optim.zero_grad()
        architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, w_optim)
        alpha_optim.step()

        # phase 1. child network step (w)
        w_optim.zero_grad() 
        logits = model(trn_X)
        loss = model.criterion(logits, trn_y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()

        # prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        # top1.update(prec1.item(), N)
        # top5.update(prec5.item(), N)

        # if step % config.print_freq == 0 or step == len(train_loader)-1:
        #     logger.info(
        #         "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
        #         "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
        #             epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
        #             top1=top1, top5=top5))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        # writer.add_scalar('train/top1', prec1.item(), cur_step)
        # writer.add_scalar('train/top5', prec5.item(), cur_step)
        cur_step += 1

    # logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))


def validate(valid_loader, model, epoch, cur_step):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits = model(X)
            loss = model.criterion(logits, y)
            # prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            # top1.update(prec1.item(), N)
            # top5.update(prec5.item(), N)

            # if step % config.print_freq == 0 or step == len(valid_loader)-1:
            #     logger.info(
            #         "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
            #         "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
            #             epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses,
            #             top1=top1, top5=top5))

    writer.add_scalar('val/loss', losses.avg, cur_step)
    # writer.add_scalar('val/top1', top1.avg, cur_step)
    # writer.add_scalar('val/top5', top5.avg, cur_step)

    # logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

    return losses.avg


if __name__ == "__main__":
    main()
