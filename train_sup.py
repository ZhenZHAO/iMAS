import argparse
import yaml
import os,sys
import os.path as osp
import pprint
import time
from datetime import datetime

import random
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from imas.dataset.builder import get_loader
from imas.models.model_helper import ModelBuilder
from imas.utils.dist_helper import setup_distributed
from imas.utils.loss_helper import get_criterion
from imas.utils.lr_helper import get_optimizer, get_scheduler
from imas.utils.utils import init_log, get_rank, get_world_size, set_random_seed, setup_default_logging
from imas.utils.utils import AverageMeter, intersectionAndUnion, load_state


def main(in_args):
    args = in_args
    if args.seed is not None:
        # print("set random seed to", args.seed)
        set_random_seed(args.seed, deterministic=True)
        # set_random_seed(args.seed)
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    # will affect reproduction
    # cudnn.enabled = True
    # cudnn.benchmark = True

    rank, world_size = setup_distributed(port=args.port)

    ###########################
    # 1. output settings
    ###########################
    cfg["exp_path"] = osp.dirname(args.config)
    cfg["save_path"] = osp.join(cfg["exp_path"], cfg["saver"]["snapshot_dir"])
    cfg["log_path"] = osp.join(cfg["exp_path"], "log")
    flag_use_tb = cfg["saver"]["use_tb"]

    if not os.path.exists(cfg["log_path"]) and rank == 0:
        os.makedirs(cfg["log_path"])

    if not osp.exists(cfg["save_path"]) and rank == 0:
        os.makedirs(cfg["save_path"])
    
    if rank == 0:
        logger, curr_timestr = setup_default_logging("global", cfg["log_path"])

        # logger.propagate = 0
        # logger.parent = None
        csv_path = os.path.join(cfg["log_path"], "seg_{}_stat.csv".format(curr_timestr))
    else:
        logger, curr_timestr, csv_path = None, "", None
    
    # make sure all folder are correctly created at rank == 0
    dist.barrier()

    if rank == 0:
        logger.info("{}".format(pprint.pformat(cfg)))
        if flag_use_tb:
            tb_logger = SummaryWriter(
                osp.join(cfg["log_path"], "events_seg",curr_timestr)
            )
        else:
            tb_logger = None
    else:
        tb_logger = None
    
    ###########################
    # 2. prepare model
    ###########################
    model = ModelBuilder(cfg["net"])
    modules_back = [model.encoder]
    if cfg["net"].get("aux_loss", False):
        modules_head = [model.auxor, model.decoder]
    else:
        modules_head = [model.decoder]

    if cfg["net"].get("sync_bn", True):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    ###########################
    # 3. data
    ###########################
    criterion = get_criterion(cfg)
    # dataloader
    train_loader_sup, val_loader = get_loader(cfg, seed=args.seed)

    ##############################
    # 4. optimizer & scheduler
    ##############################
    cfg_trainer = cfg["trainer"]
    cfg_optim = cfg_trainer["optimizer"]
    times = 10 if "pascal" in cfg["dataset"]["type"] else 1

    params_list = []
    for module in modules_back:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"])
        )
    for module in modules_head:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"] * times)
        )

    optimizer = get_optimizer(params_list, cfg_optim)

    ######################################
    # 5. resume
    ######################################
    best_prec = 0
    best_epoch = -1
    last_epoch = 0
    # auto_resume > pretrain
    if cfg["saver"].get("auto_resume", False):
        lastest_model = osp.join(cfg["save_path"], "ckpt.pth")
        if not osp.exists(lastest_model):
            "No checkpoint found in '{}'".format(lastest_model)
        else:
            # print(f"Resume model from: '{lastest_model}'")
            best_prec, last_epoch = load_state(
                lastest_model, model, optimizer=optimizer, key="model_state"
            )
    # elif cfg["saver"].get("pretrain", False):
    #     load_state(cfg["saver"]["pretrain"], model, key="model_state")

    optimizer_old = get_optimizer(params_list, cfg_optim)
    lr_scheduler = get_scheduler(
        cfg_trainer, len(train_loader_sup), optimizer_old, start_epoch=last_epoch
    )

    ######################################
    # 6. training loop
    ######################################
    if rank == 0:
        logger.info('-------------------------- start training --------------------------')
    for epoch in range(last_epoch, cfg_trainer["epochs"]):
        # Training
        res_loss = train(
                        model, optimizer, lr_scheduler, criterion,
                        train_loader_sup, epoch,
                        logger, tb_logger,cfg,
                    )

        # Validation and store checkpoint
        # prec = validate(model, val_loader, epoch, logger, cfg)
        prec = validate_citys(model, val_loader, epoch, logger, cfg)

        if rank == 0:
            state = {"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),"best_miou": best_prec
                }

            if prec > best_prec:
                best_prec = prec
                best_epoch = epoch
                state["best_miou"] = prec
                
                # torch.save(state, osp.join(cfg["saver"]["snapshot_dir"], "ckpt_best.pth"))
                torch.save(state, osp.join(cfg["save_path"], "ckpt_best.pth"))

            # torch.save(state, osp.join(cfg["saver"]["snapshot_dir"], "ckpt.pth"))
            torch.save(state, osp.join(cfg["save_path"], "ckpt.pth"))
            
            # save statistics
            tmp_results = {
                        'loss': res_loss,
                        'miou': prec,
                        "best": best_prec}
            data_frame = pd.DataFrame(data=tmp_results, index=range(epoch, epoch+1))
            if epoch > 0 and osp.exists(csv_path):
                data_frame.to_csv(csv_path, mode='a', header=None, index_label='epoch')
            else:
                data_frame.to_csv(csv_path, index_label='epoch')

            logger.info(" <<Test>> - Epoch: {}.  MIoU: {:.2f}. \033[31mBest: {:.2f}/{}\033[0m".format(epoch, prec * 100, best_prec * 100, best_epoch))
            if tb_logger is not None:
                tb_logger.add_scalar("mIoU val", prec, epoch)


def train(
    model,
    optimizer,
    lr_scheduler,
    criterion,
    data_loader,
    epoch,
    logger,
    tb_logger,
    cfg
):
    model.train()

    data_loader.sampler.set_epoch(epoch)
    data_loader_iter = iter(data_loader)

    rank, world_size = dist.get_rank(), dist.get_world_size()

    losses = AverageMeter(10)
    data_times = AverageMeter(10)
    batch_times = AverageMeter(10)
    learning_rates = AverageMeter(10)

    batch_end = time.time()
    # print freq 4 times for a epoch
    print_freq = len(data_loader) // 4
    print_freq_lst = [i * print_freq for i in range(1,4)]
    print_freq_lst.append(len(data_loader) -1)
    for step in range(len(data_loader)):
        batch_start = time.time()
        data_times.update(batch_start - batch_end)

        i_iter = epoch * len(data_loader) + step # total iters till now
        lr = lr_scheduler.get_lr()
        learning_rates.update(lr[0])
        lr_scheduler.step() # lr is updated at the iteration level

        index_l, image, label = data_loader_iter.next()
        # print("="*20, image.shape, label.shape)
        batch_size, h, w = label.size()
        image, label = image.cuda(), label.cuda()
        pred, aux = model(image)
        # print("#"*20, pred.shape, label.shape)

        if "aux_loss" in cfg["net"].keys():
            loss = criterion([pred, aux], label)
        else:
            loss = criterion(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
# # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#       # gather all index_l
        # index_l = index_l.cuda()
        # index_l_all = [torch.zeros_like(index_l) for _ in range(world_size)]
        # dist.all_gather(index_l_all, index_l)
        # if rank == 0:
        #     print("="*10, index_l.shape, index_l)
        #     print("*"*10, len(index_l_all), index_l_all)
        
# # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # gather all loss from different gpus
        reduced_loss = loss.clone().detach()
        dist.all_reduce(reduced_loss)
        losses.update(reduced_loss.item())

        batch_end = time.time()
        batch_times.update(batch_end - batch_start)

        # if i_iter in print_freq_lst and rank == 0:
        if step in print_freq_lst and rank == 0:
            logger.info(
                "Epoch/Iter [{}:{:2}/{:3}]\t"
                "Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                # "LR {lr.val:.5f} ({lr.avg:.5f})\t".format(
                "LR {lr.val:.5f} \t".format(
                    cfg["trainer"]["epochs"],
                    epoch,
                    step,
                    # i_iter,
                    # cfg["trainer"]["epochs"] * len(data_loader),
                    data_time=data_times,
                    batch_time=batch_times,
                    loss=losses,
                    lr=learning_rates,
                )
            )
            if tb_logger is not None:
                tb_logger.add_scalar("lr", learning_rates.avg, i_iter)
                tb_logger.add_scalar("Loss", losses.avg, i_iter)

    return losses.avg

def validate(
    model,
    data_loader,
    epoch,
    logger,
    cfg
):
    model.eval()
    data_loader.sampler.set_epoch(epoch)

    num_classes, ignore_label = (
        cfg["net"]["num_classes"],
        cfg["dataset"]["ignore_label"],
    )
    rank, world_size = dist.get_rank(), dist.get_world_size()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    for step, batch in enumerate(data_loader):
        _, images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()
        batch_size, h, w = labels.shape

        with torch.no_grad():
            output, _ = model(images)

        # get the output produced by model_teacher
        output = output.data.max(1)[1].cpu().numpy()
        target_origin = labels.cpu().numpy()

        # start to calculate miou
        intersection, union, target = intersectionAndUnion(
            output, target_origin, num_classes, ignore_label
        )

        # # return ndarray, b*clas
        # print("="*20, type(intersection), type(union), type(target), intersection, union, target)

        # gather all validation information
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)

    if rank == 0:
        for i, iou in enumerate(iou_class):
            logger.info(" [Test] -  class [{}] IoU {:.2f}".format(i, iou * 100))
        # logger.info(" - <<Test>> - epoch {} mIoU {:.2f}".format(epoch, mIoU * 100))

    return mIoU


def validate_citys(
    model,
    data_loader,
    epoch,
    logger,
    cfg
):
    model.eval()
    data_loader.sampler.set_epoch(epoch)

    num_classes = cfg["net"]["num_classes"]
    ignore_label = cfg["dataset"]["ignore_label"]
    if cfg["dataset"]["val"].get("crop", False):
        crop_size,_ = cfg["dataset"]["val"]["crop"].get("size", [769, 769])
    else:
        crop_size = 769

    rank, world_size = dist.get_rank(), dist.get_world_size()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    for step, batch in enumerate(data_loader):
        _, images, labels = batch
        images = images.cuda()
        labels = labels.long()
        batch_size, h, w = labels.shape

        with torch.no_grad():
            final = torch.zeros(batch_size, num_classes, h, w).cuda()
            row = 0
            while row < h:
                col = 0
                while col < w:
                    pred, _ = model(images[:, :, row: min(h, row + crop_size), col: min(w, col + crop_size)])
                    final[:, :, row: min(h, row + crop_size), col: min(w, col + crop_size)] += pred.softmax(dim=1)
                    col += int(crop_size * 2 / 3)
                row += int(crop_size * 2 / 3)
            # get the output
            output = final.argmax(dim=1).cpu().numpy()
            target_origin = labels.numpy()
            # print("="*50, output.shape, output.dtype, target_origin.shape, target_origin.dtype)

        # start to calculate miou
        intersection, union, target = intersectionAndUnion(
            output, target_origin, num_classes, ignore_label
        )

        # # return ndarray, b*clas
        # print("="*20, type(intersection), type(union), type(target), intersection, union, target)

        # gather all validation information
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)

    if rank == 0:
        for i, iou in enumerate(iou_class):
            logger.info(" [Test] -  class [{}] IoU {:.2f}".format(i, iou * 100))
        # logger.info(" - <<Test>> - epoch {} mIoU {:.2f}".format(epoch, mIoU * 100))

    return mIoU


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", default=None, type=int)
    args = parser.parse_args()
    main(args)
