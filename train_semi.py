import argparse
import yaml
import os, sys
import os.path as osp
import pprint
import time
import pickle

import random
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from imas.dataset.cutmix_tensor import cut_mix, cut_mix_using_v, cut_mix_by_hardness_for_prob, cut_mix_by_hardness, cut_mix_by_hardness_beta
from imas.dataset.builder import get_loader
from imas.models.model_helper import ModelBuilder
from imas.utils.dist_helper import setup_distributed
from imas.utils.loss_helper import get_criterion, compute_unsupervised_loss_by_threshold, compute_ulb_hardness_all
from imas.utils.lr_helper import get_optimizer, get_scheduler
from imas.utils.utils import AverageMeter, intersectionAndUnion, load_state, label_onehot
from imas.utils.utils import init_log, get_rank, get_world_size, set_random_seed, setup_default_logging
from imas.dataset.hardness import HardnessWrite

import warnings 
warnings.filterwarnings('ignore')


def main(in_args):
    args = in_args
    if args.seed is not None:
        # print("set random seed to", args.seed)
        set_random_seed(args.seed, deterministic=True)
        # set_random_seed(args.seed)
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    # cudnn.enabled = True
    # cudnn.benchmark = True
    rank, word_size = setup_distributed(port=args.port)

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
    # setup logger and csv paths
    if rank == 0:
        logger, curr_timestr = setup_default_logging("global", cfg["log_path"])
        csv_path = os.path.join(cfg["log_path"], "seg_{}_stat.csv".format(curr_timestr))
        csv_path_for_hardness = os.path.join(cfg["log_path"], "seg_{}_hardness.csv".format(curr_timestr))
    else:
        logger, curr_timestr = None, ""
        csv_path, csv_path_for_hardness = None, None
    # make sure all folders and csv handler are correctly created on rank ==0.
    dist.barrier()

    # create hardness instance
    if "pascal" in cfg["dataset"]["type"]:
        num_ulb = 10582 - cfg["dataset"]["n_sup"]
    elif "cityscapes" in cfg["dataset"]["type"]:
        num_ulb = 2975 - cfg["dataset"]["n_sup"]
    else:
        ValueError

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
    # 2. prepare model 1
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

    ###########################
    # 3. data
    ###########################
    sup_loss_fn = get_criterion(cfg)

    train_loader_sup, train_loader_unsup, val_loader = get_loader(cfg, seed=args.seed)

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

    ###########################
    # 5. prepare model more
    ###########################
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    # Teacher model
    model_teacher = ModelBuilder(cfg["net"])
    model_teacher.cuda()
    model_teacher = torch.nn.parallel.DistributedDataParallel(
        model_teacher,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    for p in model_teacher.parameters():
        p.requires_grad = False

    # initialize teacher model -- not neccesary if using warmup
    with torch.no_grad():
        for t_params, s_params in zip(model_teacher.parameters(), model.parameters()):
            t_params.data = s_params.data

    ######################################
    # 6. resume
    ######################################
    last_epoch = 0
    best_prec = 0
    best_epoch = -1
    best_prec_stu = 0
    best_epoch_stu = -1
    # auto_resume > pretrain
    if cfg["saver"].get("auto_resume", False):
        lastest_model = os.path.join(cfg["save_path"], "ckpt.pth")
        if not os.path.exists(lastest_model):
            "No checkpoint found in '{}'".format(lastest_model)
        else:
            print(f"Resume model from: '{lastest_model}'")
            best_prec, last_epoch = load_state(
                lastest_model, model, optimizer=optimizer, key="model_state"
            )
            _, _ = load_state(
                lastest_model, model_teacher, optimizer=optimizer, key="teacher_state"
            )

    optimizer_start = get_optimizer(params_list, cfg_optim)
    lr_scheduler = get_scheduler(
        cfg_trainer, len(train_loader_sup), optimizer_start, start_epoch=last_epoch
    )

    ######################################
    # 7. training loop
    ######################################
    if rank == 0:
        logger.info('-------------------------- start training --------------------------')
    # Start to train model
    for epoch in range(last_epoch, cfg_trainer["epochs"]):
        # Training
        res_loss_sup, res_loss_unsup, res_hardness_dict = train(
            model,
            model_teacher,
            optimizer,
            lr_scheduler,
            sup_loss_fn,
            train_loader_sup,
            train_loader_unsup,
            epoch,
            tb_logger,
            logger,
            cfg
        )

        # Update hardness
        if rank == 0 and epoch > cfg["trainer"].get("sup_only_epoch", 0):
            # record hardness for further analysis
            tmp_lst_hardness = {str(x):res_hardness_dict.get(x, None) for x in range(num_ulb)}
            tmp_df_hardness = pd.DataFrame(data=tmp_lst_hardness, index=range(epoch, epoch+1))
            if epoch > 0 and osp.exists(csv_path_for_hardness):
                tmp_df_hardness.to_csv(csv_path_for_hardness, mode='a', header=None, index_label='epoch')
            else:
                tmp_df_hardness.to_csv(csv_path_for_hardness, index_label='epoch')
        # # make sure hardness is updated!!
        # dist.barrier()

        # Validation
        # prec_stu = validate(model, val_loader, epoch, logger, cfg)
        # prec_tea = validate(model_teacher, val_loader, epoch, logger, cfg)
        prec_stu = validate_citys(model, val_loader, epoch, logger, cfg)
        prec_tea = validate_citys(model_teacher, val_loader, epoch, logger, cfg)

        prec = prec_tea

        if rank == 0:
            state = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "teacher_state": model_teacher.state_dict(),
                "best_miou": best_prec,
            }
            if prec_stu > best_prec_stu:
                best_prec_stu = prec_stu
                best_epoch_stu = epoch

            if prec > best_prec:
                best_prec = prec
                best_epoch = epoch
                state["best_miou"] = prec
                torch.save(state, osp.join(cfg["save_path"], "ckpt_best.pth"))

            torch.save(state, osp.join(cfg["save_path"], "ckpt.pth"))
            # save statistics
            tmp_results = {
                        'loss_lb': res_loss_sup,
                        'loss_ub': res_loss_unsup,
                        'miou_stu': prec_stu,
                        'miou_tea': prec_tea,
                        "best": best_prec,
                        "best-stu":best_prec_stu}
            data_frame = pd.DataFrame(data=tmp_results, index=range(epoch, epoch+1))
            if epoch > 0 and osp.exists(csv_path):
                data_frame.to_csv(csv_path, mode='a', header=None, index_label='epoch')
            else:
                data_frame.to_csv(csv_path, index_label='epoch')
            
            logger.info(" <<Test>> - Epoch: {}.  MIoU: {:.2f}/{:.2f}.  \033[34mBest-STU:{:.2f}/{}  \033[31mBest-EMA: {:.2f}/{}\033[0m".format(epoch, 
                prec_stu * 100, prec_tea * 100, best_prec_stu * 100, best_epoch_stu, best_prec * 100, best_epoch))
            if tb_logger is not None:
                tb_logger.add_scalar("mIoU val", prec, epoch)


def train(
    model,
    model_teacher,
    optimizer,
    lr_scheduler,
    sup_loss_fn,
    loader_l,
    loader_u,
    epoch,
    tb_logger,
    logger,
    cfg,
):
    
    ema_decay_origin = cfg["net"]["ema_decay"]
    rank, world_size = dist.get_rank(), dist.get_world_size()
    model.train()
    
    # data loader
    loader_l.sampler.set_epoch(epoch)
    loader_u.sampler.set_epoch(epoch)
    loader_l_iter = iter(loader_l)
    loader_u_iter = iter(loader_u)
    assert len(loader_l) == len(
        loader_u
    ), f"labeled data {len(loader_l)} unlabeled data {len(loader_u)}, imbalance!"

    # metric indicators
    sup_losses = AverageMeter(20)
    uns_losses = AverageMeter(20)
    batch_times = AverageMeter(20)
    learning_rates = AverageMeter(20)
    meter_high_pseudo_ratio = AverageMeter(20)
    
    # print freq 4 times for a epoch
    print_freq = len(loader_u) // 8 # 8 for semi 4 for sup
    print_freq_lst = [i * print_freq for i in range(1,8)]
    print_freq_lst.append(len(loader_u) -1)

    # create the hardness return
    dict_hardness = dict()

    # start iterations
    model.train()
    model_teacher.eval()
    for step in range(len(loader_l)):
        batch_start = time.time()
        i_iter = epoch * len(loader_l) + step # total iters till now
        lr = lr_scheduler.get_lr()
        learning_rates.update(lr[0])
        lr_scheduler.step() # lr is updated at the iteration level

        # obtain labeled and unlabeled data
        _, image_l, label_l = loader_l_iter.next()
        batch_size, h, w = label_l.size()
        image_l, label_l = image_l.cuda(), label_l.cuda()
        index_u, image_u_weak, image_u_aug, _ = loader_u_iter.next()
        index_u, image_u_weak, image_u_aug = index_u.cuda(), image_u_weak.cuda(), image_u_aug.cuda()
        
        # start the training
        if epoch < cfg["trainer"].get("sup_only_epoch", 0):
            # forward
            pred, aux = model(image_l)
            # supervised loss
            if "aux_loss" in cfg["net"].keys():
                sup_loss = sup_loss_fn([pred, aux], label_l)
                del aux
            else:
                sup_loss = sup_loss_fn(pred, label_l)
                del pred

            # no unlabeled data during the warmup period
            unsup_loss = torch.tensor(0.0).cuda()
            tensor_hardness = None
            pseduo_high_ratio = torch.tensor(0.0).cuda()

        else:
            # 1. generate pseudo labels and hardness firstly
            p_threshold = cfg["trainer"]["unsupervised"].get("threshold", 0.95)
            with torch.no_grad():
                model.eval()
                pred_u_stu, _ = model(image_u_weak.detach())
                model_teacher.eval()
                pred_u, _ = model_teacher(image_u_weak.detach())
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
                if cfg["dataset"]["train"].get("hardness_aware", False):
                    
                    flag_using_hardnness = True
                    # setup detials of using hardness
                    flag_cal_iou_weighted = cfg["dataset"]["train"]["hardness_aware"].get("flag_cal_iou_weighted", True)
                    flag_cal_iou_ignore_bg = cfg["dataset"]["train"]["hardness_aware"].get("flag_cal_iou_ignore_bg", True)
                    flag_using_v1 = cfg["dataset"]["train"]["hardness_aware"].get("flag_using_v1", True)
                    flag_hardness_weighted_loss = cfg["dataset"]["train"]["hardness_aware"].get("flag_hardness_weighted_loss", True)
                    flag_cmix_trigger_by_hard = cfg["dataset"]["train"]["hardness_aware"].get("flag_cmix_trigger_by_hard", False)
                    flag_augs_mixup_by_hard = cfg["dataset"]["train"]["hardness_aware"].get("flag_augs_mixup_by_hard", False)
                    flag_mapping_random = cfg["dataset"]["train"]["hardness_aware"].get("flag_mapping_random", False)
                    flag_mapping_gaussian = cfg["dataset"]["train"]["hardness_aware"].get("flag_mapping_gaussian", False)
                else:
                    flag_using_hardnness = False
                    # still record hardness even if not using hardness
                    flag_cal_iou_weighted = True
                    flag_cal_iou_ignore_bg = True
                    flag_using_v1 = True
                    flag_hardness_weighted_loss = False
                    flag_cmix_trigger_by_hard = False
                    flag_augs_mixup_by_hard = False
                    flag_mapping_gaussian = False
                    flag_mapping_random = False

                hardness_v1, hardness_v2, hardness_ratio = compute_ulb_hardness_all(pred_u_stu, pred_u, p_threshold, 
                    flag_using_cls_weighted_iou=flag_cal_iou_weighted,
                    flag_ignoring_background=flag_cal_iou_ignore_bg)
                pseduo_high_ratio = hardness_ratio.mean()
                
                if cfg["trainer"]["unsupervised"].get("flag_ema_pseudo", True):
                    pred_u = F.softmax(pred_u, dim=1)
                    logits_u_aug, label_u_aug = torch.max(pred_u, dim=1)
                    del pred_u, pred_u_stu
                else:
                    pred_u_stu = F.softmax(pred_u_stu, dim=1)
                    logits_u_aug, label_u_aug = torch.max(pred_u_stu, dim=1)
                    del pred_u, pred_u_stu
                
                # pred_u = F.softmax(pred_u, dim=1)
                # pred_u_stu = F.softmax(pred_u_stu, dim=1)
                # # pred_u_mix = 0.5 * pred_u + 0.5 * pred_u_stu
                # pred_u_mix = ema_decay_origin * pred_u + (1 - ema_decay_origin) * pred_u_stu
                # logits_u_aug, label_u_aug = torch.max(pred_u_mix, dim=1)
                # del pred_u, pred_u_stu, pred_u_mix

            model.train()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             
            
            # 2. obtain curr and hist hardness
            if flag_using_v1:
                tensor_hardness = hardness_v1
            else:
                tensor_hardness = hardness_v2
            cur_hardness = tensor_hardness.detach().cpu().numpy().tolist()
            hardness_avg = np.mean(cur_hardness)
            
            # 3. apply mixup between strong and weak
            flag_loss_intensity_aug = False
            if cfg["dataset"]["train"].get("strong_aug", False):
                flag_loss_intensity_aug = True
                if flag_using_hardnness:
                    if flag_augs_mixup_by_hard:
                        mixup_ratio = np.array(cur_hardness)
                        mixup_ratio = mixup_ratio.reshape((-1,1,1,1)).astype(np.float32)
                        mixup_ratio = torch.from_numpy(mixup_ratio).cuda()
                        # image_u_weak ==> cur, image_u_aug ===> past
                        image_u_aug = image_u_aug * mixup_ratio + image_u_weak * (1.0 - mixup_ratio)
                    else:
                        cur_hardness_arr = np.array(cur_hardness)
                        generate_radomness = np.random.uniform(size=cur_hardness_arr.shape)
                        mask = generate_radomness < cur_hardness_arr
                        mask_float = mask.reshape((-1,1,1,1)).astype(np.float32)
                        mask_float = torch.from_numpy(mask_float).cuda()
                        image_u_aug = image_u_aug * mask_float + image_u_weak * (1.0 - mask_float)

            # 4. apply cutmix on image_u_weak
            flag_loss_cutmix_aug = False
            if flag_cmix_trigger_by_hard:
                trigger_prob = hardness_avg
            else:
                trigger_prob = cfg["trainer"]["unsupervised"].get("use_cutmix_trigger_prob", 1.0)
            if np.random.uniform(0, 1) < trigger_prob and cfg["trainer"]["unsupervised"].get("use_cutmix", False):
                flag_loss_cutmix_aug = True
                if cfg["trainer"]["unsupervised"].get("use_cutmix_beta", False):
                        image_u_aug_cm, label_u_aug_cm, logits_u_aug_cm = cut_mix(
                            image_u_weak,
                            label_u_aug,
                            logits_u_aug,
                        )
                else:
                    c_range = cfg["trainer"]["unsupervised"].get("use_cutmix_range", [0.3, 1/3])
                    if flag_using_hardnness:
                        image_u_aug_cm, label_u_aug_cm, logits_u_aug_cm = cut_mix_by_hardness(
                        image_u_weak,
                        label_u_aug,
                        logits_u_aug, 
                        hardness=cur_hardness,
                        scale=c_range, 
                        flag_hardness_random=flag_mapping_random, 
                        flag_hardness_gaussion=flag_mapping_gaussian)
                    else:
                        image_u_aug_cm, label_u_aug_cm, logits_u_aug_cm = cut_mix_using_v(
                            image_u_weak,
                            label_u_aug,
                            logits_u_aug,
                            scale=c_range
                        )

            # 5. forward concated labeled + unlabeld into student networks
            num_labeled = len(image_l)
            if flag_loss_cutmix_aug:
                if flag_loss_intensity_aug:
                    pred_all, aux_all = model(torch.cat((image_l, image_u_weak, image_u_aug_cm, image_u_aug), dim=0))
                    del image_l, image_u_weak, image_u_aug_cm, image_u_aug
                    pred_l= pred_all[:num_labeled]
                    pred_u_weak, pred_u_strong_cm, pred_u_strong = pred_all[num_labeled:].chunk(3)
                    del pred_all
                else:
                    pred_all, aux_all = model(torch.cat((image_l, image_u_weak, image_u_aug_cm), dim=0))
                    del image_l, image_u_weak, image_u_aug_cm
                    pred_l= pred_all[:num_labeled]
                    pred_u_weak, pred_u_strong_cm = pred_all[num_labeled:].chunk(2)
                    del pred_all
            else:
                pred_all, aux_all = model(torch.cat((image_l, image_u_weak, image_u_aug), dim=0))
                del image_l, image_u_weak, image_u_aug
                pred_l= pred_all[:num_labeled]
                pred_u_weak, pred_u_strong = pred_all[num_labeled:].chunk(2)
                del pred_all
            
            # 6. supervised loss
            if "aux_loss" in cfg["net"].keys():
                aux = aux_all[:num_labeled]
                sup_loss = sup_loss_fn([pred_l, aux], label_l)
                del aux_all, aux
            else:
                # sup_loss = sup_loss_fn(pred_l, label_l.clone())
                sup_loss = sup_loss_fn(pred_l, label_l)

            # 7. unsupervised loss
            if flag_hardness_weighted_loss:
                input_hardness = tensor_hardness
            else:
                input_hardness = None

            if flag_loss_cutmix_aug and flag_loss_intensity_aug:
                unsup_loss_aug = compute_unsupervised_loss_by_threshold(
                        pred_u_strong, label_u_aug.detach(),
                        logits_u_aug.detach(), thresh=p_threshold, hardness_tensor=input_hardness)
                unsup_loss_cm = compute_unsupervised_loss_by_threshold(
                        pred_u_strong_cm, label_u_aug_cm.detach(),
                        logits_u_aug_cm.detach(), thresh=p_threshold, hardness_tensor=input_hardness)
                unsup_loss = (unsup_loss_aug + unsup_loss_cm) * 0.5

                del pred_l, pred_u_strong, pred_u_weak, label_u_aug, logits_u_aug, label_u_aug_cm, logits_u_aug_cm
            elif flag_loss_cutmix_aug:
                unsup_loss = compute_unsupervised_loss_by_threshold(
                        pred_u_strong_cm, label_u_aug_cm.detach(),
                        logits_u_aug_cm.detach(), thresh=p_threshold, hardness_tensor=input_hardness)
                del pred_l, pred_u_weak, label_u_aug, logits_u_aug, label_u_aug_cm, logits_u_aug_cm

            elif flag_loss_intensity_aug:
                unsup_loss = compute_unsupervised_loss_by_threshold(
                        pred_u_strong, label_u_aug.detach(),
                        logits_u_aug.detach(), thresh=p_threshold, hardness_tensor=input_hardness)
                del pred_l, pred_u_strong, pred_u_weak, label_u_aug, logits_u_aug
            else:
                unsup_loss = compute_unsupervised_loss_by_threshold(
                        pred_u_strong, label_u_aug.detach(),
                        logits_u_aug.detach(), thresh=p_threshold, hardness_tensor=input_hardness)
                del pred_l, pred_u_strong, pred_u_weak, label_u_aug, logits_u_aug

            unsup_loss *= cfg["trainer"]["unsupervised"].get("loss_weight", 1.0)            

        loss = sup_loss + unsup_loss

        # 8. update student model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 9. update teacher model with EMA
        with torch.no_grad():
            if epoch > cfg["trainer"].get("sup_only_epoch", 0):
                ema_decay = min(
                    1
                    - 1
                    / (
                        i_iter
                        - len(loader_l) * cfg["trainer"].get("sup_only_epoch", 0)
                        + 1
                    ),
                    ema_decay_origin,
                )
            else:
                ema_decay = 0.0
            # print("="*20, i_iter, ema_decay)
            # update bn of teachers
            for param_train, param_eval in zip(model.parameters(), model_teacher.parameters()):
                # param_eval.copy_(param_eval * ema_decay + param_train.detach() * (1 - ema_decay))
                param_eval.data = param_eval.data * ema_decay + param_train.data * (1 - ema_decay)

            for buffer_train, buffer_eval in zip(model.buffers(), model_teacher.buffers()):
                # buffer_eval.copy_(buffer_train)
                buffer_eval.data = buffer_eval.data * ema_decay + buffer_train.data * (1 - ema_decay)
                # buffer_eval.data = buffer_train.data

        # 10. gather all loss from different gpus
        reduced_sup_loss = sup_loss.clone().detach()
        dist.all_reduce(reduced_sup_loss)
        sup_losses.update(reduced_sup_loss.item() / world_size)

        reduced_uns_loss = unsup_loss.clone().detach()
        dist.all_reduce(reduced_uns_loss)
        uns_losses.update(reduced_uns_loss.item() / world_size)

        reduced_pseudo_high_ratio = pseduo_high_ratio.clone().detach()
        dist.all_reduce(reduced_pseudo_high_ratio)
        meter_high_pseudo_ratio.update(reduced_pseudo_high_ratio.item() / world_size)

        # 11. gather all hardness from different gpus
        log_hardness_avg = 0.0
        if tensor_hardness is not None:
            index_u_all = [torch.zeros_like(index_u) for _ in range(world_size)]
            dist.all_gather(index_u_all, index_u)
            hardness_all = [torch.zeros_like(tensor_hardness) for _ in range(world_size)]
            dist.all_gather(hardness_all, tensor_hardness)
            if rank == 0:
                # print("="*50)
                # print(len(index_u_all), index_u_all[0].shape)
                # print(index_u_all)
                index_u_all = torch.cat(index_u_all)
                hardness_all = torch.cat(hardness_all)
                log_hardness_avg = hardness_all.mean().item()
                # print(index_u_all.shape, hardness_all.shape)
                # print(index_u_all)
                # print(hardness_all)
                # print("="*50)
                index_u_all = index_u_all.cpu().numpy().tolist()
                hardness_all = hardness_all.cpu().numpy().tolist()
                tmp_dict = {x:y for x,y in zip(index_u_all, hardness_all)}
                dict_hardness.update(tmp_dict)

        # 12. print log information
        batch_end = time.time()
        batch_times.update(batch_end - batch_start)
        # if i_iter % 10 == 0 and rank == 0:
        if step in print_freq_lst and rank == 0:
            logger.info(
                "Epoch/Iter [{}:{:3}/{:3}].  "
                "AvgHard:{:.3}.  "
                "Sup:{sup_loss.val:.3f}({sup_loss.avg:.3f})  "
                "Uns:{uns_loss.val:.3f}({uns_loss.avg:.3f})  "
                "Pseudo:{high_ratio.val:.3f}({high_ratio.avg:.3f})  "
                "Time:{batch_time.avg:.2f}  "
                "LR:{lr.val:.5f}".format(
                    cfg["trainer"]["epochs"], epoch, step,
                    log_hardness_avg,
                    # i_iter, cfg["trainer"]["epochs"] * len(loader_l),
                    sup_loss=sup_losses,
                    uns_loss=uns_losses,
                    high_ratio=meter_high_pseudo_ratio,
                    batch_time=batch_times,
                    lr=learning_rates,
                )
            )
            if tb_logger is not None:
                tb_logger.add_scalar("lr", learning_rates.avg, i_iter)
                tb_logger.add_scalar("Sup Loss", sup_losses.avg, i_iter)
                tb_logger.add_scalar("Uns Loss", uns_losses.avg, i_iter)
                tb_logger.add_scalar("Pseudo ratio", meter_high_pseudo_ratio.avg, i_iter)
    
    return sup_losses.avg, uns_losses.avg, dict_hardness


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

        with torch.no_grad():
            output, _ = model(images)

        # get the output produced by model_teacher
        output = output.data.max(1)[1].cpu().numpy()
        target_origin = labels.cpu().numpy()

        # start to calculate miou
        intersection, union, target = intersectionAndUnion(
            output, target_origin, num_classes, ignore_label
        )

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
