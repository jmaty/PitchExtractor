#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging
import os
import os.path as osp
import shutil
from logging import StreamHandler

import torch
import yaml
from accelerate import Accelerator
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from munch import munchify

from meldataset import build_dataloader
from model import JDCNet
from optimizers import build_optimizer
from trainer import Trainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

torch.backends.cudnn.benchmark = True


def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, "r", encoding="utf-8") as f:
        train_list = f.readlines()
    with open(val_path, "r", encoding="utf-8") as f:
        val_list = f.readlines()

    # train_list = train_list[-500:]
    # val_list = train_list[:500]
    return train_list, val_list


def main():
    parser = argparse.ArgumentParser(description="Pitch and voicing training")
    parser.add_argument("config_path", type=str, help="path to config")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    parser.add_argument(
        "--precompute_f0",
        action="store_true",
        default=False,
        help="only precompute F0 features",
    )
    parser.add_argument(
        "--skip_check",
        action="store_true",
        default=False,
        help="skip F0 checking",
    )
    args = parser.parse_args()

    cfg = munchify(yaml.safe_load(open(args.config_path, encoding="utf-8")))
    log_dir = cfg.log_dir

    # Initialize Accelerator for distributed training
    acc = Accelerator(
        mixed_precision=cfg.get("mixed_precision", "no"),  # can be 'fp16', 'bf16', or 'no'
        gradient_accumulation_steps=cfg.get("grad_accum_steps", 1),
        # log_with="tensorboard",
        # project_dir=log_dir,
    )

    # Only create directories on main process
    if acc.is_main_process:
        if not osp.exists(log_dir):
            os.mkdir(log_dir)
        shutil.copy(args.config_path, osp.join(log_dir, osp.basename(args.config_path)))

    # Write logs only on main process
    if acc.is_main_process:
        writer = SummaryWriter(log_dir + "/tensorboard")
        file_handler = logging.FileHandler(osp.join(log_dir, "train.log"))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(levelname)s:%(asctime)s: %(message)s"))
        logger.addHandler(file_handler)
    else:
        writer = None

    # Set device
    device = acc.device

    train_list, val_list = get_data_path_list(cfg.train_data, cfg.val_data)

    train_dataloader = build_dataloader(
        train_list,
        batch_size=cfg.batch_size,
        num_workers=args.num_workers,
        dataset_config=cfg.get("dataset_params", {}),
        device=device,
    )

    val_dataloader = build_dataloader(
        val_list,
        batch_size=cfg.batch_size,
        validation=True,
        num_workers=args.num_workers // 2,
        device=device,
        dataset_config=cfg.get("dataset_params", {}),
    )

    if acc.is_main_process:
        tot_effect_batch = cfg.batch_size * acc.num_processes * acc.gradient_accumulation_steps
        logger.info("=" * 50)
        logger.info("Training Configuration")
        logger.info("=" * 50)
        logger.info("Device:                       %s", device)
        logger.info("Number of GPUs:               %d", acc.num_processes)
        logger.info("Mixed precision:              %s", acc.mixed_precision)
        logger.info("Batch size per GPU:           %d", cfg.batch_size)
        logger.info("Effective batch size:         %d", cfg.batch_size * acc.num_processes)
        logger.info("Gradient accumulation steps:  %d", acc.gradient_accumulation_steps)
        logger.info("Total effective batch:        %d", tot_effect_batch)
        logger.info("Pretrained model:             %s", cfg.get("pretrained_model", ""))
        logger.info("Training samples:             %d", len(train_list))
        logger.info("Validation samples:           %d", len(val_list))
        logger.info("Train batches per epoch:      %d", len(train_dataloader))
        logger.info("Validation batches per epoch: %d", len(val_dataloader))
        logger.info("=" * 50)
        logger.info("")

    if not args.skip_check:
        # Precompute all F0 for training and validation data
        if acc.is_main_process:
            logger.info("Checking if all F0 data is computed...")
        for _ in enumerate(train_dataloader):
            continue
        for _ in enumerate(val_dataloader):
            continue
        if acc.is_main_process:
            logger.info("All F0 data is computed.")

    # Exit if only precomputing F0
    if args.precompute_f0 and not args.skip_check:
        if acc.is_main_process:
            logger.info("F0 precomputed, exiting.")
        return 0

    # Define model
    model = JDCNet(num_class=1)  # num_class = 1 means regression

    # Automatic learning rate scaling based on number of GPUs
    base_lr = float(cfg.optimizer_params.lr)
    # Scale LR linearly with number of GPUs
    scaled_lr = base_lr * acc.num_processes

    # Adjust warmup based on scaling factor
    base_pct_start = float(cfg.optimizer_params.pct_start)
    if acc.num_processes == 1:
        # Single GPU: use config value
        pct_start = base_pct_start
    elif acc.num_processes == 2:
        # 2 GPUs: minimal or no warmup needed for 2× scaling
        pct_start = max(base_pct_start, 0.02)  # at least 2% warmup
    elif acc.num_processes <= 4:
        # 3-4 GPUs: moderate warmup for stability
        pct_start = max(base_pct_start, 0.05)  # at least 5% warmup
    else:
        # 5+ GPUs: longer warmup for large LR
        pct_start = max(base_pct_start, 0.1)  # at least 10% warmup

    if acc.is_main_process:
        warmup_steps = int(pct_start * cfg.epochs * len(train_dataloader))
        logger.info("Optimizer Configuration")
        logger.info("-" * 50)
        logger.info("Base learning rate:           %.6f", base_lr)
        logger.info("Scaled learning rate:         %.6f", scaled_lr)
        logger.info("LR scaling factor:            %d", acc.num_processes)
        logger.info("Warmup percentage:            %.1f%%", pct_start * 100)
        logger.info("Warmup steps:                 %d", warmup_steps)
        logger.info("-" * 50)
        logger.info("")

    scheduler_params = {
        "max_lr": scaled_lr,
        "pct_start": pct_start,
        "epochs": cfg.epochs,
        "steps_per_epoch": len(train_dataloader),
    }

    optimizer, scheduler = build_optimizer(
        {
            "params": model.parameters(),
            "optimizer_params": {},
            "scheduler_params": scheduler_params,
        }
    )

    # Prepare everything with accelerator
    model, optimizer, train_dataloader, val_dataloader, scheduler = acc.prepare(
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        scheduler,
    )

    criterion = {
        "l1": nn.SmoothL1Loss(),  # F0 loss (regression)
        "ce": nn.BCEWithLogitsLoss(),  # silence loss (binary classification)
    }

    trainer = Trainer(
        acc,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_config=cfg.loss_params,
    )

    if cfg.get("pretrained_model", "") != "":
        trainer.load_checkpoint(
            cfg.pretrained_model,
            load_only_params=cfg.get("load_only_params", False),
        )

    for epoch in range(1, cfg.epochs + 1):
        train_results = trainer.train_epoch()
        eval_results = trainer.eval_epoch()
        results = train_results.copy()
        results.update(eval_results)

        # Only log on main process
        if acc.is_main_process:
            logger.info("--- epoch %d ---", epoch)
            for key, value in results.items():
                if isinstance(value, float):
                    logger.info("%-15s: %.4f", key, value)
                    if writer is not None:
                        writer.add_scalar(key, value, epoch)
                else:
                    if writer is not None:
                        writer.add_figure(key, (value), epoch)

            if epoch % cfg.save_freq == 0:
                trainer.save_checkpoint(osp.join(log_dir, f"epoch_{epoch:05d}.pth"))

    return 0


if __name__ == "__main__":
    main()
