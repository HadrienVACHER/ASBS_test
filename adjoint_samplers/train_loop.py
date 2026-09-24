# Copyright (c) Meta Platforms, Inc. and affiliates.

from omegaconf import DictConfig

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics.aggregation import MeanMetric

import adjoint_samplers.utils.train_utils as train_utils
from adjoint_samplers.components.matcher import Matcher

import torch.nn.functional as F


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def train_one_epoch(
    matcher: Matcher,
    model: torch.nn.Module,
    source: torch.nn.Module,
    optimizer: Optimizer,
    lr_schedule: LRScheduler | None,
    epoch: int,
    device: str,
    cfg: DictConfig,
):
    # build dataloader
    B = cfg.resample_batch_size
    M = matcher.resample_size // (B * cfg.world_size)
    loss_scale = matcher.loss_scale

    is_asbs_init_stage = train_utils.is_asbs_init_stage(epoch, cfg)

    for _ in range(M):
        x0 = source.sample([B,]).to(device)
        timesteps = train_utils.get_timesteps(**cfg.timesteps).to(device)
        matcher.populate_buffer(x0, timesteps, is_asbs_init_stage)

    dataloader = matcher.build_dataloader(cfg.train_batch_size)
    epoch_loss = MeanMetric().to(device, non_blocking=True)

    loader = iter(cycle(dataloader))

    # model.train(True)
    # for _ in range(cfg.train_itr_per_epoch):
    #     optimizer.zero_grad()

    #     data = next(loader)

    #     input, target = matcher.prepare_target(data, device)
    #     output = model(*input)

    #     loss = loss_scale * ((output - target)**2).mean()
    #     loss.backward()

    #     if cfg.clip_grad_norm:
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e20)

    #     optimizer.step()

    #     epoch_loss.update(loss.item())
    #     if lr_schedule:
    #         lr_schedule.step()

    model.train(True)
    for _ in range(cfg.train_itr_per_epoch):
        data = next(loader)

        input, target, *extra = matcher.prepare_target(data, device)
        output = model(*input)

        # if extra:
        #     phi = extra[0]
        #     t = input[0]

        #     # Step A: fit the gate to the current residual u + a
        #     c_t = matcher.temporal_gate(t)
        #     target_omega = (output - target).detach()
        #     loss_omega = F.mse_loss(c_t * phi, target_omega)
        #     matcher.gate_optimizer.zero_grad()
        #     loss_omega.backward()
        #     matcher.gate_optimizer.step()

        #     # Step B: drift target -(a - c * phi)
        #     target = (target + c_t.detach() * phi).detach()

        if extra:
            phi = extra[0]

            # Step A: fit the neural SCV to the current residual u + a
            target_omega = (output - target).detach()
            loss_omega = F.mse_loss(phi, target_omega)
            matcher.scv_optimizer.zero_grad()
            loss_omega.backward()
            matcher.scv_optimizer.step()

            # Step B: drift target -(a - phi)
            target = (target + phi.detach()).detach()

        optimizer.zero_grad()
        loss = loss_scale * ((output - target) ** 2).mean()
        loss.backward()

        optimizer.zero_grad()
        loss = loss_scale * ((output - target) ** 2).mean()
        loss.backward()

        if cfg.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e20)

        optimizer.step()

        epoch_loss.update(loss.item())
        if lr_schedule:
            lr_schedule.step()

    return float(epoch_loss.compute().detach().cpu())
