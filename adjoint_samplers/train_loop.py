# Copyright (c) Meta Platforms, Inc. and affiliates.

from omegaconf import DictConfig

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics.aggregation import MeanMetric

import adjoint_samplers.utils.train_utils as train_utils
from adjoint_samplers.components.matcher import Matcher

import torch.nn.functional as F

import math


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

    matcher._epoch_logw = []
    matcher._fresh_count = 0

    for _ in range(M):
        x0 = source.sample([B,]).to(device)
        timesteps = train_utils.get_timesteps(**cfg.timesteps).to(device)
        matcher.populate_buffer(x0, timesteps, is_asbs_init_stage)

    psi_stats = None
    if getattr(matcher, "potential", None) is not None and hasattr(matcher, "write_sb_logw"):
        # The first corrector stage still corresponds to a zero corrector: the
        # network has not been trained, so the bridge weight uses ψ = 0.
        zero_psi = is_asbs_init_stage or (
            getattr(matcher, "sb_on_corrector", False)
            and not train_utils.corrector_has_been_trained(epoch, cfg)
        )
        if not zero_psi:
            psi_stats = matcher.fit_potential(cfg.train_batch_size, device)
            if psi_stats is not None:
                matcher._psi_ready = True
        matcher.write_sb_logw(device, zero_psi)

    score_stats = None
    if getattr(matcher, "cond_score", None) is not None and hasattr(matcher, "fit_cond_score"):
        score_stats = matcher.fit_cond_score(cfg.train_batch_size, device)

    dataloader = matcher.build_dataloader(cfg.train_batch_size)
    epoch_loss = MeanMetric().to(device, non_blocking=True)

    cv_bias = MeanMetric().to(device, non_blocking=True)
    cv_var = MeanMetric().to(device, non_blocking=True)
    raw_var = MeanMetric().to(device, non_blocking=True)
    target_mag = MeanMetric().to(device, non_blocking=True)
    relative_bias = MeanMetric().to(device, non_blocking=True)
    cv_mse_cost = MeanMetric().to(device, non_blocking=True)
    var_gain = MeanMetric().to(device, non_blocking=True)
    saw_cv = False
    phi_r2 = MeanMetric().to(device, non_blocking=True)

    # lambda_scv_metric = MeanMetric().to(device, non_blocking=True)

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
    for itr in range(cfg.train_itr_per_epoch):
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

            t_in, xt_in = input
            m = matcher.cond_mean(t_in.detach(), xt_in.detach())
            loss_m = F.mse_loss(m, phi.detach())
            matcher.cond_optimizer.zero_grad()
            loss_m.backward()
            matcher.cond_optimizer.step()
            ss_res = (phi.detach() - m.detach()).pow(2).mean()
            ss_tot = phi.detach().var(unbiased=False)
            phi_r2.update((1.0 - ss_res / (ss_tot + 1e-8)).item())

            # Step A: fit the neural SCV to the current residual u + a
            target_omega = (output - target).detach()
            loss_omega = F.mse_loss(phi, target_omega)
            matcher.scv_optimizer.zero_grad()
            loss_omega.backward()
            matcher.scv_optimizer.step()

            phi_det = phi.detach()
            residual = (output - target).detach()
            saw_cv = True

            bias = phi_det.mean()
            mag = target.detach().abs().mean()
            raw = residual.var(unbiased=False)
            controlled = (residual - phi_det).var(unbiased=False)
            target_mag.update(mag)
            relative_bias.update(bias.abs() / (mag + 1e-8))
            cv_mse_cost.update(bias ** 2)
            var_gain.update(raw - controlled)

            cv_bias.update(bias)
            cv_var.update(controlled)
            raw_var.update(raw)

            

            # Step B: drift target -(a - phi)
            # progress = (epoch * cfg.train_itr_per_epoch + itr) / (
            #     cfg.num_epochs * cfg.train_itr_per_epoch
            # )
            # lambda_scv = 0.5 * (1.0 + math.cos(math.pi * progress))
            # lambda_scv_metric.update(lambda_scv)

            # target = (target + (lambda_scv * phi).detach()).detach()

            target = (target + phi.detach()).detach()

        optimizer.zero_grad()
        loss = loss_scale * ((output - target) ** 2).mean()
        loss.backward()

        if cfg.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e20)

        if cfg.get("clip_grad_value"):
            torch.nn.utils.clip_grad_value_(model.parameters(), cfg.clip_grad_value)

        optimizer.step()

        epoch_loss.update(loss.item())
        # if lr_schedule:
        #     lr_schedule.step()
        # if hasattr(matcher, "scv_scheduler"):
        #     matcher.scv_scheduler.step()

    stats = {"loss": float(epoch_loss.compute().detach().cpu())}
    if psi_stats:
        stats.update(psi_stats)
    if score_stats:
        stats.update(score_stats)
    if getattr(matcher, "_epoch_logw", None):
        logw = torch.cat(matcher._epoch_logw).double()
        w = (logw - logw.max()).exp()
        ess = (w.sum() ** 2) / w.square().sum().clamp_min(1e-30)
        stats["ess"] = float((ess / logw.numel()).cpu())
        stats["logZ"] = float((torch.logsumexp(logw, 0) - torch.log(torch.tensor(logw.numel(), dtype=torch.float64))).cpu())

    if saw_cv:
        stats["cv_bias"] = float(cv_bias.compute().detach().cpu())
        stats["cv_var"] = float(cv_var.compute().detach().cpu())
        stats["raw_var"] = float(raw_var.compute().detach().cpu())
        stats["target_mag"] = float(target_mag.compute().detach().cpu())
        stats["relative_bias"] = float(relative_bias.compute().detach().cpu())
        stats["cv_mse_cost"] = float(cv_mse_cost.compute().detach().cpu())
        stats["var_gain"] = float(var_gain.compute().detach().cpu())
        # stats["lambda_scv"] = float(lambda_scv_metric.compute().detach().cpu())
        stats["phi_r2"] = float(phi_r2.compute().detach().cpu())
    return stats
