# Copyright (c) Meta Platforms, Inc. and affiliates.

import sys
import traceback
import hydra
import numpy as np
import termcolor

from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from adjoint_samplers.components.sde import ControlledSDE, sdeint
from adjoint_samplers.train_loop import train_one_epoch
import adjoint_samplers.utils.train_utils as train_utils
import adjoint_samplers.utils.distributed_mode as distributed_mode

# from adjoint_samplers.components.model import TemporalGate

from adjoint_samplers.components.model import EGNNPotential

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')


cudnn.benchmark = True


def red(content): return termcolor.colored(str(content),"red",attrs=["bold"])
def green(content): return termcolor.colored(str(content),"green",attrs=["bold"])
def blue(content): return termcolor.colored(str(content),"blue",attrs=["bold"])
def cyan(content): return termcolor.colored(str(content),"cyan",attrs=["bold"])
def yellow(content): return termcolor.colored(str(content),"yellow",attrs=["bold"])
def magenta(content): return termcolor.colored(str(content),"magenta",attrs=["bold"])


@hydra.main(config_path="configs", config_name="train.yaml", version_base="1.1")
def main(cfg):

    try:
        train_utils.setup(cfg)
        print(str(cfg))

        device = "cuda"

        # fix the seed for reproducibility
        seed = cfg.seed + distributed_mode.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)

        print("Instantiating energy...")
        energy = hydra.utils.instantiate(cfg.energy, device=device)


        print("Instantiating source...")
        source = hydra.utils.instantiate(cfg.source, device=device)


        print('Instantiating model...')
        ref_sde = hydra.utils.instantiate(cfg.ref_sde).to(device)
        controller = hydra.utils.instantiate(cfg.controller).to(device)
        sde = ControlledSDE(ref_sde, controller).to(device)


        if "corrector" in cfg:
            print('Instantiating corrector & corrector matcher...')
            corrector = hydra.utils.instantiate(cfg.corrector).to(device)
            corrector_matcher = hydra.utils.instantiate(cfg.corrector_matcher, sde=sde)
        else:
            corrector = corrector_matcher = None


        print("Instantiating grad of costs...")
        grad_term_cost = hydra.utils.instantiate(
            cfg.term_cost,
            corrector=corrector,
            energy=energy,
            ref_sde=ref_sde,
            source=source,
        )


        print("Instantiating adjoint matcher...")
        adjoint_matcher = hydra.utils.instantiate(
            cfg.adjoint_matcher,
            grad_term_cost=grad_term_cost,
            sde=sde,
        )

        # Stein control variate is off. prepare_target then returns the plain
        # adjoint target. The bridge weight is the only change to ASBS.
        # from adjoint_samplers.components.model import NeuralSCV, CondMean, CondScore, EGNNCondScore
        # neural_scv = NeuralSCV(dim=cfg.dim).to(device)
        # adjoint_matcher.neural_scv = neural_scv
        # adjoint_matcher.scv_optimizer = torch.optim.Adam(
        #     neural_scv.parameters(), lr=1e-3
        # )
        # cond_mean = CondMean(dim=cfg.dim).to(device)
        # adjoint_matcher.cond_mean = cond_mean
        # adjoint_matcher.cond_optimizer = torch.optim.Adam(
        #     cond_mean.parameters(), lr=1e-3
        # )
        # if cfg.get("n_particles") is not None:
        #     cond_score = EGNNCondScore(
        #         n_particles=int(cfg.n_particles),
        #         spatial_dim=int(cfg.spatial_dim),
        #         hidden_nf=128,
        #         n_layers=4,
        #     )
        # else:
        #     cond_score = CondScore(dim=int(cfg.dim))
        # adjoint_matcher.cond_score = cond_score.to(device)
        # adjoint_matcher.cond_score_steps = 20
        # adjoint_matcher.cond_score_optimizer = torch.optim.Adam(
        #     cond_score.parameters(), lr=1e-3,
        # )

        # Adjoint-sampling weights are the wrong terminal factor for ASBS.
        adjoint_matcher.use_is = corrector is None
        if corrector is not None and cfg.get("n_particles") is not None:
            print("Instantiating scalar potential...")
            c = cfg.corrector
            potential = EGNNPotential(
                n_particles=int(cfg.n_particles),
                spatial_dim=int(cfg.spatial_dim),
                hidden_nf=int(c.get("hidden_nf", 128)),
                n_layers=int(c.get("n_layers", 5)),
                act_fn=torch.nn.SiLU(),
                recurrent=bool(c.get("recurrent", True)),
                tanh=bool(c.get("tanh", True)),
                attention=bool(c.get("attention", True)),
                condition_time=True,
                agg=str(c.get("agg", "sum")),
            ).to(device)
            adjoint_matcher.potential = potential
            adjoint_matcher.potential_steps = 20
            adjoint_matcher.phi0_samples = 8
            adjoint_matcher.potential_optimizer = torch.optim.Adam(
                potential.parameters(), lr=1e-4,
            )
            # Same ψ and the same bridge weight on the corrector buffer.
            corrector_matcher.potential = potential
            corrector_matcher.potential_steps = adjoint_matcher.potential_steps
            corrector_matcher.phi0_samples = adjoint_matcher.phi0_samples
            corrector_matcher.potential_optimizer = adjoint_matcher.potential_optimizer
            corrector_matcher.grad_term_cost = grad_term_cost
            corrector_matcher.sb_on_corrector = True


        print("Instantiating optimizer...")
        if corrector is not None:
            optimizer = torch.optim.Adam([
                {'params': controller.parameters(), **cfg.adjoint_matcher.optim},
                {'params': corrector.parameters(), **cfg.corrector_matcher.optim},
            ])
        else:
            optimizer = torch.optim.Adam(
                controller.parameters(), **cfg.adjoint_matcher.optim,
            )

        # checkpoint_path = Path(cfg.checkpoint or "checkpoints/checkpoint_latest.pt")
        # checkpoint_path.parent.mkdir(exist_ok=True)
        # if checkpoint_path.exists():
        #     print(f"Loading checkpoint from {checkpoint_path}...")
        #     checkpoint = torch.load(checkpoint_path)
        #     start_epoch = train_utils.load(
        #         checkpoint,
        #         optimizer,
        #         controller,
        #         adjoint_matcher,
        #         corrector=corrector,
        #         corrector_matcher=corrector_matcher,
        #     )
        #     # Note: Not wrapping this in a DDP since we don't differentiate through SDE simulation.
        # else:
        #     start_epoch = 0

        start_epoch = 0


        if cfg.distributed:
            controller = torch.nn.parallel.DistributedDataParallel(
                controller, device_ids=[cfg.gpu], find_unused_parameters=True
            )
            if corrector is not None:
                corrector = torch.nn.parallel.DistributedDataParallel(
                    corrector, device_ids=[cfg.gpu], find_unused_parameters=True
                )


        print("Instantiating writer...")
        writer = train_utils.Writer(
            name=cfg.exp_name,
            cfg=cfg,
            is_main_process=distributed_mode.is_main_process(),
        )


        print("Instantiating evaluator...")
        eval_dir = Path("eval_figs")
        eval_dir.mkdir(exist_ok=True)
        evaluator = hydra.utils.instantiate(cfg.evaluator, energy=energy)


        print(f"Starting from {start_epoch}/{cfg.num_epochs} epochs...")
        lr_schedule = None
        for epoch in range(start_epoch, cfg.num_epochs):
            stage = train_utils.determine_stage(epoch, cfg)

            matcher, model = {
                "adjoint": (adjoint_matcher, controller),
                "corrector": (corrector_matcher, corrector),
            }.get(stage)

            stats = train_one_epoch(
                matcher, model, source, optimizer, lr_schedule, epoch, device, cfg
            )
            loss = stats["loss"]

            log_dict = {
                f"{stage}_loss": loss,
                f"{stage}_buffer_size": len(matcher.buffer),
            }
            if "ess" in stats:
                log_dict[f"{stage}_ess"] = stats["ess"]
                log_dict[f"{stage}_logZ"] = stats["logZ"]
            if "psi_rel" in stats:
                log_dict[f"{stage}_psi_rel"] = stats["psi_rel"]
                log_dict[f"{stage}_psi_loss"] = stats["psi_loss"]
            if "score_loss" in stats:
                log_dict[f"{stage}_score_loss"] = stats["score_loss"]
            if "score_fp_rel" in stats:
                log_dict[f"{stage}_score_fp_rel"] = stats["score_fp_rel"]

            if "cv_bias" in stats:
                log_dict[f"{stage}_cv_bias"] = stats["cv_bias"]
                log_dict[f"{stage}_cv_var"] = stats["cv_var"]
                log_dict[f"{stage}_raw_var"] = stats["raw_var"]
                log_dict[f"{stage}_target_mag"] = stats["target_mag"]
                log_dict[f"{stage}_relative_bias"] = stats["relative_bias"]
                log_dict[f"{stage}_cv_mse_cost"] = stats["cv_mse_cost"]
                log_dict[f"{stage}_var_gain"] = stats["var_gain"]
                # log_dict[f"{stage}_lambda_scv"] = stats["lambda_scv"]
                log_dict[f"{stage}_phi_r2"] = stats["phi_r2"]
            writer.log(log_dict, step=epoch)

            line = green(f"loss={loss:.4f}")
            if "ess" in stats:
                line = line + "  " + magenta(f"ess={stats['ess']:.3f}")
            if "psi_rel" in stats:
                line = line + "  " + magenta(f"psi_rel={stats['psi_rel']:.3f}")
            if "score_fp_rel" in stats:
                line = line + "  " + magenta(f"score_fp={stats['score_fp_rel']:.3f}")
            print("[{0} | {1}] {2}".format(
                cyan(  f"{stage:<7}"),
                yellow(f"ep={epoch:04}"),
                line,
            ))

            # Eval epoch according to the frequency
            # otherwise eval at the end of adjoint matching
            if "eval_freq" in cfg:
                eval_this_epoch = epoch > 0 and epoch % cfg.eval_freq == 0
            else:
                eval_this_epoch = train_utils.is_last_am_epoch(epoch, cfg)
            plot_this_epoch = "plot_freq" in cfg and epoch > 0 and epoch % cfg.plot_freq == 0

            if distributed_mode.is_main_process() and eval_this_epoch:
                if stage == "adjoint" or "plot_freq" in cfg:
                    n_gen_samples = 0
                    x1_list = []
                    while n_gen_samples < cfg.num_eval_samples:
                        B = min(cfg.eval_batch_size, cfg.num_eval_samples - n_gen_samples)
                        x0 = source.sample([B,]).to(device)
                        timesteps = train_utils.get_timesteps(**cfg.timesteps).to(x0)

                        x0, x1 = sdeint(
                            sde,
                            x0,
                            timesteps,
                            only_boundary=True,
                        )
                        x1_list.append(x1)
                        n_gen_samples += x1.shape[0]
                        print("Generated {} samples (total: {}/{})".format(
                            x1.shape[0],
                            n_gen_samples,
                            cfg.num_eval_samples,
                        ))

                    samples = torch.cat(x1_list, dim=0)
                    if "plot_freq" in cfg:
                        eval_dict = evaluator(samples, plot=plot_this_epoch)
                    else:
                        eval_dict = evaluator(samples)
                    print(f"Evaluated @{epoch=}!")

                    if "hist_img" in eval_dict:
                        eval_dict["hist_img"].save(eval_dir / "gen.png")

                    writer.log(eval_dict, step=epoch)

                print("Saving checkpoint ... ")
                train_utils.save(
                    epoch,
                    cfg,
                    optimizer,
                    controller,
                    adjoint_matcher,
                    corrector=corrector,
                    corrector_matcher=corrector_matcher,
                )

    except Exception as e:
        # This way we have the full traceback in the log.  otherwise Hydra
        # will handle the exception and store only the error in a pkl file
        print(traceback.format_exc(), file=sys.stderr)
        raise e


if __name__ == "__main__":
    main()
