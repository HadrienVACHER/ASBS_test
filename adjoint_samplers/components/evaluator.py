# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from typing import Dict
from pathlib import Path
import torch
import ot as pot
import numpy as np

from adjoint_samplers.energies import DoubleWellEnergy, LennardJonesEnergy
from adjoint_samplers.utils.graph_utils import remove_mean
from adjoint_samplers.utils.eval_utils import (
    dist_point_clouds,
    interatomic_dist,
    get_fig_axes,
    fig2img,
)


class DemoEvaluator:
    def __init__(self, energy) -> None:
        from adjoint_samplers.energies.dist_energy import DistEnergy
        assert isinstance(energy, DistEnergy)
        self.dist = energy.dist

        # Plot target samples
        self.fig, axes = get_fig_axes(ncol=6, nrow=10, ax_length_in=3)
        self.axes = axes.reshape(-1)
        self.subplot_idx = 0

    @property
    def ax(self):
        return self.axes[self.subplot_idx]

    def plot_hist(self, x, title=None) -> None:
        B, D = x.shape
        assert D == 1

        if title is None:
            title = f"Eval #{self.subplot_idx}"

        x = (x.reshape(-1)).detach().cpu()
        self.ax.hist(x, bins=50, density=True)
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(0, 1)
        self.ax.grid(True)
        self.ax.set_title(title)

    def __call__(self, samples: torch.Tensor) -> Dict:
        # Plot target samples for reference
        if self.subplot_idx == 0:
            target_samples = self.dist.sample([10000,]).cpu()
            self.plot_hist(target_samples, title="Target")
            self.subplot_idx += 1

        # Plot model samples
        self.plot_hist(samples.cpu())
        self.subplot_idx += 1

        # Return figure
        self.fig.canvas.draw()
        PIL_img = fig2img(self.fig)
        return {"hist_img": PIL_img}


class SyntheticEenergyEvaluator:
    def __init__(self, ref_samples_path, energy) -> None:

        assert isinstance(energy, (DoubleWellEnergy, LennardJonesEnergy))
        self.energy = energy
        self.n_particles = energy.n_particles
        self.n_spatial_dim = energy.n_spatial_dim

        # Extract reference samples
        root = Path(os.path.abspath(__file__)).parent.parent.parent
        ref_samples_np = np.load(root / Path(ref_samples_path), allow_pickle=True)
        self.ref_samples = remove_mean(
            torch.tensor(ref_samples_np),
            energy.n_particles,
            energy.n_spatial_dim,
        )

    def __call__(self, samples: torch.Tensor) -> Dict:

        B, D = samples.shape
        assert D == self.energy.dim

        # Sample reference samples
        idxs = torch.randperm(len(self.ref_samples))[:B]
        ref_samples = self.ref_samples[idxs].to(samples.device)


        print("Computing energy W2...")
        gen_energy = self.energy.eval(samples)
        ref_energy = self.energy.eval(ref_samples)
        energy_w2 = pot.emd2_1d(ref_energy.cpu().numpy(), gen_energy.cpu().numpy())**0.5


        print("Computing interatomic W2...")
        gen_dist = interatomic_dist(samples, self.n_particles, self.n_spatial_dim)
        ref_dist = interatomic_dist(ref_samples, self.n_particles, self.n_spatial_dim)
        dist_w2 = pot.emd2_1d(
            gen_dist.cpu().numpy().reshape(-1),
            ref_dist.cpu().numpy().reshape(-1),
        )


        print("Computing particles W2...")
        M = dist_point_clouds(
            samples.reshape(-1, self.n_particles, self.n_spatial_dim).cpu(),
            ref_samples.reshape(-1, self.n_particles, self.n_spatial_dim).cpu(),
        )
        a = torch.ones(M.shape[0]) / M.shape[0]
        b = torch.ones(M.shape[0]) / M.shape[0]
        eq_w2 = pot.emd2(M=M**2, a=a, b=b)**0.5
        eq_w2 = eq_w2.item()


        return {
            "energy_w2": energy_w2,
            "eq_w2": eq_w2,
            "dist_w2": dist_w2,
        }
