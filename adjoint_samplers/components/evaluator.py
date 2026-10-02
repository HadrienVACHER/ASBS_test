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
        self.fig, axes = get_fig_axes(ncol=5, nrow=10, ax_length_in=3)
        self.axes = axes.reshape(-1)
        self.subplot_idx = 0

    @property
    def ax(self):
        return self.axes[self.subplot_idx]

    def plot_hist(self, x, title=None) -> None:
        B, D = x.shape
        if title is None:
            title = f"Eval #{self.subplot_idx}"
        x = x.detach().cpu()
        if D == 1:
            self.ax.hist(x.reshape(-1), bins=50, density=True)
            self.ax.set_xlim(-10, 10)
            self.ax.set_ylim(0, 0.4)
        else:
            self.ax.scatter(x[:, 0], x[:, 1], s=2, alpha=0.3)  # first two coords
            self.ax.set_xlim(-10, 10)
            self.ax.set_ylim(-10, 10)
        self.ax.grid(True)
        self.ax.set_title(title)

    def __call__(self, samples: torch.Tensor, plot: bool = True) -> Dict:
        result = {}
        if plot:
            if self.subplot_idx == 0:
                target_samples = self.dist.sample([10000,]).cpu()
                self.plot_hist(target_samples, title="Target")
                self.subplot_idx += 1

            self.plot_hist(samples.cpu())
            self.subplot_idx += 1
            self.fig.canvas.draw()
            result["hist_img"] = fig2img(self.fig)

        target = self.dist.sample([samples.shape[0]]).detach().cpu().numpy()
        generated = samples.detach().cpu().numpy()
        if generated.shape[1] == 1:
            result["w2"] = pot.emd2_1d(target.reshape(-1), generated.reshape(-1)) ** 0.5
        else:
            n = generated.shape[0]
            a = np.full(n, 1.0 / n)
            M = pot.dist(target, generated)  # squared Euclidean
            result["w2"] = pot.emd2(a, a, M) ** 0.5
        return result


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


class GMMEvaluator:
    def __init__(self, energy, n_proj: int = 100, n_bins: int = 50, seed: int = 0) -> None:
        from adjoint_samplers.energies.dist_energy import DistEnergy
        assert isinstance(energy, DistEnergy)
        self.dist = energy.dist
        self.n_bins = n_bins
        g = torch.Generator().manual_seed(seed)
        theta = torch.randn(n_proj, self.dist.dim, generator=g)
        self.theta = theta / theta.norm(dim=1, keepdim=True)

    def __call__(self, samples: torch.Tensor) -> Dict:
        x = samples.detach()
        y = self.dist.sample([x.shape[0]]).to(x.device)

        # mode TVD (0.5 factor so that the value lies in [0, 1])
        counts = torch.bincount(self.dist.assign_mode(x), minlength=self.dist.n_modes).float()
        pi_hat = counts / counts.sum()
        mode_tvd = 0.5 * (self.dist.weights - pi_hat).abs().sum().item()

        # sliced TVD
        theta = self.theta.to(x.device)
        px = (x @ theta.T).cpu().numpy()
        py = (y @ theta.T).cpu().numpy()
        tvds = []
        for p in range(px.shape[1]):
            lo = min(px[:, p].min(), py[:, p].min())
            hi = max(px[:, p].max(), py[:, p].max())
            hx, _ = np.histogram(px[:, p], bins=self.n_bins, range=(lo, hi))
            hy, _ = np.histogram(py[:, p], bins=self.n_bins, range=(lo, hi))
            tvds.append(0.5 * np.abs(hx / hx.sum() - hy / hy.sum()).sum())

        # W2 with POT
        xn = x.cpu().numpy().astype(np.float64)
        yn = y.cpu().numpy().astype(np.float64)
        n = xn.shape[0]
        a = np.full(n, 1.0 / n)
        w2 = pot.emd2(a, a, pot.dist(yn, xn), numItermax=10_000_000) ** 0.5

        return {"mode_tvd": mode_tvd, "sliced_tvd": float(np.mean(tvds)), "w2": float(w2)}