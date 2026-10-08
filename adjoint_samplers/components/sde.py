# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations
import numpy as np
from typing import List

import torch

from adjoint_samplers.utils.dist_utils import CenteredParticlesGauss
import adjoint_samplers.utils.graph_utils as graph_utils


class BaseSDE(torch.nn.Module):
    """ dX_t = f(t, X_t) dt + g(t) dW_t
    """
    def __init__(self):
        super().__init__()

    def register(self, name: str, val: float):
        self.register_buffer(
            name,
            torch.tensor(val, dtype=torch.float),
            persistent=False,
        )

    @property
    def has_drift(self) -> bool:
        return True

    def randn_like(self, x: torch.Tensor):
        return torch.randn_like(x)

    def propagate(self, x, dx):
        return x + dx

    def _pt_gauss_param(
        self,
        t: torch.Tensor,
        mu0: torch.Tensor,
        var0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ time-marginal p_t(x) as a Gaussian given p0 = N(mu0, var0)
        """
        raise NotImplementedError

    def pt_gauss_param(self, t, mu0, var0):
        # dump func for graph assertion
        return self._pt_gauss_param(t, mu0, var0)

    def cond_score(self, x0: torch.Tensor, t: torch.Tensor, xt: torch.Tensor):
        """ p_{t|0}(x|x0) = N(x; μ, Σ) as a Gaussian
            ∇log p = (μ - x) / Σ
        """
        loc, var = self._pt_gauss_param(t, x0)
        return (loc - xt) / var

    # f
    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # g
    def diff(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BrownianMotionSDE(BaseSDE):
    """ dX_t = σ dW_t
        dμ_t = 0 dt  , μ(0) = μ_0 ---> μ(t) = μ_0
        dΣ_t = σ^2 dt, Σ(0) = Σ_0 ---> Σ(t) = Σ_0 + σ^2 t
    """
    def __init__(self, sigma: float = 2.0):
        super().__init__()
        assert sigma > 0
        self.register("sigma", sigma)
        self.sigma: torch.Tensor

    @property
    def has_drift(self) -> bool:
        return False

    def drift(self, t, x):
        return torch.zeros_like(x)

    def diff(self, t):
        return torch.full_like(t, self.sigma)

    def _pt_gauss_param(
        self,
        t: torch.Tensor,
        mu0: torch.Tensor,
        var0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        var = self.sigma**2 * t
        if var0 is not None:
            var = var + var0
        return mu0, var

    def sample_posterior(self, t, x0, x1):
        """ t: (B, 1)  x0: (B, D)  x1: (B, D)
            return: xt: (B, D)
        """
        (B, D), T = x0.shape, t.shape[0]
        assert x1.shape == (B, D) and t.shape == (T, 1) and B == T

        mean = (1 - t) * x0 + t * x1
        var = (1 - t) * t * self.sigma**2
        var[var < 0] = 0
        noise = var.sqrt() * self.randn_like(mean)
        assert mean.shape == noise.shape == (B, D)

        return mean + noise


class VESDE(BaseSDE):
    def __init__(self, sigma_min, sigma_max):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_diff = sigma_max / sigma_min
        self.total_var = sigma_max**2 - sigma_min**2

    @property
    def has_drift(self) -> bool:
        return False

    def _diffsquare_integral(self, t):
        '''
        integral g^2(t) from 0 to t
        Note that integral g(t) from 0 to 1 is sigma_max
        '''
        return (self.sigma_max**2) * (1 - (self.sigma_diff) ** (-2 * t))

    def drift(self, t, x):
        return torch.zeros_like(x)

    def diff(self, t):
        return self.sigma_min \
            * (self.sigma_diff ** (1-t)) \
            * ((2 * np.log(self.sigma_diff)) ** 0.5)

    def _pt_gauss_param(self, t, mu0, var0 = None):
        var = self._diffsquare_integral(t)
        if var0 is not None:
            var = var + var0
        return mu0, var

    def sample_posterior(self, t, x0, x1, z=None):
        """ t: (B, 1)  x0: (B, D)  x1: (B, D)
            return: xt: (B, D)
        """
        (B, D) = x0.shape
        assert x1.shape == (B, D) and t.shape == (B, 1)

        t_reparam = self._diffsquare_integral(t) / self.total_var

        if z is None:
            z = self.randn_like(x0)
        assert z.shape == (B, D)

        mean = (1 - t_reparam) * x0 + t_reparam * x1
        coeff = self.total_var * t_reparam * (1 - t_reparam)
        coeff[coeff < 0] = 0 # NOTE(ghliu) avoid numerical error close to boundary
        noise = torch.sqrt(coeff) * z
        assert mean.shape == noise.shape == (B, D)

        return mean + noise


class VPSDE(BaseSDE):
    """
        dX_t = - β_t / 2 * X_t dt + σ^2 sqrt(β_t) dW_t

        Note: if X_0 ~ N(0, σ^2), then X_t ~ N(0, σ^2) for all t ∈ [0,1]

    """
    def __init__(
        self,
        beta0: float = 20.0,
        beta1: float = 0.1,
        sigma: float = 1.0,
    ):
        super().__init__()

        self.register_buffer(
            "beta1",
            torch.tensor(beta1, dtype=torch.float),
            persistent=False,
        )
        self.beta1: torch.Tensor
        self.register_buffer(
            "beta0",
            torch.tensor(beta0, dtype=torch.float),
            persistent=False,
        )
        self.beta0: torch.Tensor
        self.register_buffer(
            "sigma",
            torch.tensor(sigma, dtype=torch.float),
            persistent=False,
        )

    def coeff2(self, t):
        bt, b1 = self._beta(t), self.beta1
        return -0.25 * (1-t) * (bt + b1) # = $ -\\frac{1}{2} \int^1_t \\beta(s) ds$

    def _beta(self, t: torch.Tensor) -> torch.Tensor:
        return torch.lerp(self.beta0, self.beta1, t)

    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return - 0.5 * self._beta(t) * x

    def diff(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma * torch.sqrt(self._beta(t))

    def score(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return - 1. / (self.sigma ** 2) * x

    def coeff2(self, t):
        bt, b1 = self._beta(t), self.beta1
        return -0.25 * (1-t) * (bt + b1) # = $ -\\frac{1}{2} \int^1_t \\beta(s) ds$

    def _pt_gauss_param(
        self,
        t: torch.Tensor,
        mu0: torch.Tensor,
        var0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        bt, b0 = self._beta(t), self.beta0
        coeff = -0.25 * t * (bt + b0) # = $ -\\frac{1}{2} \int^t_0 \\beta(s) ds$

        mu = mu0 * torch.exp(coeff)
        var = self.sigma**2 * (1 - torch.exp(2 * coeff))
        var[var<0] = 0

        if var0 is not None:
            var = var + torch.exp(2 * coeff) * var0
        return mu, var

    def sample_posterior(self, t, x0, x1):
        bt, b0, b1 = self._beta(t), self.beta0, self.beta1
        coeff1 = -0.25 * t * (bt + b0) # = $ -\\frac{1}{2} \int^t_0 \\beta(s) ds$
        coeff2 = -0.25 * (1-t) * (bt + b1) # = $ -\\frac{1}{2} \int^1_t \\beta(s) ds$
        coeff3 = -0.25 * (b1 + b0)

        mu = torch.exp(coeff1) * (1 - torch.exp(2*coeff2)) / (1 - torch.exp(2*coeff3)) * x0 \
           + torch.exp(coeff2) * (1 - torch.exp(2*coeff1)) / (1 - torch.exp(2*coeff3)) * x1

        var = (1 - torch.exp(2*coeff1)) * (1 - torch.exp(2*coeff2)) / (1 - torch.exp(2*coeff3) + 1e-8)
        var[var < 0] = 0
        std = self.sigma * torch.sqrt(var)

        z = torch.randn_like(mu)
        xt = mu + std * z
        return xt


## Graph ##
class Graph:
    def __init__(self, n_particles: int = 3, spatial_dim: int | None = None):
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.projcted_gauss = CenteredParticlesGauss(
            n_particles,
            spatial_dim,
            scale=1,
        )

    def is_freemean(self, x: torch.Tensor):
        return graph_utils.is_freemean(
            x,
            self.n_particles,
            self.spatial_dim
        )

    def randn_like(self, x: torch.Tensor):
        B, D = x.shape
        assert D == self.spatial_dim * self.n_particles

        noise = self.projcted_gauss.sample((B,)).to(x)
        assert noise.shape == x.shape
        return noise

    def propagate(self, x, dx):
        return graph_utils.remove_mean(x + dx, self.n_particles, self.spatial_dim)

    def pt_gauss_param(self, *args):
        raise NotImplementedError("This should never be called!")


# note: Graph goes before BaseSDE to override rand_like!
class GraphVESDE(Graph, VESDE):
    """ dX_t = σ A dW_t, where A is the matrix such that y = Ax has zero COM
        note: _pt_gauss_param output intermideate results
    """
    def __init__(self, n_particles: int = 3, spatial_dim: int | None = None, *args, **kwargs):
        Graph.__init__(self, n_particles, spatial_dim)
        VESDE.__init__(self, *args, **kwargs)


class GraphVPSDE(Graph, VPSDE):
    """ dX_t = σ A dW_t, where A is the matrix such that y = Ax has zero COM
        note: _pt_gauss_param output intermideate results
    """
    def __init__(self, n_particles: int = 3, spatial_dim: int | None = None, *args, **kwargs):
        Graph.__init__(self, n_particles, spatial_dim)
        VPSDE.__init__(self, *args, **kwargs)


class ControlledSDE(BaseSDE):
    """ dX_t = ( b(t,x) + g(t)^2 u(t,x) )(t, X_t) dt + g(t) dW_t
    """
    def __init__(
        self,
        ref_sde: BaseSDE,
        u: torch.nn.Module,
    ):
        super().__init__()
        self.ref_sde = ref_sde
        self.u = u

    def sample_base_posterior(self, t, x0, x1):
        # p^{base}_t(x | x0, x1)
        return self.ref_sde.sample_posterior(t, x0, x1)

    def randn_like(self, x):
        return self.ref_sde.randn_like(x)

    def propagate(self, x, dx):
        return self.ref_sde.propagate(x, dx)

    def diff(self, t):
        return self.ref_sde.diff(t)

    def drift(self, t, x):
        return self.ref_sde.drift(t, x) + (self.diff(t)**2) * self.u(t, x)


@torch.no_grad()
def sdeint(
    sde: BaseSDE,
    state0: torch.Tensor,
    timesteps: torch.Tensor,
    only_boundary: bool = False,
) -> List[torch.Tensor]:

    T = len(timesteps)
    assert len(timesteps) > 1

    sde.train(False)

    state = state0.clone()

    states = [state0,]
    for i in range(T - 1):
        t = timesteps[i]
        dt = timesteps[i + 1] - t

        drift = sde.drift(t, state) * dt
        diffusion = sde.diff(t) * dt.sqrt() * sde.randn_like(state)

        d_state = drift + diffusion
        state = sde.propagate(state, d_state)

        states.append(state)

    if only_boundary:
        return states[0], states[-1]
    return states

@torch.no_grad()
def sdeint_logw(sde, state0, timesteps):
    """Same integrator as sdeint(..., only_boundary=True), plus log(dP_ref / dP_u)."""
    sde.train(False)
    state = state0.clone()
    logw = torch.zeros(state.shape[0], device=state.device)
    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        dt = timesteps[i + 1] - t
        g = sde.diff(t)
        u = sde.u(t, state)
        xi = sde.randn_like(state)
        d_state = (
            sde.ref_sde.drift(t, state) + (g ** 2) * u
        ) * dt + g * dt.sqrt() * xi
        logw = logw - (u * xi).sum(-1) * g * dt.sqrt() - 0.5 * u.pow(2).sum(-1) * (g ** 2) * dt
        state = sde.propagate(state, d_state)
    return state0, state, logw
