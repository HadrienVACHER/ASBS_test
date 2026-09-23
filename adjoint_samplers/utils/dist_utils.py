# Copyright (c) Meta Platforms, Inc. and affiliates.

import math
import torch
from torch import distributions


########################################
######### Target Distributions #########
########################################

class GMM1D(distributions.Distribution):
    """ A simple tri-modal Gaussian mixtures in 1D for demo purposes
    """
    def __init__(self, device="cpu") -> None:
        super().__init__()

        self.dim = 1
        self.name = "gmm1d"
        self._initialize_distr(device)

    def _initialize_distr(self, device) -> None:
        loc = torch.tensor([-6, -2, 5], device=device, dtype=torch.float).reshape(3, 1)
        scale = torch.tensor([1, 1, 1], device=device, dtype=torch.float).reshape(3, 1)
        weights = torch.tensor([.3, .1, .6], device=device, dtype=torch.float).reshape(3)

        modes = distributions.Independent(
            distributions.Normal(loc, scale), 1
        )
        mix = distributions.Categorical(weights)
        self.distr = distributions.MixtureSameFamily(mix, modes)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_prob = self.distr.log_prob(x).unsqueeze(-1)
        assert log_prob.shape == (*x.shape[:-1], 1)
        return log_prob

    def sample(self, shape: tuple) -> torch.Tensor:
        return self.distr.sample(torch.Size(shape))

    def to(self, device) -> distributions.Distribution:
        self._initialize_distr(device)
        return self

########################################
######### Source Distributions #########
########################################

class Gauss(distributions.Distribution):
    def __init__(
        self,
        dim,
        loc: float = 0.0,
        scale: float = 1.0,
        device: str ="cpu"
    ) -> None:
        super().__init__()

        self.dim = dim
        self.loc = torch.tensor(loc, device=device, dtype=torch.float)
        self.scale = torch.tensor(scale, device=device, dtype=torch.float)
        self.name = "gauss"

    def sample(self, shape: tuple) -> torch.Tensor:
        z = torch.randn(*shape, self.dim, device=self.loc.device)
        return z * self.scale + self.loc


class Delta(distributions.Distribution):
    def __init__(self, dim, loc: float = 0.0, device="cpu") -> None:
        super().__init__()

        self.name = "delta"
        self.dim = dim
        self.loc = torch.tensor(loc, device=device, dtype=torch.float)
        self.scale = torch.tensor(0.0, device=device, dtype=torch.float)

    def sample(self, shape: tuple) -> torch.Tensor:
        return self.loc.repeat(*shape, self.dim)


class CenteredParticlesGauss(distributions.Distribution):
    """ Sample particles with zero center of mass
    """
    def __init__(
        self,
        n_particles,
        spatial_dim,
        scale: float = 1.0,
        device="cpu",
    ):
        super().__init__()
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.dim = n_particles * spatial_dim
        # Centered distribution always has zero mean
        self.loc = torch.tensor(0.0, device=device, dtype=torch.float)
        self.scale = torch.tensor(scale, device=device, dtype=torch.float)
        self.device = device
        self.name = "meanfree"

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        if shape is None:
            shape = tuple()
        samples = torch.randn(*shape, self.dim, device=self.device) * self.scale
        samples = samples.reshape(-1, self.n_particles, self.spatial_dim)
        samples = samples - samples.mean(-2, keepdims=True)
        return samples.reshape(*shape, self.n_particles * self.spatial_dim)


class CenteredParticlesHarmonic(distributions.Distribution):
    """ Sample particles with zero center of mass from
        non-isotropic Gaussian based on a harmonic prior
        https://arxiv.org/pdf/2304.02198
    """
    def __init__(
        self,
        n_particles,
        spatial_dim,
        scale: float = 1.0,
        device="cpu",
    ):
        super().__init__()
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.dim = n_particles * spatial_dim
        # Centered distribution always has zero mean
        self.loc = torch.tensor(0.0, device=device, dtype=torch.float)
        self.scale = torch.tensor(scale, device=device, dtype=torch.float)
        self.device = device
        self.name = "harmonic"

        cov = self._compute_cov(n_particles, spatial_dim)
        self.rank, self.A = self._decompose_svd(cov)

    def _compute_cov(self, n_particles, spatial_dim):
        """ e.g., n_particles = 2, spatial_dim = 3 would generate

            R = tensor([[  1,   0,   0, -0.5,   0,   0 ],
                        [  0,   1,   0,   0, -0.5,   0 ],
                        [  0,   0,   1,   0,    0, -0.5],
                        [-0.5,  0,   0,   1,    0,   0 ],
                        [  0, -0.5,   0,  0,    1,   0 ],
                        [  0,   0, -0.5,  0,    0,   1 ]])

            Denote x = [a1, a2, a3, b1, b2, b3]. This yields

            0.5 * x^T R x = a**2 + b**2 - ab = (a - b)**2
        """
        # TODO(ghliu) assume all particles are connected; otherwise changes A
        A = - 0.5 * torch.ones(n_particles, n_particles)
        A[torch.arange(n_particles), torch.arange(n_particles)] = 1.
        B = torch.eye(spatial_dim)
        return torch.kron(A, B).inverse()

    def _decompose_svd(self, cov):
        """ return the rank of cov and A where `cov = UΣV = AA^T`
        """
        U, S, Vt = torch.svd(cov)

        rank = (S > 1e-8).sum()
        A = U[:, :rank] @ torch.diag(S[:rank]).sqrt()
        return rank, A

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        """ generate samples by
            1. z ~ N(0,I) in the subspace with dim=rank
            2. x = Az, hence x ~ N(0, AA^T)
            3. make x zero COM
        """
        if shape is None:
            shape = tuple()

        B = math.prod(shape) # batch
        z = torch.randn(B, self.rank, device=self.device)
        samples = z @ self.A.to(z).T # (B, R) x (R, D) = (B, D)
        assert samples.shape == (B, self.dim)

        samples = samples * self.scale # note: scale = sqrt(alpha)

        samples = samples.reshape(*shape, self.n_particles, self.spatial_dim)
        samples = samples - samples.mean(-2, keepdims=True)
        return samples.reshape(*shape, self.dim)
