# Copyright (c) Meta Platforms, Inc. and affiliates.

import math

import torch
from torch.func import grad
from torch.utils.data import DataLoader, WeightedRandomSampler

from adjoint_samplers.components.buffer import BatchBuffer
from adjoint_samplers.components.sde import BaseSDE, sdeint, sdeint_logw
from adjoint_samplers.components.state_cost import GradStateCost, ZeroGradStateCost
from adjoint_samplers.components.term_cost import GradEnergy

import adjoint_samplers.utils.graph_utils as graph_utils

class Matcher:
    def __init__(
        self,
        sde: BaseSDE | None = None,
        buffer: BatchBuffer | None = None,
        resample_size: int | None = None,
        duplicates: int | None = None,
        loss_scale: float = 1,
        **kwargs,
    ):
        self.sde = sde
        self.buffer = buffer
        self.resample_size = resample_size
        self.duplicates = duplicates
        self.loss_scale = loss_scale

    def build_dataloader(self, batch_size, collate_fn=None) -> DataLoader:
        dataset = self.buffer.build_dataset(self.duplicates)
        if "logw" in dataset.total_data:
            logw = dataset.total_data["logw"].double()
            w = (logw - logw.max()).exp()
            w = w / w.sum()
            cap = 10.0 / w.numel()
            w = torch.clamp(w, max=cap)
            # The residual is negative when the cap does not bind and the sum
            # rounds to slightly more than 1. An underflowed weight then goes
            # below 0 and multinomial rejects the distribution.
            w = (w + (1.0 - w.sum()) / w.numel()).clamp_min(0)
            w = w / w.sum()
            weights = w.repeat(dataset.duplicates).float()
            sampler = WeightedRandomSampler(
                weights, num_samples=len(dataset), replacement=True,
            )
            return DataLoader(
                dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn,
            )
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
        )

    def populate_buffer(self):
        raise NotImplementedError()

    def prepare_target(self):
        raise NotImplementedError()


class AdjointMatcher(Matcher):
    def __init__(
        self,
        grad_term_cost: GradEnergy | None = None,
        grad_state_cost: GradStateCost | None = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.grad_term_cost = grad_term_cost
        self.grad_state_cost = grad_state_cost

    @torch.no_grad()
    def _backward_simulate(self, adjoint1, timesteps, xs):
        (T, B, D) = xs.shape
        assert len(timesteps) == T and T > 1
        assert adjoint1.shape == (B, D)

        adjoint = adjoint1.clone()
        adjoints = [adjoint]
        for j in range(T - 1, 0, -1):
            dt = timesteps[j] - timesteps[j - 1]
            assert dt > 0

            t = timesteps[j].repeat((B, 1))
            x = xs[j]
            assert t.shape == (B, 1) and x.shape == (B, D)

            # Compute a^T ∇ b(t, x)
            ref_sde = self.sde.ref_sde
            if not ref_sde.has_drift:
                f = torch.zeros_like(x)
            else:
                with torch.enable_grad():
                    f = grad(
                        lambda x: torch.sum(adjoint * ref_sde.drift(t, x))
                    )(x)

            # Compute gradient of state cost
            f = f + self.grad_state_cost(t, x)

            adjoint = adjoint + f * dt
            adjoints.append(adjoint)

        adjoints = torch.stack(adjoints[::-1])
        assert adjoints.shape == (T, B, D)
        return adjoints

    def _compute_adjoint1(self, x1, is_asbs_init_stage):
        if is_asbs_init_stage:
            # IPF init: First Adjoint Matching stage
            # of ASBS uses zero corrector
            adjoint1 = self.grad_term_cost.grad_E(x1)
        else:
            adjoint1 = self.grad_term_cost(x1)
        return adjoint1

    def populate_buffer(
            self,
            x0: torch.Tensor,
            timesteps: torch.Tensor,
            is_asbs_init_stage: bool,
    ):
        (B, D), T = x0.shape, len(timesteps)
        assert x0.device == timesteps.device

        ts = timesteps.unsqueeze(1).repeat((1, B))[..., None]
        assert ts.shape == (T, B, 1)

        xs = sdeint(
            self.sde,
            x0,
            timesteps,
            only_boundary=False,
        )
        xs = torch.stack(xs)
        assert xs.shape == (T, B, D)

        adjoint1 = self._compute_adjoint1(xs[-1], is_asbs_init_stage).clone()
        adjoints = self._backward_simulate(adjoint1, timesteps, xs)
        assert adjoints.shape == (T, B, D)

        # note: use entire traj as one smaple. this improves training.
        ts = ts.transpose(0, 1)
        xs = xs.transpose(0, 1)
        adjoints = adjoints.transpose(0, 1)
        assert ts.shape == (B, T, 1)
        assert adjoints.shape == xs.shape == (B, T, D)

        self.buffer.add({
            "t": ts.reshape(B, T).detach().cpu(),
            "xt": xs.reshape(B, T * D).detach().cpu(),
            "adjointt": adjoints.reshape(B, T * D).detach().cpu(),
        })

    def prepare_target(self, data, device):
        t = data["t"].to(device)
        xt = data["xt"].to(device)
        adjointt = data["adjointt"].to(device)

        (B, T), D = t.shape, xt.shape[1] // t.shape[1]
        assert xt.shape == adjointt.shape == (B, T * D)

        # randomly select B index
        # TODO(ghliu) not used for AS / ASBS. Refac this logic.
        idx = torch.randint(high=T, size=(B,))
        idx = [i*T+id for i, id in enumerate(idx)]

        t = t.reshape(B * T, 1)[idx]
        xt = xt.reshape(B * T, D)[idx]
        adjointt = adjointt.reshape(B * T, D)[idx]
        return (t, xt), - adjointt


class AdjointVEMatcher(AdjointMatcher):
    """ Efficient computation of AM when the base SDE has no drift (e.g., VE)
        and the SOC problem has no state cost.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._check_soc_problem()

    def _check_soc_problem(self):
        assert not self.sde.ref_sde.has_drift
        assert isinstance(self.grad_state_cost, ZeroGradStateCost)

    def _check_buffer_sample_shape(self, x0, x1, adjoint1):
        (B, D) = x0.shape
        assert x1.shape == adjoint1.shape == (B, D)

    def _buffer_cat(self, key):
        parts = self.buffer.batches[key]
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=0)

    def _set_terminal_var(self, timesteps):
        """Variance of the reference Euler endpoint, matching sdeint_logw."""
        ref = self.sde.ref_sde
        var = timesteps.new_zeros(())
        for i in range(timesteps.shape[0] - 1):
            dt = timesteps[i + 1] - timesteps[i]
            g = ref.diff(timesteps[i])
            var = var + g.pow(2) * dt
        self._terminal_var = float(var.detach().cpu())

    def _phi0_rollouts(self, x0, timesteps, reference_only):
        """K controlled paths from each x0. log φ₀ = logmeanexp(G − E − ψ)."""
        K = int(getattr(self, "phi0_samples", 8))
        B, D = x0.shape
        x0k = x0.repeat_interleave(K, dim=0)
        if reference_only:
            _, x1k = sdeint(self.sde.ref_sde, x0k, timesteps, only_boundary=True)
            loggk = torch.zeros(B * K, device=x0.device, dtype=x0.dtype)
        else:
            _, x1k, loggk = sdeint_logw(self.sde, x0k, timesteps)
        return x1k.reshape(B, K, D), loggk.reshape(B, K)

    def populate_buffer(self, x0, timesteps, is_asbs_init_stage):
        # ASBS: one controlled path is kept for training. K paths from the same
        # x0 estimate φ₀. write_sb_logw applies the current ψ to those paths.
        if getattr(self, "potential", None) is not None:
            if self.buffer.batches and "phi0_logg" not in self.buffer.batches:
                self.buffer.batches = {}
            self._set_terminal_var(timesteps)
            x1k, loggk = self._phi0_rollouts(x0, timesteps, reference_only=False)
            x1 = x1k[:, 0]
            log_g = loggk[:, 0]
            adjoint1 = self._compute_adjoint1(x1, is_asbs_init_stage).clone()
            self._check_buffer_sample_shape(x0, x1, adjoint1)
            self._fresh_count = getattr(self, "_fresh_count", 0) + x0.shape[0]
            self.buffer.add({
                "x0": x0.to("cpu"),
                "x1": x1.detach().cpu(),
                "adjoint1": adjoint1.to("cpu"),
                "log_girsanov": log_g.detach().cpu(),
                "phi0_x1": x1k.detach().cpu(),
                "phi0_logg": loggk.detach().cpu(),
            })
            return

        if not getattr(self, "use_is", True):
            x0, x1 = sdeint(self.sde, x0, timesteps, only_boundary=True)
            adjoint1 = self._compute_adjoint1(x1, is_asbs_init_stage).clone()
            self._check_buffer_sample_shape(x0, x1, adjoint1)
            self.buffer.add({
                "x0": x0.to("cpu"),
                "x1": x1.to("cpu"),
                "adjoint1": adjoint1.to("cpu"),
            })
            return

        (x0, x1, logw) = sdeint_logw(self.sde, x0, timesteps)
        adjoint1 = self._compute_adjoint1(x1, is_asbs_init_stage).clone()

        ref = self.sde.ref_sde
        sigma2 = ref.total_var if hasattr(ref, "total_var") else ref.sigma ** 2
        d_eff = x1.shape[-1] - ref.spatial_dim if hasattr(ref, "n_particles") else x1.shape[-1]
        log_const = torch.log(x1.new_tensor(2 * torch.pi * float(sigma2)))
        log_pbase = -0.5 * x1.pow(2).sum(-1) / sigma2 - 0.5 * d_eff * log_const
        energy = self.grad_term_cost.energy.eval(x1)
        logw = logw - energy.view(-1) - log_pbase
        if not hasattr(self, "_epoch_logw"):
            self._epoch_logw = []
        self._epoch_logw.append(logw.detach().cpu())

        self._check_buffer_sample_shape(x0, x1, adjoint1)
        self.buffer.add({
            "x0": x0.to("cpu"),
            "x1": x1.to("cpu"),
            "adjoint1": adjoint1.to("cpu"),
            "logw": logw.detach().cpu(),
        })

    def sample_t(self, x):
        (B, D) = x.shape
        return torch.rand(B, 1)

    def _check_target_shape(self, t, xt, adjoint):
        (B, D) = xt.shape
        assert t.shape == (B, 1) and adjoint.shape == (B, D)


    def prepare_target(self, data, device):
        x0 = data["x0"].to(device)
        x1 = data["x1"].to(device)
        adjoint1 = data["adjoint1"].to(device)

        t = self.sample_t(x0).to(device)
        adjoint = adjoint1
        use_cv = (
            getattr(self, "cond_score", None) is not None
            and getattr(self, "neural_scv", None) is not None
        )
        if use_cv:
            t = t.clamp(1e-3, 1.0 - 1e-3)
        xt = self.sde.sample_base_posterior(t, x0, x1)

        if not use_cv:
            self._check_target_shape(t, xt, adjoint)
            return (t, xt), -adjoint
        ref = self.sde.ref_sde
        x1_req = x1.detach().requires_grad_(True)
        xt_det = xt.detach()
        t_det = t.detach()
        x0_det = x0.detach()
        if hasattr(ref, "total_var"):
            lam = ref._diffsquare_integral(t_det) / ref.total_var
            sigma2 = ref.total_var
        else:
            lam = t_det
            sigma2 = ref.sigma ** 2
        lam = lam.clamp(1e-3, 1.0 - 1e-3)

        E = self.grad_term_cost.energy.eval(x1_req)
        if E.ndim == 1:
            E = E.unsqueeze(-1)

        h = self.neural_scv(x0_det, x1_req, E, xt_det, t_det)
        F_ = (1.0 - lam).unsqueeze(-1) * h
        d = x1_req.shape[-1]
        if hasattr(ref, "n_particles"):
            B = F_.shape[0]
            F_ = graph_utils.remove_mean(
                F_.reshape(B * d, d), ref.n_particles, ref.spatial_dim,
            ).reshape(B, d, d)

        # ∇_{x1} log q(x1 | x0, xt, t) = ∇_{x1} log q(x1 | x0) + bridge term.
        # The network supplies the buffer conditional score. The bridge term is exact.
        s = self.cond_score(x0_det, x1_req.detach())
        if hasattr(ref, "n_particles"):
            s = graph_utils.remove_mean(s, ref.n_particles, ref.spatial_dim)
        bridge = (xt_det - (1.0 - lam) * x0_det - lam * x1) / (sigma2 * (1.0 - lam))
        score1 = (s + bridge).detach()

        rows = []
        for j in range(d):
            div_j = 0.0
            for i in range(d):
                div_j = div_j + torch.autograd.grad(
                    F_[:, j, i].sum(), x1_req,
                    create_graph=True, retain_graph=True,
                )[0][:, i]
            rows.append(div_j)
        div = torch.stack(rows, dim=-1)

        phi = div + torch.einsum("bji,bi->bj", F_, score1)

        self._check_target_shape(t, xt, adjoint)
        return (t, xt), -adjoint, phi

    # # scv
    # def prepare_target(self, data, device):
    #     x0 = data["x0"].to(device)
    #     x1 = data["x1"].to(device)
    #     adjoint1 = data["adjoint1"].to(device)

    #     t = self.sample_t(x0).to(device).clamp(1e-3, 1.0 - 1e-3)
    #     xt = self.sde.sample_base_posterior(t, x0, x1)
    #     adjoint = adjoint1

    #     ref = self.sde.ref_sde
    #     x1_req = x1.detach().requires_grad_(True)
    #     xt_det = xt.detach()
    #     t_det = t.detach()
    #     if hasattr(ref, "total_var"):
    #         lam = ref._diffsquare_integral(t_det) / ref.total_var
    #         sigma2 = ref.total_var
    #     else:
    #         lam = t_det
    #         sigma2 = ref.sigma ** 2
    #     lam = lam.clamp(1e-3, 1.0 - 1e-3)

    #     E = self.grad_term_cost.energy.eval(x1_req)
    #     if E.ndim == 1:
    #         E = E.unsqueeze(-1)

    #     h = self.neural_scv(x0.detach(), x1_req, E, xt_det, t_det)
    #     F_ = (1.0 - lam).unsqueeze(-1) * h
    #     d = x1_req.shape[-1]
    #     if hasattr(ref, "n_particles"):
    #         B = F_.shape[0]
    #         F_ = graph_utils.remove_mean(
    #             F_.reshape(B * d, d), ref.n_particles, ref.spatial_dim,
    #         ).reshape(B, d, d)

    #     score1 = ((xt_det - x1) / (sigma2 * (1.0 - lam)) - adjoint1).detach()

    #     rows = []
    #     for j in range(d):
    #         div_j = 0.0
    #         for i in range(d):
    #             div_j = div_j + torch.autograd.grad(
    #                 F_[:, j, i].sum(), x1_req,
    #                 create_graph=True, retain_graph=True,
    #             )[0][:, i]
    #         rows.append(div_j)
    #     div = torch.stack(rows, dim=-1)

    #     phi = div + torch.einsum("bji,bi->bj", F_, score1)

    #     self._check_target_shape(t, xt, adjoint)
    #     return (t, xt), -adjoint, phi

    def fit_potential(self, batch_size, device):
        """Regress ∇ψ onto the corrector on the current buffer endpoints."""
        potential = getattr(self, "potential", None)
        if potential is None or "x1" not in self.buffer.batches:
            return None
        x1_all = self._buffer_cat("x1")
        n = x1_all.shape[0]
        if n == 0:
            return None

        corrector = self.grad_term_cost.corrector
        steps = int(getattr(self, "potential_steps", 20))
        bs = min(int(batch_size), n)
        potential.train()
        was_training = corrector.training
        corrector.eval()
        last_loss = 0.0
        last_rel = 0.0
        try:
            for _ in range(steps):
                idx = torch.randint(0, n, (bs,))
                x1 = x1_all[idx].to(device)
                with torch.no_grad():
                    t1 = torch.ones(x1.shape[0], 1, device=device, dtype=x1.dtype)
                    h = corrector(t1, x1)
                _, g = potential.gradient(x1)
                loss = (g - h).pow(2).mean()
                self.potential_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(potential.parameters(), max_norm=100.0)
                self.potential_optimizer.step()
                with torch.no_grad():
                    err = (g.detach() - h).pow(2).sum(-1).mean().sqrt()
                    den = h.pow(2).sum(-1).mean().sqrt().clamp_min(1e-8)
                    last_loss = float(loss.detach())
                    last_rel = float(err / den)
        finally:
            corrector.train(was_training)
        return {"psi_loss": last_loss, "psi_rel": last_rel}

    def fit_cond_score(self, batch_size, device):
        """Sliced score matching for ∇_{x1} log q(x1 | x0) on the training buffer."""
        net = getattr(self, "cond_score", None)
        if net is None or "x1" not in self.buffer.batches:
            return None
        x0_all = self._buffer_cat("x0")
        x1_all = self._buffer_cat("x1")
        n = x1_all.shape[0]
        if n == 0:
            return None
        if "logw" in self.buffer.batches:
            logw = self._buffer_cat("logw").double()
            w = (logw - logw.max()).exp()
            w = (w / w.sum()).float()
        else:
            w = None

        ref = self.sde.ref_sde
        centered = hasattr(ref, "n_particles")
        sigma2 = float(getattr(self, "_terminal_var", getattr(ref, "total_var", 1.0)))
        has_adj = "adjoint1" in self.buffer.batches
        adj_all = self._buffer_cat("adjoint1") if has_adj else None

        steps = int(getattr(self, "cond_score_steps", 20))
        bs = min(int(batch_size), n)
        net.train()
        last_loss = 0.0
        last_gap = None
        for _ in range(steps):
            if w is None:
                idx = torch.randint(0, n, (bs,))
            else:
                idx = torch.multinomial(w, bs, replacement=True)
            x0 = x0_all[idx].to(device)
            x1 = x1_all[idx].to(device).detach().requires_grad_(True)
            s = net(x0, x1)
            if centered:
                s = graph_utils.remove_mean(s, ref.n_particles, ref.spatial_dim)
            div = torch.zeros(bs, device=device, dtype=x1.dtype)
            for i in range(x1.shape[-1]):
                grad_i = torch.autograd.grad(
                    s[:, i].sum(), x1, create_graph=True, retain_graph=True,
                )[0]
                div = div + grad_i[:, i]
            loss = (s.pow(2).sum(-1) + 2.0 * div).mean()
            self.cond_score_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
            self.cond_score_optimizer.step()
            last_loss = float(loss.detach())
            if has_adj:
                with torch.no_grad():
                    s_fp = (x0 - x1.detach()) / sigma2 - adj_all[idx].to(device)
                    if centered:
                        s_fp = graph_utils.remove_mean(s_fp, ref.n_particles, ref.spatial_dim)
                    err = (s.detach() - s_fp).pow(2).sum(-1).mean().sqrt()
                    den = s_fp.pow(2).sum(-1).mean().sqrt().clamp_min(1e-8)
                    last_gap = float(err / den)
        out = {"score_loss": last_loss}
        if last_gap is not None:
            out["score_fp_rel"] = last_gap
        return out

    @torch.no_grad()
    def write_sb_logw(self, device, is_init):
        """log dP*/dP_u = Girsanov − E(X_1) − ψ(X_1) − log φ_0(X_0).

        φ₀(x0) = E_u[ exp(Girsanov − E − ψ) | X_0 = x0 ], estimated by the K
        controlled rollouts stored with the sample. The integrand is constant
        at the optimal control, so the Monte Carlo noise vanishes there.
        During the initial stage the corrector is zero, so ψ = 0.
        """
        if "phi0_logg" not in self.buffer.batches:
            return
        chunks = []
        for x1, log_g, x1k, loggk in zip(
            self.buffer.batches["x1"],
            self.buffer.batches["log_girsanov"],
            self.buffer.batches["phi0_x1"],
            self.buffer.batches["phi0_logg"],
        ):
            chunks.append(self._sb_logw(
                x1.to(device), log_g.to(device), x1k.to(device), loggk.to(device), is_init,
            ).detach().cpu())
        self.buffer.batches["logw"] = chunks
        fresh = torch.cat(chunks, dim=0)
        n_fresh = int(getattr(self, "_fresh_count", fresh.shape[0]))
        self._epoch_logw = [fresh[-n_fresh:]]

    def _sb_logw(self, x1, log_girsanov, phi0_x1, phi0_logg, is_init):
        step = 512
        parts = []
        for i in range(0, x1.shape[0], step):
            parts.append(self._sb_logw_batch(
                x1[i:i + step],
                log_girsanov[i:i + step],
                phi0_x1[i:i + step],
                phi0_logg[i:i + step],
                is_init,
            ))
        return torch.cat(parts, dim=0)

    def _sb_logw_batch(self, x1, log_girsanov, phi0_x1, phi0_logg, is_init):
        B, K, D = phi0_x1.shape
        flat = phi0_x1.reshape(B * K, D)
        energy_k = self.grad_term_cost.energy.eval(flat).reshape(B, K)
        use_psi = (not is_init) and getattr(self, "_psi_ready", False)
        if use_psi:
            was = self.potential.training
            self.potential.eval()
            psi_k = self.potential(flat).reshape(B, K)
            psi_x1 = self.potential(x1).reshape(B)
            self.potential.train(was)
        else:
            psi_k = torch.zeros(B, K, device=x1.device, dtype=x1.dtype)
            psi_x1 = torch.zeros(B, device=x1.device, dtype=x1.dtype)
        log_phi0 = torch.logsumexp(phi0_logg - energy_k - psi_k, dim=1) - math.log(K)
        energy_x1 = self.grad_term_cost.energy.eval(x1).reshape(B)
        return log_girsanov.reshape(B) - energy_x1 - psi_x1 - log_phi0


    # # scv hutchinson
    # def prepare_target(self, data, device):
    #     x0 = data["x0"].to(device)
    #     x1 = data["x1"].to(device)
    #     adjoint1 = data["adjoint1"].to(device)

    #     t = self.sample_t(x0).to(device).clamp(1e-3, 1.0 - 1e-3)
    #     xt = self.sde.sample_base_posterior(t, x0, x1)
    #     adjoint = adjoint1

    #     sigma = self.sde.ref_sde.sigma
    #     x1_req = x1.detach().requires_grad_(True)
    #     xt_det = xt.detach()
    #     t_det = t.detach()

    #     E = self.grad_term_cost.energy.eval(x1_req)
    #     if E.ndim == 1:
    #         E = E.unsqueeze(-1)

    #     h = self.neural_scv(x0.detach(), x1_req, E, xt_det, t_det)   # (B, d, d)
    #     F_ = (1.0 - t_det).unsqueeze(-1) * h                          # (B, d, d)
    #     d = x1_req.shape[-1]

    #     score1 = ((xt_det - x1) / (sigma ** 2 * (1.0 - t_det)) - adjoint1).detach()

    #     eps = torch.empty_like(x1_req).bernoulli_(0.5).mul_(2).sub_(1)
    #     v = torch.einsum("bji,bi->bj", F_, eps)          # (B, d)
    #     rows = []
    #     for j in range(d):
    #         grad_v = torch.autograd.grad(
    #             v[:, j].sum(), x1_req, create_graph=True, retain_graph=True,
    #         )[0]
    #         rows.append((grad_v * eps).sum(dim=-1))
    #     div = torch.stack(rows, dim=-1)
    #     phi = div + torch.einsum("bji,bi->bj", F_, score1)

    #     self._check_target_shape(t, xt, adjoint)
    #     return (t, xt), -adjoint, phi

    # # scv hutchinson v2
    # def prepare_target(self, data, device):
    #     x0 = data["x0"].to(device)
    #     x1 = data["x1"].to(device)
    #     adjoint1 = data["adjoint1"].to(device)

    #     t = self.sample_t(x0).to(device).clamp(1e-3, 1.0 - 1e-3)
    #     xt = self.sde.sample_base_posterior(t, x0, x1)
    #     adjoint = adjoint1

    #     ref = self.sde.ref_sde
    #     xt_det = xt.detach()
    #     t_det = t.detach()
    #     if hasattr(ref, "total_var"):
    #         lam = ref._diffsquare_integral(t_det) / ref.total_var
    #         sigma2 = ref.total_var
    #     else:
    #         lam = t_det
    #         sigma2 = ref.sigma ** 2
    #     lam = lam.clamp(1e-3, 1.0 - 1e-3)

    #     score1 = ((xt_det - x1) / (sigma2 * (1.0 - lam)) - adjoint1).detach()

    #     eps = torch.empty_like(x1).bernoulli_(0.5).mul_(2).sub_(1)
    #     centered = hasattr(ref, "n_particles")
    #     if centered:
    #         eps = graph_utils.remove_mean(eps, ref.n_particles, ref.spatial_dim)

    #     def v_of(x):
    #         E = self.grad_term_cost.energy.eval(x)
    #         if E.ndim == 1:
    #             E = E.unsqueeze(-1)
    #         h = self.neural_scv(x0.detach(), x, E, xt_det, t_det)
    #         F_ = (1.0 - lam).unsqueeze(-1) * h
    #         if centered:
    #             B, d, _ = F_.shape
    #             F_ = graph_utils.remove_mean(
    #                 F_.reshape(B * d, d), ref.n_particles, ref.spatial_dim,
    #             ).reshape(B, d, d)
    #         v = torch.einsum("bji,bi->bj", F_, eps)
    #         return v, F_

    #     (_, F_), (div, _) = torch.autograd.functional.jvp(
    #         v_of, x1.detach(), eps, create_graph=True,
    #     )
    #     phi = div + torch.einsum("bji,bi->bj", F_, score1)

    #     self._check_target_shape(t, xt, adjoint)
    #     return (t, xt), -adjoint, phi



class AdjointVPMatcher(AdjointVEMatcher):
    """ Efficient computation of AM when the base SDE has linear drift (e.g., VP)
        and the SOC problem has no state cost.
    """
    def _check_soc_problem(self):
        assert self.sde.ref_sde.has_drift
        assert isinstance(self.grad_state_cost, ZeroGradStateCost)

    def prepare_target(self, data, device):
        x0 = data["x0"].to(device)
        x1 = data["x1"].to(device)
        adjoint1 = data["adjoint1"].to(device)

        t = self.sample_t(x0).to(device)
        xt = self.sde.sample_base_posterior(t, x0, x1)
        adjoint = adjoint1 # const w.r.t. time in this case
        adjoint = adjoint * torch.exp(self.sde.ref_sde.coeff2(t))

        self._check_target_shape(t, xt, adjoint)
        return (t, xt), - adjoint


class CorrectorMatcher(Matcher):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _check_buffer_sample_shape(self, x0, x1):
        (B, D) = x0.shape
        assert x1.shape == (B, D)

    def _set_terminal_var(self, timesteps):
        return AdjointVEMatcher._set_terminal_var(self, timesteps)

    def _buffer_cat(self, key):
        return AdjointVEMatcher._buffer_cat(self, key)

    def fit_potential(self, batch_size, device):
        return AdjointVEMatcher.fit_potential(self, batch_size, device)

    def write_sb_logw(self, device, is_init):
        return AdjointVEMatcher.write_sb_logw(self, device, is_init)

    def _phi0_rollouts(self, x0, timesteps, reference_only):
        return AdjointVEMatcher._phi0_rollouts(self, x0, timesteps, reference_only)

    def _sb_logw(self, x1, log_girsanov, phi0_x1, phi0_logg, is_init):
        return AdjointVEMatcher._sb_logw(
            self, x1, log_girsanov, phi0_x1, phi0_logg, is_init,
        )

    def _sb_logw_batch(self, x1, log_girsanov, phi0_x1, phi0_logg, is_init):
        return AdjointVEMatcher._sb_logw_batch(
            self, x1, log_girsanov, phi0_x1, phi0_logg, is_init,
        )

    def populate_buffer(
            self,
            x0: torch.Tensor,
            timesteps: torch.Tensor,
            is_asbs_init_stage: bool,
    ):
        # IPF init: First Corrector Matching stage
        # of ASBS uses zero controller (i.e., ref_sde)
        if getattr(self, "potential", None) is not None:
            if self.buffer.batches and "phi0_logg" not in self.buffer.batches:
                self.buffer.batches = {}
            self._set_terminal_var(timesteps)
            x1k, loggk = self._phi0_rollouts(
                x0, timesteps, reference_only=is_asbs_init_stage,
            )
            x1 = x1k[:, 0]
            log_g = loggk[:, 0]
            self._check_buffer_sample_shape(x0, x1)
            self._fresh_count = getattr(self, "_fresh_count", 0) + x0.shape[0]
            self.buffer.add({
                "x0": x0.to("cpu"),
                "x1": x1.detach().cpu(),
                "log_girsanov": log_g.detach().cpu(),
                "phi0_x1": x1k.detach().cpu(),
                "phi0_logg": loggk.detach().cpu(),
            })
            return

        sde = self.sde.ref_sde if is_asbs_init_stage else self.sde

        (x0, x1) = sdeint(
            sde,
            x0,
            timesteps,
            only_boundary=True,
        )

        self._check_buffer_sample_shape(x0, x1)
        self.buffer.add({
            "x0": x0.to("cpu"),
            "x1": x1.to("cpu"),
        })

    def _check_target_shape(self, t1, x1, score):
        (B, D) = x1.shape
        assert t1.shape == (B, 1) and score.shape == (B, D)

    def prepare_target(self, data, device):
        x0 = data["x0"].to(device)
        x1 = data["x1"].to(device)

        t1 = torch.ones(x0.shape[0], 1).to(device)
        score = self.sde.ref_sde.cond_score(x0, t1, x1)

        self._check_target_shape(t1, x1, score)
        return (t1, x1,), score
