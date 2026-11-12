"""Mixed models of the statistical plan (protocol 7.9), written as they are computed.

Every model has one random intercept u_i per unit i (training run in mode B, frozen policy in
mode A); j runs over the episodes of the unit; x_ij is the fixed-effect row (formulas.py):

  LMM       y_ij = x_ij' beta + u_i + e_ij,       u_i ~ N(0, s_u^2),  e_ij ~ N(0, s^2)
  NB2-GLMM  y_ij | u_i ~ NB(mu_ij, phi),          log mu_ij = log E_ij + x_ij' beta + u_i,
                                                  Var(y) = mu + mu^2 / phi
  BB-GLMM   y_ij | u_i ~ BetaBin(n_ij, p_ij phi, (1 - p_ij) phi),
                                                  logit p_ij = x_ij' beta + u_i

Estimation: maximum likelihood. LMM - exact profile likelihood in lambda = s_u^2 / s^2. GLMMs -
marginal likelihood by adaptive Gauss-Hermite quadrature over u_i (Q nodes centred at the mode
of each unit) and L-BFGS-B with the analytic gradient; the standard errors come from the Hessian
(central differences of the gradient). A fit is `converged` only if the optimiser succeeded,
the Hessian of the free parameters is positive definite and doubling Q changes the log-likelihood
by less than QUAD_TOL - otherwise the caller must treat the model as failed (plan.py).

Inference for one coefficient (the same procedure for every model, no Kenward-Roger and no plain
chi-square LRT): signed likelihood root r = sign(beta_j) * sqrt(2 (l_full - l_null)); the
one-sided p-value is its parametric-bootstrap tail under the fitted null model
(bootstrap_signed_root).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, special

QUAD_NODES = 15               # adaptive Gauss-Hermite nodes per unit
QUAD_TOL = 0.01               # log-likelihood units between Q and 2Q nodes
SE_MAX = 1e3                  # a standard error above this means the coefficient is not identified
LOG_SU = (-8.0, 3.0)          # bounds of log s_u
LOG_PHI = (-5.0, 12.0)        # bounds of log phi


@dataclass
class Fit:
    names: list
    beta: np.ndarray
    se: np.ndarray
    loglik: float
    sigma_u: float
    scale: float              # s (LMM) or phi (NB2, beta-binomial)
    converged: bool
    message: str = ""
    extra: dict = field(default_factory=dict)

    def coef(self, name):
        j = self.names.index(name)
        return float(self.beta[j]), float(self.se[j])


class Model:
    """Data of one model: y, design matrix X with column names, unit codes, optional offset
    (log E) or trials n. Rows are sorted by unit."""
    family = None

    def __init__(self, y, X, names, units, offset=None, trials=None):
        order = np.argsort(units, kind="stable")
        self.y = np.asarray(y, float)[order]
        self.X = np.asarray(X, float)[order]
        self.names = list(names)
        codes = np.unique(np.asarray(units)[order], return_inverse=True)[1]
        self.g = codes
        self.starts = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
        self.G = len(self.starts)
        self.n_g = np.diff(np.r_[self.starts, len(codes)])
        self.offset = np.zeros(len(codes)) if offset is None else np.asarray(offset, float)[order]
        self.trials = None if trials is None else np.asarray(trials, float)[order]

    def with_y(self, y):
        m = object.__new__(type(self))
        m.__dict__.update(self.__dict__)
        m.y = np.asarray(y, float)
        return m

    def drop(self, name):
        """The null model of a coefficient: the same data without that column."""
        j = self.names.index(name)
        m = object.__new__(type(self))
        m.__dict__.update(self.__dict__)
        m.X = np.delete(self.X, j, axis=1)
        m.names = [n for n in self.names if n != name]
        return m


# ---------------------------------------------------------------------- LMM
class LMM(Model):
    family = "gaussian"

    def fit(self):
        y, X, g = self.y, self.X, self.g
        N, p = X.shape
        Sx = np.zeros((self.G, p))
        np.add.at(Sx, g, X)
        Sy = np.bincount(g, y, minlength=self.G)
        XtX, Xty, yty = X.T @ X, X.T @ y, float(y @ y)

        def profile(lam):
            c = lam / (1.0 + self.n_g * lam)
            A = XtX - (Sx * c[:, None]).T @ Sx
            b = Xty - Sx.T @ (c * Sy)
            beta = np.linalg.solve(A, b)
            rss = yty - float((c * Sy ** 2).sum()) - float(b @ beta)
            s2 = max(rss / N, 1e-300)
            ll = -0.5 * (N * np.log(2 * np.pi * s2) + np.log1p(self.n_g * lam).sum() + N)
            return ll, beta, s2, A

        res = optimize.minimize_scalar(lambda t: -profile(np.exp(t))[0], bounds=(-14.0, 14.0),
                                       method="bounded", options=dict(xatol=1e-10))
        lam = float(np.exp(res.x))
        if profile(0.0)[0] >= profile(lam)[0]:
            lam = 0.0
        ll, beta, s2, A = profile(lam)
        se = np.sqrt(np.diag(s2 * np.linalg.inv(A)))
        ok = bool(np.all(np.isfinite(beta)) and np.all(np.isfinite(se)) and np.isfinite(ll))
        return Fit(self.names, beta, se, float(ll), float(np.sqrt(lam * s2)), float(np.sqrt(s2)), ok,
                   "" if ok else "non-finite estimate")

    def simulate(self, fit, rng):
        u = rng.normal(0.0, fit.sigma_u, self.G)[self.g] if fit.sigma_u > 0 else 0.0
        return self.X @ fit.beta + u + rng.normal(0.0, fit.scale, len(self.y))


# ---------------------------------------------------------------------- GLMMs
def _nb_logpmf(y, eta, phi, trials):
    """NB2 log-pmf, d/d eta, d/d phi, d2/d eta2."""
    eta = np.clip(eta, -700, 700)
    mu = np.exp(eta)
    lp = (special.gammaln(y + phi) - special.gammaln(phi) - special.gammaln(y + 1.0)
          + phi * (np.log(phi) - np.log(phi + mu)) + y * (eta - np.log(phi + mu)))
    d_eta = phi * (y - mu) / (phi + mu)
    d_phi = (special.digamma(y + phi) - special.digamma(phi) + np.log(phi) - np.log(phi + mu)
             + (mu - y) / (phi + mu))
    return lp, d_eta, d_phi, -phi * mu * (phi + y) / (phi + mu) ** 2


def _bb_logpmf(y, eta, phi, trials):
    """Beta-binomial log-pmf (a = p phi, b = (1 - p) phi), d/d eta, d/d phi, d2/d eta2."""
    p = np.clip(special.expit(eta), 1e-12, 1 - 1e-12)
    a, b = p * phi, (1.0 - p) * phi
    lp = (special.gammaln(trials + 1.0) - special.gammaln(y + 1.0) - special.gammaln(trials - y + 1.0)
          + special.betaln(y + a, trials - y + b) - special.betaln(a, b))
    common = special.digamma(phi) - special.digamma(trials + phi)
    d_a = special.digamma(y + a) - special.digamma(a) + common
    d_b = special.digamma(trials - y + b) - special.digamma(b) + common
    da = phi * p * (1.0 - p)                                   # da / d eta = -db / d eta
    d2 = ((special.polygamma(1, y + a) - special.polygamma(1, a)
           + special.polygamma(1, trials - y + b) - special.polygamma(1, b)) * da ** 2
          + (d_a - d_b) * da * (1.0 - 2.0 * p))
    return lp, (d_a - d_b) * da, p * d_a + (1.0 - p) * d_b, d2


class GLMM(Model):
    logpmf = None

    def _modes(self, eta0, phi, su, tr, iters=50):
        """Mode and curvature scale of the integrand of every unit (Newton, warm start)."""
        u = getattr(self, "_u", None)
        u = np.zeros(self.G) if u is None else u.copy()
        f = type(self).logpmf
        for _ in range(iters):
            _, d1, _, d2 = f(self.y, eta0 + u[self.g], phi, tr)
            g1 = np.add.reduceat(d1, self.starts) - u / su ** 2
            g2 = np.minimum(np.add.reduceat(d2, self.starts) - 1.0 / su ** 2, -1e-8)
            step = np.clip(-g1 / g2, -3.0, 3.0)
            u = u + step
            if np.max(np.abs(step)) < 1e-9:
                break
        _, _, _, d2 = f(self.y, eta0 + u[self.g], phi, tr)
        g2 = np.minimum(np.add.reduceat(d2, self.starts) - 1.0 / su ** 2, -1e-8)
        self._u = u
        return u, 1.0 / np.sqrt(-g2)

    def _nll(self, theta, Q, grad=True):
        """Negative marginal log-likelihood and its gradient w.r.t. theta = (beta, log phi,
        log s_u). Adaptive Gauss-Hermite quadrature: Q nodes centred at the mode of each unit."""
        p = self.X.shape[1]
        beta, phi, su = theta[:p], np.exp(theta[p]), np.exp(theta[p + 1])
        eta0 = self.X @ beta + self.offset
        tr = self.trials
        uhat, sd = self._modes(eta0, phi, su, tr)
        z, w = np.polynomial.hermite.hermgauss(Q)
        U = uhat[:, None] + np.sqrt(2.0) * sd[:, None] * z[None, :]            # (G, Q)
        lp, d_eta, d_phi, _ = type(self).logpmf(self.y[:, None], eta0[:, None] + U[self.g], phi,
                                                None if tr is None else tr[:, None])
        S = (np.add.reduceat(lp, self.starts, axis=0) - 0.5 * np.log(2 * np.pi) - np.log(su)
             - U ** 2 / (2 * su ** 2) + (np.log(w) + z ** 2)[None, :] + np.log(np.sqrt(2.0) * sd)[:, None])
        ll_g = special.logsumexp(S, axis=1)
        if not grad:
            return -float(ll_g.sum())
        post = np.exp(S - ll_g[:, None])                                       # (G, Q)
        rows = post[self.g]
        g = np.r_[self.X.T @ (rows * d_eta).sum(axis=1), phi * float((rows * d_phi).sum()),
                  float((post * (U ** 2 / su ** 2 - 1.0)).sum())]
        return -float(ll_g.sum()), -g

    def _start(self):
        raise NotImplementedError

    def fit(self, Q=QUAD_NODES):
        p = self.X.shape[1]
        bounds = [(-30.0, 30.0)] * p + [LOG_PHI, LOG_SU]
        lo, hi = np.array(bounds).T
        best, ok = None, False
        for su0 in (np.log(0.3), np.log(0.05)):       # second start only if the first one fails
            x0 = np.r_[self._start(), np.log(5.0), su0]
            res = optimize.minimize(self._nll, x0, args=(Q,), jac=True, method="L-BFGS-B",
                                    bounds=bounds, options=dict(maxiter=1000, ftol=1e-13, gtol=1e-7))
            if not np.isfinite(res.fun):
                continue
            if best is None or res.fun < best.fun - 1e-8:
                best = res
                inside = (res.x > lo + 1e-6) & (res.x < hi - 1e-6)
                ok = bool(res.success or np.max(np.abs(res.jac[inside]), initial=0.0) < 1e-3)
            if ok:
                break
        if best is None:
            nan = np.full(p, np.nan)
            return Fit(self.names, nan, nan, float("nan"), float("nan"), float("nan"), False,
                       "optimiser failed")
        theta = best.x
        free = [i for i in range(p + 2) if lo[i] + 1e-6 < theta[i] < hi[i] - 1e-6]
        H = _hessian(lambda t: self._nll(_put(theta, free, t), Q)[1][free], theta[free])
        problems = [] if ok else [f"optimiser: {best.message}"]
        se = np.full(p, np.nan)
        ev = np.linalg.eigvalsh(H) if np.all(np.isfinite(H)) else np.array([np.nan])
        if not (np.all(np.isfinite(ev)) and ev.min() > 1e-10 * max(1.0, ev.max())):
            problems.append("Hessian is not positive definite")
        else:
            cov = np.linalg.inv(H)
            for k, i in enumerate(free):
                if i < p:
                    se[i] = np.sqrt(cov[k, k])
        dq = abs(self._nll(theta, 2 * Q, grad=False) - best.fun)
        if dq > QUAD_TOL:
            problems.append(f"quadrature: {dq:.3g} log-lik units between {Q} and {2 * Q} nodes")
        if not problems and not np.all(np.isfinite(se)):
            problems.append("no standard error")
        # no finite maximum: no events at all, a coefficient at its bound or not identified
        if self._degenerate():
            problems.append(self._degenerate())
        elif np.any(np.abs(theta[:p]) > 30.0 - 1e-6) or np.any(se[np.isfinite(se)] > SE_MAX):
            problems.append("a coefficient is not identified (estimate at the bound or huge standard error)")
        return Fit(self.names, theta[:p].copy(), se, -float(best.fun), float(np.exp(theta[p + 1])),
                   float(np.exp(theta[p])), not problems, "; ".join(problems),
                   dict(su_at_bound=bool(p + 1 not in free), phi_at_bound=bool(p not in free)))

    def _degenerate(self):
        return ""

    def _linear(self, fit, rng):
        u = rng.normal(0.0, fit.sigma_u, self.G)[self.g]
        return self.X @ fit.beta + self.offset + u


def _put(theta, free, t):
    out = np.array(theta, float)
    out[free] = t
    return out


def _hessian(grad, x, h=1e-5):
    """Hessian from central differences of an analytic gradient, symmetrised."""
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = h
        H[i] = (grad(x + e) - grad(x - e)) / (2 * h)
    return 0.5 * (H + H.T)


class NB2GLMM(GLMM):
    """Counts with exposure: offset = log E."""
    family = "nbinom2"
    logpmf = staticmethod(_nb_logpmf)

    def _start(self):
        b = np.zeros(self.X.shape[1])
        rate = (self.y.sum() + 0.5) / np.exp(self.offset).sum()
        b[self.names.index("Intercept")] = np.log(rate)
        return b

    def _degenerate(self):
        return "" if self.y.sum() > 0 else "no events: the maximum-likelihood estimate does not exist"

    def simulate(self, fit, rng):
        mu = np.exp(self._linear(fit, rng))
        lam = rng.gamma(fit.scale, mu / fit.scale)
        return rng.poisson(lam).astype(float)


class BetaBinGLMM(GLMM):
    """Blocked attempts out of `trials` attempts."""
    family = "betabinomial"
    logpmf = staticmethod(_bb_logpmf)

    def _start(self):
        b = np.zeros(self.X.shape[1])
        share = (self.y.sum() + 0.5) / (self.trials.sum() + 1.0)
        b[self.names.index("Intercept")] = special.logit(share)
        return b

    def _degenerate(self):
        if 0 < self.y.sum() < self.trials.sum():
            return ""
        return "all attempts blocked or none: the maximum-likelihood estimate does not exist"

    def simulate(self, fit, rng):
        p = np.clip(special.expit(self._linear(fit, rng)), 1e-12, 1 - 1e-12)
        q = rng.beta(p * fit.scale, (1.0 - p) * fit.scale)
        return rng.binomial(self.trials.astype(int), q).astype(float)


# ---------------------------------------------------------------------- inference
def signed_root(full, null, name):
    """sign(beta) * sqrt(2 (l_full - l_null)): a likelihood-ratio statistic with a direction."""
    b = full.coef(name)[0]
    return float(np.sign(b) * np.sqrt(max(0.0, 2.0 * (full.loglik - null.loglik))))


def bootstrap_signed_root(model, name, direction, n_boot, rng, max_failed=0.1):
    """One-sided test of a coefficient. direction = -1: H1 beta < 0; +1: H1 beta > 0.
    Returns a dict: estimate, se, root r, p_boot = (1 + #{r* at least as extreme}) / (B + 1) under
    the parametric bootstrap from the fitted null model, p_wald (asymptotic, descriptive),
    converged. Replicates whose fits do not converge are dropped and counted; more than
    max_failed of them makes the test fail (converged = False, p_boot = None)."""
    full, null_model = model.fit(), model.drop(name)
    null = null_model.fit()
    out = dict(name=name, direction=direction, estimate=None, se=None, root=None, p_boot=None,
               p_wald=None, n_boot=n_boot, n_failed=0, converged=False, message="",
               sigma_u=full.sigma_u, scale=full.scale, loglik=full.loglik)
    if not (full.converged and null.converged):
        out["message"] = "full: " + (full.message or "ok") + "; null: " + (null.message or "ok")
        if np.all(np.isfinite(full.beta)):
            out["estimate"], out["se"] = full.coef(name)
        return out
    est, se = full.coef(name)
    r = signed_root(full, null, name)
    out.update(estimate=est, se=se, root=r,
               p_wald=float(special.ndtr(-direction * est / se)) if se > 0 else None)
    stats = []
    for _ in range(n_boot):
        m = model.with_y(null_model.simulate(null, rng))
        f1, f0 = m.fit(), m.drop(name).fit()
        if f1.converged and f0.converged:
            stats.append(signed_root(f1, f0, name))
        else:
            out["n_failed"] += 1
    if n_boot and out["n_failed"] > max_failed * n_boot:
        out["message"] = f"{out['n_failed']} of {n_boot} bootstrap fits failed"
        return out
    stats = np.asarray(stats)
    extreme = np.sum(direction * stats >= direction * r - 1e-12)
    out.update(p_boot=float((1 + extreme) / (len(stats) + 1)), converged=True)
    return out
