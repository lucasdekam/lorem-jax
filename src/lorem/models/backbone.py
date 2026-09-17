import numpy as np
import jax
import jax.numpy as jnp

from collections.abc import Sequence

import e3x
import flax.linen as nn
from flax.core import FrozenDict
from marathon.utils import masked


def _masked(fn, x, mask):
    """Apply fn only where mask is True."""
    return masked(fn, x, mask)


# -- initial embeddings --


class Initial(nn.Module):
    cutoff: float = 5.0
    max_degree: int = 4
    num_features: int = 128
    num_radial: int = 32
    num_species: int = 8
    num_spherical_features: int = 4
    cutoff_fn: str = "cosine_cutoff"
    radial_basis: str = "basic_bernstein"

    @nn.compact
    def __call__(
        self,
        R_ij,
        Z_i,
        pair_mask,
        atom_mask,
    ):
        cutoff_fn = getattr(e3x.nn.functions, self.cutoff_fn)

        R_ij, r_ij = e3x.ops.normalize_and_return_norm(R_ij, axis=-1)
        R_ij *= pair_mask[..., None]

        cutoffs = cutoff_fn(r_ij, cutoff=self.cutoff) * pair_mask  # -> [pairs]

        radial_expansion = (
            RadialEmbedding(
                self.num_radial,
                self.cutoff,
                function=self.radial_basis,
            )(r_ij)
            * cutoffs[..., None]
        )

        spherical_expansion = e3x.so3.spherical_harmonics(
            R_ij, self.max_degree, r_is_normalized=True
        )
        spherical_expansion *= pair_mask[..., None]

        species_expansion = (
            ChemicalEmbedding(num_features=self.num_species)(Z_i) * atom_mask[..., None]
        )

        return (
            radial_expansion,
            spherical_expansion,
            species_expansion,
            cutoffs,
            r_ij,
        )


class ChemicalEmbedding(nn.Module):
    num_features: int
    total_species: int = 100

    @nn.compact
    def __call__(self, species):
        return nn.Embed(num_embeddings=self.total_species, features=self.num_features)(
            species
        )


class RadialEmbedding(nn.Module):
    num_features: int
    cutoff: int
    function: str = "basic_gaussian"
    args: FrozenDict = FrozenDict({})
    learned_transform: bool = False

    @nn.compact
    def __call__(self, r):
        function = getattr(e3x.nn.functions, self.function)

        expansion = function(
            r, **{"limit": self.cutoff, "num": self.num_features, **self.args}
        )

        if self.learned_transform:
            expansion = nn.Dense(features=self.num_features, use_bias=False)(expansion)

        return expansion


# -- basic modules --


class MLP(nn.Module):
    features: Sequence[int]
    activation: str = "silu"
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        activation = getattr(jax.nn, self.activation)
        num_layers = len(self.features)

        for i, f in enumerate(self.features):
            x = nn.Dense(features=f, use_bias=self.use_bias)(x)
            if i != num_layers - 1:
                x = activation(x)

        return x


class Update(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x, y, atom_mask):
        x += _masked(
            MLP(features=[2 * self.features, self.features]),
            y,
            atom_mask,
        )
        x = _masked(nn.LayerNorm(), x, atom_mask)
        x += _masked(MLP(features=[2 * self.features, self.features]), x, atom_mask)
        x = _masked(nn.LayerNorm(), x, atom_mask)

        return x


# -- other modules --


class RadialCoefficients(nn.Module):
    features: int

    @nn.compact
    def __call__(self, pair_features, radial_expansion, cutoffs, pair_mask):
        num_radial = radial_expansion.shape[-1]

        coefficients = _masked(
            MLP(
                features=[
                    self.features,
                    num_radial * self.features,
                ]
            ),
            pair_features,
            pair_mask,
        )
        coefficients = coefficients.reshape(-1, num_radial, self.features)
        coefficients = jnp.einsum("prf,pr->pf", coefficients, radial_expansion)

        return coefficients


# -- helpers to deal with spherical features --


def degree_wise_trace(
    x,
    max_degree,
):
    segments = np.concatenate(
        [np.array([l] * (2 * l + 1)) for l in range(max_degree + 1)]
    ).reshape(-1)

    return jax.vmap(
        lambda _x: jax.ops.segment_sum(_x, segments, num_segments=(max_degree + 1)),
    )(x)


def degree_wise_repeat(x, max_degree, axis):
    repeats = np.array([2 * l + 1 for l in range(max_degree + 1)])

    return jnp.repeat(x, repeats, total_repeat_length=repeats.sum(), axis=axis)


def degree_wise_repeat_last_axis(x, max_degree: int):
    return jax.vmap(
        lambda y: degree_wise_repeat(y, max_degree, -1),
        in_axes=-1,
        out_axes=-1,
    )(x)


# keeps x/||x|| (the norm's gradient) smooth at x=0 for higher-order autodiff
_SPHERICAL_NORM_EPS = 1e-12


def spherical_norm(X, max_degree):
    squared = jax.lax.square(X)
    trace = degree_wise_trace(squared, max_degree)
    return jnp.sqrt(trace + _SPHERICAL_NORM_EPS)


def spherical_norm_last_axis(X, max_degree):
    # X is a e3x-style array, i.e. [batch, 1|2, lm, features]:
    # we vmap over parity and feature dimensions
    return jax.vmap(
        lambda z: jax.vmap(
            lambda x: spherical_norm(x, max_degree),
            in_axes=-1,
            out_axes=-1,
        )(z),
        in_axes=1,
        out_axes=1,
    )(X)


# -- charge conditioning --


class ChargeConditioning(nn.Module):
    # FiLM conditioning of invariant node features on the per-atom charge Q_i
    features: int

    @nn.compact
    def __call__(self, Q_i, x, atom_mask):
        gamma_beta = _masked(
            MLP(features=[self.features, 2 * self.features]),
            Q_i[..., None],
            atom_mask,
        )
        gamma, beta = jnp.split(gamma_beta, 2, axis=-1)
        return (1.0 + gamma) * x + beta  # near-identity at init


class QuadraticReadout(nn.Module):
    """Per-atom energy as an explicit quadratic in the total charge.

        E_i = E0_i(R) + Phi0_i(R) Q + 1/2 kappa_i(R) Q^2

    The alternative, `ChargeConditioning`, lets Q modulate the features and so
    leaves E(Q) an arbitrary learned function. Here the Q dependence is the
    capacitor expansion the capacitance formalism assumes, with all three
    coefficients learned per structure -- so `d2E/dQ2` is `sum_i kappa_i`, a
    quantity the model states rather than one a second derivative has to
    uncover, and no constant-capacitance assumption is imposed.

    `kappa` passes through a softplus, so every atom's curvature contribution
    is positive and the structure sum cannot come out negative: a capacitance
    has a sign. `kappa_bias` shifts the softplus so the structure total starts
    near the data rather than at `n_atoms * softplus(0) = 0.69 n_atoms`, which
    for a 108-atom slab would be ~75 V/e against a true value near 9.
    """

    features: int
    # Calibrated so a ~100-atom slab starts near the razor data's 9 V/e rather
    # than at an arbitrary scale. Two things make the naive estimate too low:
    # the readout runs once per site (initial, per message-passing step, and
    # long-range), and softplus is convex so its mean over the head's output
    # spread exceeds softplus(bias). Measured on the real 108-atom geometry
    # with two readout sites: -2.5 gives 23 V/e, -3.5 gives ~9. It is a
    # learnable parameter, so this only sets where training starts.
    kappa_bias_init: float = -3.5

    @nn.compact
    def __call__(self, Q_i, x, atom_mask):
        out = _masked(
            MLP(features=[self.features, self.features, 3]), x, atom_mask
        )
        e0, phi0, kappa = out[..., 0], out[..., 1], out[..., 2]
        bias = self.param(
            "kappa_bias", nn.initializers.constant(self.kappa_bias_init), ()
        )
        kappa = jax.nn.softplus(kappa + bias)
        return e0 + phi0 * Q_i + 0.5 * kappa * Q_i**2
