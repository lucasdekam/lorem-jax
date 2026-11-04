"""coulomb_energy: the long-range charges' own Coulomb energy added to Lorem's readout, gated."""

import numpy as np
import jax
import jax.numpy as jnp

import pytest
from ase import Atoms
from ase.build import molecule
from jaxpme.batched_mixed import Ewald

from lorem.batching import to_batch, to_sample
from lorem.models.mlip import COULOMB, Lorem

TIP3P = {"O": -0.834, "H": 0.417}


def batch_of(atoms, cutoff=5.0):
    return to_batch([to_sample(atoms, cutoff, keys=())], [])


def coulomb(atoms, charges, cutoff=5.0):
    """sum_i q_i phi_i through the Ewald call Lorem makes, in e^2/A."""
    b = batch_of(atoms, cutoff)
    q = np.zeros(b.sr.atom_mask.shape[0], np.float32)
    q[: len(atoms)] = charges
    phi = Ewald().potentials(jnp.asarray(q), b.sr, b.nopbc, b.pbc)
    return float(jnp.sum(q * phi))


def water_slab(theta_deg=30.0):
    """One water in water_Ne_single's cell, T T F, its dipole at theta from +z in xz."""
    w = molecule("H2O")
    w.positions -= w.positions[0]
    w.rotate(theta_deg, "y")  # ase's H2O has its dipole along +z
    w.positions += [4.2, 4.9, 17.5]
    w.cell = [8.464068, 9.773464, 35.0]
    w.pbc = [True, True, False]
    return w


def test_potentials_carry_the_half():
    # an isolated pair: sum_i q_i phi_i is the pair energy q1 q2 / d, counted once
    pair = Atoms("HH", positions=[[0, 0, 0], [0, 0, 2.0]])
    np.testing.assert_allclose(coulomb(pair, [1.0, -1.0]), -1.0 / 2.0, rtol=1e-5)


def test_slab_ewald_has_the_lateral_images():
    # the water's energy with its own lateral images, against a direct lattice sum
    w = water_slab()
    q = np.array([TIP3P[s] for s in w.get_chemical_symbols()])
    lone = w.copy()
    lone.cell, lone.pbc = None, False
    e_images = COULOMB * (coulomb(w, q) - coulomb(lone, q))

    n = np.arange(-150, 151)
    I, J = np.meshgrid(n, n, indexing="ij")
    keep = (I != 0) | (J != 0)
    L = I[keep][:, None] * w.cell[0] + J[keep][:, None] * w.cell[1]
    direct = sum(q[i] * q[j] * np.sum(1.0 / np.linalg.norm(w.positions[i] - w.positions[j] - L, axis=1))
                 for i in range(3) for j in range(3))
    # 0.3 % apart (13.28 vs 13.25 meV): float32 Ewald and the direct sum's tail; a
    # wrong factor 1/2 or unit would be 100 % or more
    np.testing.assert_allclose(e_images, 0.5 * float(COULOMB) * direct, rtol=1e-2)


def small(**kwargs):
    return Lorem(cutoff=5.0, num_features=8, num_spherical_features=2, num_radial=4,
                 max_degree=2, max_degree_lr=0, num_message_passing=1, **kwargs)


def test_off_by_default_and_the_gate_is_the_only_new_parameter():
    assert Lorem().coulomb_energy is False
    off, on = small(), small(coulomb_energy=True)
    p_off = off.init(jax.random.key(0), *off.dummy_inputs())
    p_on = on.init(jax.random.key(0), *on.dummy_inputs())
    gate = p_on["params"].pop("coulomb_gate")
    assert np.asarray(gate).shape == () and float(gate) == 0.0
    assert jax.tree.structure(p_off) == jax.tree.structure(p_on)
    for a, b in zip(jax.tree.leaves(p_off), jax.tree.leaves(p_on)):
        np.testing.assert_array_equal(a, b)


def with_gate(params, value):
    return {"params": {**params["params"], "coulomb_gate": jnp.asarray(value, jnp.float32)}}


def test_energy_gains_exactly_the_gated_coulomb_energy():
    off, on = small(), small(coulomb_energy=True)
    params = off.init(jax.random.key(0), *off.dummy_inputs())
    w = water_slab()
    b = batch_of(w)
    args = (b.atomic_numbers, b.sr, b.nopbc, b.pbc, b.inputs)
    e_off, state = off.apply(params, *args, capture_intermediates=True, mutable=["intermediates"])
    # the scalar-charge head: the one MLP whose hidden layer is 2d wide
    (head,) = [k for k, v in params["params"].items()
               if k.startswith("MLP") and v["Dense_0"]["kernel"].shape[1] == 2 * off.num_features]
    q = np.asarray(state["intermediates"][head]["__call__"][0])[:, 0]
    q = jnp.asarray(q * np.asarray(b.sr.atom_mask), jnp.float32)
    phi = Ewald().potentials(q, b.sr, b.nopbc, b.pbc)
    mask = np.asarray(b.sr.atom_mask)
    coulomb = np.asarray(COULOMB * q * phi)[mask]
    assert np.abs(coulomb).max() > 1e-4  # the term is not trivially zero

    # at init (gate 0) the model is the one without the term
    np.testing.assert_array_equal(np.asarray(on.apply(with_gate(params, 0.0), *args)), np.asarray(e_off))
    for g in (1.0, 0.5):
        e_on = on.apply(with_gate(params, g), *args)
        np.testing.assert_allclose(np.asarray(e_on - e_off)[mask], g * coulomb, rtol=1e-4, atol=1e-6)


def test_needs_lr():
    model = small(coulomb_energy=True, lr=False)
    with pytest.raises(ValueError, match="needs lr"):
        model.init(jax.random.key(0), *model.dummy_inputs())
