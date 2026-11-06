import numpy as np
import jax
import jax.numpy as jnp

import pytest
from ase.build import bulk, molecule
from marathon.data.properties import DEFAULT_PROPERTIES

from lorem.batching import to_batch, to_sample
from lorem.calculator import Calculator
from lorem.models.backbone import FiLM
from lorem.models.bec import LoremBEC
from lorem.models.mlip import Lorem

PROPERTIES = {
    **DEFAULT_PROPERTIES,
    "total_charge": {"shape": (1,), "storage": "atoms.info"},
    "external_field": {"shape": (3,), "storage": "atoms.info"},
}


def test_film_module():
    key = jax.random.key(0)
    num_atoms, d = 4, 6
    x = jax.random.normal(key, (num_atoms, d))
    atom_mask = jnp.ones(num_atoms, dtype=bool)
    c_i = jnp.array([[1.0, 0.2], [1.0, 0.2], [-1.0, 0.0], [-1.0, 0.0]])

    model = FiLM(features=d)
    params = model.init(key, c_i, x, atom_mask)

    assert not jnp.allclose(model.apply(params, c_i, x, atom_mask), x)
    np.testing.assert_allclose(model.apply(params, jnp.zeros_like(c_i), x, atom_mask), x)


@pytest.mark.parametrize("cls", [Lorem, LoremBEC])
def test_energy_depends_on_total_charge(cls):
    model = cls(
        cutoff=5.0,
        num_features=8,
        num_spherical_features=2,
        num_radial=4,
        num_message_passing=1,
        charge_conditioning=True,
    )
    calc = Calculator.from_model(model, properties=PROPERTIES)

    energies = []
    for atoms, q in [
        (molecule("H2O"), -1.0),
        (molecule("H2O"), 1.0),
        (bulk("Ar") * 2, 1.0),
    ]:
        atoms.info["total_charge"] = q
        calc.calculate(atoms)
        assert np.all(np.isfinite(calc.results["energy"]))
        assert np.all(np.isfinite(calc.results["forces"]))
        energies.append(calc.results["energy"])

    assert not np.allclose(energies[0], energies[1], atol=1e-6)


# -- field conditioning --


def field_model(**kwargs):
    return Lorem(
        cutoff=5.0,
        num_features=8,
        num_spherical_features=2,
        num_radial=4,
        max_degree=2,
        max_degree_lr=1,
        num_message_passing=1,
        field_conditioning=True,
        **kwargs,
    )


def field_batch(model, atoms, field):
    atoms = atoms.copy()
    atoms.info["external_field"] = np.asarray(field, dtype=float)
    sample = to_sample(
        atoms, model.cutoff, keys=(), inputs=model.inputs, properties=PROPERTIES
    )
    batch = to_batch([sample], [], inputs=model.inputs, properties=PROPERTIES)
    return jax.tree.map(jnp.asarray, batch)


@pytest.fixture(scope="module")
def water_field():
    model = field_model(predict_bec=True)
    params = model.init(jax.random.key(3), *model.dummy_inputs())
    atoms = molecule("H2O")
    atoms.rotate(37, "x")  # a dipole along z, so the field response is first order
    atoms.center()
    predict = jax.jit(model.predict)

    def run(atoms, field):
        out = predict(params, field_batch(model, atoms, field))
        n = len(atoms)
        return (
            float(out["energy"][0]),
            np.asarray(out["forces"][:n]),
            np.asarray(out["dipole"][0]),
            np.asarray(out["bec_z"][:n]),
        )

    return atoms, run


def test_field_sign_matters(water_field):
    atoms, run = water_field
    e_plus, *_ = run(atoms, [0.0, 0.0, 0.2])
    e_minus, *_ = run(atoms, [0.0, 0.0, -0.2])
    e_zero, *_ = run(atoms, [0.0, 0.0, 0.0])
    assert abs(e_plus - e_minus) > 1e-6
    assert abs(e_plus - e_zero) > 1e-6


def test_field_equivariance(water_field):
    # rotating, or mirroring, the structure and the field together leaves E unchanged
    atoms, run = water_field
    field = np.array([0.05, -0.1, 0.2])
    e, f, mu, _ = run(atoms, field)

    from scipy.spatial.transform import Rotation

    R = Rotation.from_euler("yz", [71, 23], degrees=True).as_matrix()
    rot = atoms.copy()
    c = atoms.get_center_of_mass()
    rot.positions = (atoms.positions - c) @ R.T + c
    e_rot, f_rot, mu_rot, _ = run(rot, R @ field)
    np.testing.assert_allclose(e_rot, e, atol=1e-5)
    np.testing.assert_allclose(f_rot, f @ R.T, atol=1e-4)
    np.testing.assert_allclose(mu_rot, R @ mu, atol=1e-4)

    mir = atoms.copy()
    mir.positions[:, 2] *= -1
    e_mir, *_ = run(mir, field * [1, 1, -1])
    np.testing.assert_allclose(e_mir, e, atol=1e-5)


def test_zero_field_has_no_preferred_direction(water_field):
    # at E = 0 the model is plain O(3)-invariant: mirroring the structure alone
    atoms, run = water_field
    mir = atoms.copy()
    mir.positions[:, 2] *= -1
    e, *_ = run(atoms, [0.0, 0.0, 0.0])
    e_mir, *_ = run(mir, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(e_mir, e, atol=1e-5)


def test_dipole_and_bec_z_are_field_derivatives(water_field):
    atoms, run = water_field
    field, h = np.array([0.0, 0.0, 0.1]), 0.01
    _, f, mu, bec_z = run(atoms, field)
    e_p, f_p, *_ = run(atoms, field + [0, 0, h])
    e_m, f_m, *_ = run(atoms, field - [0, 0, h])
    scale = np.abs(mu[2]) + 1e-3
    np.testing.assert_allclose(mu[2], -(e_p - e_m) / (2 * h), atol=2e-3 * scale + 1e-4)
    np.testing.assert_allclose(
        bec_z, (f_p - f_m) / (2 * h), atol=2e-3 * np.abs(bec_z).max() + 1e-4
    )
    assert np.abs(bec_z).max() > 0


def test_calculator_reports_dipole():
    model = field_model()
    calc = Calculator.from_model(model, properties=PROPERTIES)
    atoms = molecule("H2O")
    atoms.center(vacuum=4.0)
    atoms.pbc = True
    for field in ([0.0, 0.0, 0.1], [0.0, 0.0, -0.1]):
        atoms.info["external_field"] = field
        atoms.calc = calc
        assert np.all(np.isfinite(atoms.get_forces()))
        assert calc.get_property("dipole", atoms).shape == (3,)


# -- the field from the charge --


def test_field_from_charge_values():
    from lorem.models.backbone import EPSILON_0, field_from_charge

    cell = jnp.array([np.diag([10.0, 8.0, 30.0]), np.zeros((3, 3))])
    E = field_from_charge(jnp.array([0.4, 0.4]), cell, 2)
    np.testing.assert_allclose(E[0], [0.0, 0.0, 0.4 / (EPSILON_0 * 80.0)], rtol=1e-6)
    np.testing.assert_allclose(E[1], 0.0)  # padding: no cell, no field
    grad = jax.grad(lambda c: field_from_charge(jnp.array([0.4, 0.4]), c, 2).sum())(cell)
    assert np.all(np.isfinite(grad))


@pytest.fixture(scope="module")
def water_slab_charge():
    model = Lorem(
        cutoff=5.0,
        num_features=8,
        num_spherical_features=2,
        num_radial=4,
        max_degree=2,
        max_degree_lr=1,
        num_message_passing=1,
        charge_conditioning=True,
        field_from_charge=True,
    )
    params = model.init(jax.random.key(3), *model.dummy_inputs())
    atoms = molecule("H2O")
    atoms.rotate(37, "x")  # a dipole along z, so the field couples at first order
    atoms.set_cell([9.0, 9.5, 20.0])
    atoms.center()
    atoms.pbc = True
    predict = jax.jit(model.predict)

    def run(atoms, q):
        atoms = atoms.copy()
        atoms.info["total_charge"] = q
        sample = to_sample(
            atoms, model.cutoff, keys=(), inputs=model.inputs, properties=PROPERTIES
        )
        batch = to_batch([sample], [], inputs=model.inputs, properties=PROPERTIES)
        out = predict(params, jax.tree.map(jnp.asarray, batch))
        return float(out["energy"][0]), float(out["work_function"][0])

    return model, atoms, run


def test_field_from_charge_needs_charge_only():
    model = Lorem(charge_conditioning=True, field_from_charge=True, field_conditioning=True)
    with pytest.raises(ValueError):
        model.init(jax.random.key(0), *model.dummy_inputs())


def test_field_from_charge_sees_the_mirror(water_slab_charge):
    # the charge's field points along +z, so mirroring the water changes E at q != 0,
    # and at q = 0 the model is plain O(3)-invariant again
    _, atoms, run = water_slab_charge
    mir = atoms.copy()
    mir.positions[:, 2] = atoms.cell[2, 2] - atoms.positions[:, 2]
    assert abs(run(mir, 0.05)[0] - run(atoms, 0.05)[0]) > 1e-6
    np.testing.assert_allclose(run(mir, 0.0)[0], run(atoms, 0.0)[0], atol=1e-5)


def test_work_function_includes_the_field_path(water_slab_charge):
    _, atoms, run = water_slab_charge
    q, h = 0.02, 1e-3
    _, wf = run(atoms, q)
    e_p, _ = run(atoms, q + h)
    e_m, _ = run(atoms, q - h)
    np.testing.assert_allclose(wf, (e_p - e_m) / (2 * h), rtol=2e-3, atol=1e-4)


def gated_slab_model(**kwargs):
    return Lorem(cutoff=5.0, num_features=8, num_spherical_features=2, num_radial=4, max_degree=2,
                 max_degree_lr=1, num_message_passing=1, charge_conditioning=True,
                 field_from_charge=True, **kwargs)


def test_field_gate_is_the_only_new_parameter_and_starts_at_zero():
    assert Lorem().field_gate is False
    off, on = gated_slab_model(), gated_slab_model(field_gate=True)
    p_off = off.init(jax.random.key(0), *off.dummy_inputs())
    p_on = on.init(jax.random.key(0), *on.dummy_inputs())
    gate = p_on["params"].pop("field_coupling_gate")
    assert np.asarray(gate).shape == () and float(gate) == 0.0
    assert jax.tree.structure(p_off) == jax.tree.structure(p_on)
    for a, b in zip(jax.tree.leaves(p_off), jax.tree.leaves(p_on)):
        np.testing.assert_array_equal(a, b)


def test_field_gate_scales_the_coupling(water_slab_charge):
    # gate 0: the field path is off, so dE/dQ is the FiLM's alone (the model with the
    # coupling's tensor product zeroed); gate 1: exactly the ungated model
    _, atoms, _ = water_slab_charge
    off, on = gated_slab_model(), gated_slab_model(field_gate=True)
    params = off.init(jax.random.key(3), *off.dummy_inputs())
    atoms = atoms.copy()
    atoms.info["total_charge"] = 0.05
    sample = to_sample(atoms, off.cutoff, keys=(), inputs=off.inputs, properties=PROPERTIES)
    batch = jax.tree.map(jnp.asarray, to_batch([sample], [], inputs=off.inputs, properties=PROPERTIES))

    def predict(model, p):
        out = model.predict(p, batch)
        return float(out["energy"][0]), float(out["work_function"][0])

    gated = lambda g: {"params": {**params["params"], "field_coupling_gate": jnp.float32(g)}}  # noqa: E731
    np.testing.assert_allclose(predict(on, gated(1.0)), predict(off, params), rtol=1e-6, atol=1e-6)
    # the coupling is the Tensor that takes the field harmonics: the last Tensor before
    # the message pass, Tensor_0 here
    no_coupling = {"params": {**params["params"], "Tensor_0": jax.tree.map(jnp.zeros_like,
                                                                          params["params"]["Tensor_0"])}}
    np.testing.assert_allclose(predict(on, gated(0.0)), predict(off, no_coupling), rtol=1e-6, atol=1e-6)
    assert abs(predict(on, gated(0.0))[1] - predict(off, params)[1]) > 1e-6  # the gate does something


def test_field_gate_needs_a_field_path():
    model = Lorem(charge_conditioning=True, field_gate=True)
    with pytest.raises(ValueError):
        model.init(jax.random.key(0), *model.dummy_inputs())


@pytest.mark.parametrize("field_from_charge", [False, True])
def test_charge_second_derivatives(field_from_charge):
    # bec_z = (A eps0) dF/dq and d2Edq2 = d2E/dq2, against differences of the forces and
    # the work function in q
    from lorem.models.backbone import EPSILON_0

    model = Lorem(
        cutoff=5.0,
        num_features=8,
        num_spherical_features=2,
        num_radial=4,
        max_degree=2,
        max_degree_lr=1,
        num_message_passing=1,
        charge_conditioning=True,
        field_from_charge=field_from_charge,
        predict_bec=True,
    )
    params = model.init(jax.random.key(5), *model.dummy_inputs())
    atoms = molecule("H2O")
    atoms.rotate(37, "x")
    atoms.set_cell([9.0, 9.5, 20.0])
    atoms.center()
    atoms.pbc = True
    predict = jax.jit(model.predict)

    def run(q):
        a = atoms.copy()
        a.info["total_charge"] = q
        sample = to_sample(a, model.cutoff, keys=(), inputs=model.inputs, properties=PROPERTIES)
        batch = to_batch([sample], [], inputs=model.inputs, properties=PROPERTIES)
        out = predict(params, jax.tree.map(jnp.asarray, batch))
        n = len(a)
        return (np.asarray(out["forces"][:n]), float(out["work_function"][0]),
                np.asarray(out["bec_z"][:n]), float(out["d2Edq2"][0]))

    # a fourth-order stencil: the untrained model bends in q, so a centred difference
    # carries an O(h^2) error of ~1% at h = 0.01
    q, h = 0.3, 1e-2
    _, _, bec, d2 = run(q)
    (f_p, w_p, *_), (f_m, w_m, *_) = run(q + h), run(q - h)
    (f_pp, w_pp, *_), (f_mm, w_mm, *_) = run(q + 2 * h), run(q - 2 * h)
    d2_fd = (8 * (w_p - w_m) - (w_pp - w_mm)) / (12 * h)
    fd = 9.0 * 9.5 * EPSILON_0 * (8 * (f_p - f_m) - (f_pp - f_mm)) / (12 * h)
    np.testing.assert_allclose(d2, d2_fd, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(bec, fd, atol=1e-3 * np.abs(fd).max() + 1e-5)
    assert np.abs(bec).max() > 0
