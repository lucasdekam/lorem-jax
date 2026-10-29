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
