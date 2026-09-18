import jax
import jax.numpy as jnp

import e3x
import flax.linen as nn
from jaxpme.batched_mixed import Ewald

from lorem.models.backbone import (
    MLP,
    ChargeConditioning,
    QuadraticReadout,
    Initial,
    RadialCoefficients,
    Update,
    degree_wise_repeat_last_axis,
    spherical_norm_last_axis,
)
from lorem.models.backbone import (
    _masked as masked,
)
from lorem.transforms import ToBatch, ToSample

# vacuum permittivity in e^2 / (eV * Angstrom)
EPSILON_0 = 0.005526349406


class Lorem(nn.Module):
    cutoff: float = 5.0
    max_degree: int = 6
    max_degree_lr: int = 2
    num_features: int = 128
    num_radial: int = 32
    num_species: int = 8
    num_spherical_features: int = 8
    cutoff_fn: str = "cosine_cutoff"
    radial_basis: str = "basic_bernstein"
    lr: bool = True
    # how the total charge enters the energy:
    #   "film"            FiLM-modulate the node features once, early. E(Q) is
    #                     then an arbitrary learned function.
    #   "quadratic"       no FiLM; the readouts emit (E0, Phi0, kappa) from
    #                     charge-free features and E(Q) is exactly quadratic,
    #                     so d2E/dQ2 is a stated quantity, positive by
    #                     construction, and available without autodiff.
    #   "quadratic_const" the same, with kappa a single learned scalar for the
    #                     whole model instead of a function of the geometry --
    #                     the constant-capacitance assumption, as a control.
    #   "quadratic_film"  the quadratic form plus one FiLM-conditioned residual
    #                     head, for anharmonicity beyond second order. E(Q) is
    #                     no longer exactly quadratic, so d2E/dQ2 has to come
    #                     off the derivative rather than off kappa.
    charge_conditioning: str = "film"
    # Pre-activation offset on the *structure* curvature, so d2E/dQ2 starts at
    # softplus(kappa_init). Zero is the neutral default and encodes nothing
    # about system size or dataset; an experiment that wants to start on its
    # data passes the number in from its config, where it can be read back
    # against the dataset it came from.
    kappa_init: float = 0.0
    num_message_passing: int = 0
    equivariant_message_passing: bool = True
    initialize_node_features: bool = True

    @property
    def to_batch(self):
        return ToBatch

    @property
    def to_sample(self):
        return ToSample

    def __call__(self, Z_i, sr, nopbc, pbc, Q):
        """Per-atom energies -- the contract every caller but `LoremQ` uses."""
        return self.coefficients(Z_i, sr, nopbc, pbc, Q)[0]

    @nn.compact
    def coefficients(self, Z_i, sr, nopbc, pbc, Q):
        """Per-atom energies, plus the charge expansion's structure-level terms.

        Returns `(energy_i, B_s, K_s)`. In the exactly-quadratic modes

            E_s(q) = A_s + B_s q + 1/2 K_s q^2

        holds identically, so `B_s + K_s q` *is* dE/dq and `K_s` *is* d2E/dq2 --
        both without autodiff, and `bec_z` becomes one reverse pass over
        `B_s + K_s q` instead of a forward-over-reverse pass over the energy.
        `LoremQ.predict` takes that route when it is available.

        `B_s` and `K_s` are None for `film` and `quadratic_film`, where E(q) is
        not quadratic and only the derivative is the honest answer. Returning
        None rather than a partial coefficient is deliberate: a caller cannot
        then quietly use the quadratic part as though it were the whole.
        """
        R = sr.positions
        i = sr.centers
        j = sr.others
        cell = sr.cell
        cell_shifts = sr.cell_shifts
        pair_mask = sr.pair_mask
        atom_mask = sr.atom_mask
        atom_to_structure = sr.atom_to_structure

        R_ij = (
            R[j] - R[i] + jnp.einsum("pA,pAa->pa", cell_shifts, cell[sr.pair_to_structure])
        )

        num_atoms = Z_i.shape[0]
        num_pairs = R_ij.shape[0]

        max_degree = self.max_degree
        max_degree_lr = self.max_degree_lr
        num_l = self.max_degree + 1
        num_lm = int((self.max_degree + 1) ** 2)

        d = self.num_features
        s = self.num_spherical_features

        Q_i = Q[atom_to_structure] * atom_mask

        modes = ("film", "quadratic", "quadratic_const", "quadratic_film")
        if self.charge_conditioning not in modes:
            raise ValueError(
                f"unknown charge_conditioning {self.charge_conditioning!r}; "
                f"expected one of {modes}"
            )
        use_film = self.charge_conditioning == "film"
        quadratic = self.charge_conditioning != "film"
        # the constant variant learns one global kappa, so the per-atom head
        # would only be dead weight
        per_atom_kappa = self.charge_conditioning != "quadratic_const"

        def readout(x):
            """One readout site's contribution to (E0_i, Phi0_i, kappa_i).

            Called once per site, so each call makes its own parameters exactly
            as the three inline MLPs did before. For `film` there is no charge
            expansion and the site produces an energy directly.
            """
            if quadratic:
                return QuadraticReadout(d, geometry_dependent_kappa=per_atom_kappa)(
                    x, atom_mask
                )
            e = masked(MLP(features=[d, d, 1]), x, atom_mask)[..., 0]
            zero = jnp.zeros_like(e)
            return e, zero, zero

        # per-atom accumulators, summed over readout sites
        e0_i = jnp.zeros((num_atoms,), dtype=R.dtype)
        phi0_i = jnp.zeros((num_atoms,), dtype=R.dtype)
        kappa_i = jnp.zeros((num_atoms,), dtype=R.dtype)

        def accumulate(x):
            nonlocal e0_i, phi0_i, kappa_i
            e, phi, kap = readout(x)
            e0_i = e0_i + e
            phi0_i = phi0_i + phi
            kappa_i = kappa_i + kap

        # empirical factors to make var of equivariant norm more uniform across l
        l_factors = (
            jnp.array([(2 * l + 1) for l in range(max_degree + 1)], dtype=float) ** 0.25
        )

        # -- initial embeddings --
        radial, spherical, species, cutoffs, r_ij = Initial(
            cutoff=self.cutoff,
            max_degree=self.max_degree,
            num_features=self.num_features,
            num_radial=self.num_radial,
            num_species=self.num_species,
            num_spherical_features=self.num_spherical_features,
            cutoff_fn=self.cutoff_fn,
            radial_basis=self.radial_basis,
        )(
            R_ij,
            Z_i,
            pair_mask,
            atom_mask,
        )

        # -- learned linear transformation of radial expansion --
        edges_scalar = RadialCoefficients(d)(
            jnp.concatenate([species[i], species[j]], axis=-1),
            radial,
            cutoffs,
            pair_mask,
        )

        # -- initial scalar and equivariant (spherical) node features
        if self.initialize_node_features:
            nodes_scalar = masked(nn.Dense(d, use_bias=True), species, atom_mask)
        else:
            nodes_scalar = jnp.zeros((num_atoms, d), dtype=species.dtype)

        updates = (
            jax.ops.segment_sum(
                masked(nn.Dense(d, use_bias=False), edges_scalar, pair_mask),
                i,
                num_segments=num_atoms,
            )
            * atom_mask[..., None]
        )

        nodes_scalar = Update(d)(nodes_scalar, updates, atom_mask)
        if use_film:
            nodes_scalar = ChargeConditioning(d)(Q_i, nodes_scalar, atom_mask)

        coefficients = masked(
            nn.Dense(num_l * s, use_bias=False), edges_scalar, pair_mask
        ).reshape(num_pairs, num_l, s)
        coefficients = degree_wise_repeat_last_axis(coefficients, max_degree)
        edges_spherical = jnp.einsum("plf,pl->plf", coefficients, spherical)

        nodes_spherical = (
            jax.ops.segment_sum(
                edges_spherical.reshape(num_pairs, 1, num_lm, s),
                i,
                num_segments=num_atoms,
            )
            * atom_mask[..., None, None, None]
        )
        nodes_spherical = e3x.nn.TensorDense(use_bias=False, include_pseudotensors=False)(
            nodes_spherical
        )

        # -- mix equivariant information into scalar node features --
        norms = spherical_norm_last_axis(nodes_spherical, max_degree)
        updates = (norms * l_factors[None, None, :, None]).reshape(num_atoms, -1)

        nodes_scalar = Update(d)(nodes_scalar, updates, atom_mask)

        # -- initial prediction --
        accumulate(nodes_scalar)

        # -- message passing (if turned on) --
        for _ in range(self.num_message_passing):
            edges_scalar = RadialCoefficients(d)(
                jnp.concatenate([nodes_scalar[i], nodes_scalar[j]], axis=-1),
                radial,
                cutoffs,
                pair_mask,
            )
            updates = (
                jax.ops.segment_sum(
                    masked(
                        nn.Dense(d, use_bias=False),
                        edges_scalar,
                        pair_mask,
                    ),
                    i,
                    num_segments=num_atoms,
                )
                * atom_mask[..., None]
            )

            nodes_scalar = Update(d)(nodes_scalar, updates, atom_mask)

            if self.equivariant_message_passing:
                coefficients = masked(
                    nn.Dense(num_l * s, use_bias=False),
                    edges_scalar,
                    pair_mask,
                ).reshape(num_pairs, num_l, s)
                coefficients = degree_wise_repeat_last_axis(coefficients, max_degree)
                edges_spherical = jnp.einsum(
                    "plf,pl->plf", coefficients, spherical
                ).reshape(num_pairs, 1, num_lm, s)

                messages = (
                    e3x.nn.MessagePass(include_pseudotensors=False)(
                        nodes_spherical,
                        edges_spherical,
                        dst_idx=i,
                        src_idx=j,
                    )
                    * atom_mask[..., None, None, None]
                )
                nodes_spherical = e3x.nn.Tensor(include_pseudotensors=False)(
                    e3x.nn.Dense(use_bias=False, features=s)(nodes_spherical),
                    e3x.nn.Dense(use_bias=False, features=s)(messages),
                )

                norms = spherical_norm_last_axis(nodes_spherical, max_degree)
                updates = (norms * l_factors[None, None, :, None]).reshape(num_atoms, -1)
                nodes_scalar = Update(d)(nodes_scalar, updates, atom_mask)

            # -- residual prediction --
            accumulate(nodes_scalar)

        if self.lr:
            # -- compute LR potentials --
            scalar_charges = masked(MLP(features=[2 * d, 1]), nodes_scalar, atom_mask)

            spherical_charges = e3x.nn.TensorDense(
                features=1,
                use_bias=False,
                max_degree=max_degree_lr,
                include_pseudotensors=False,
            )(nodes_spherical).reshape(num_atoms, -1)
            charges = jnp.concatenate([scalar_charges, spherical_charges], axis=-1)

            calculator = Ewald()
            potentials = jax.vmap(
                lambda q: calculator.potentials(q, sr, nopbc, pbc),
                in_axes=-1,
                out_axes=-1,
            )(charges)

            scalar_potential = potentials[..., 0][..., None]
            spherical_potential = potentials[..., 1:].reshape(num_atoms, 1, -1, 1)

            # -- combine LR potentials back into local features --
            spherical_potential = e3x.nn.Dense(s, use_bias=False)(spherical_potential)
            spherical_updates = e3x.nn.Tensor(include_pseudotensors=False)(
                spherical_potential, nodes_spherical
            )

            norms = spherical_norm_last_axis(spherical_updates, max_degree)
            norms = (norms * l_factors[None, None, :, None]).reshape(num_atoms, -1)
            updates = jnp.concatenate([scalar_potential, norms], axis=-1)
            nodes_scalar = Update(d)(nodes_scalar, updates, atom_mask)

            # -- residual prediction --
            accumulate(nodes_scalar)

        # -- close the charge expansion --
        num_structures = Q.shape[0]
        mask = atom_mask.astype(R.dtype)

        energy = e0_i + phi0_i * Q_i

        if quadratic:
            if per_atom_kappa:
                # The softplus closes on the *structure* sum, not per atom.
                # Positivity is a property of the capacitance, and imposing it
                # atom by atom pins every contribution at target/(n_atoms
                # n_sites) -- a few hundredths, where softplus' is ~0.04 and
                # the head's gradients are damped by that factor. Here the
                # operating point is the physical value, where softplus' ~ 1.
                raw = jax.ops.segment_sum(
                    kappa_i * mask, atom_to_structure, num_segments=num_structures
                )
            else:
                raw = self.param("kappa_const", nn.initializers.zeros, ()) * jnp.ones(
                    (num_structures,), dtype=R.dtype
                )

            K_s = jax.nn.softplus(raw + self.kappa_init)

            # spread the structure term back over its own atoms so the return
            # value stays per-atom. The split is arbitrary and exact in the
            # sum, which is all that reaches energy, forces and dE/dq. The
            # clamp keeps padded structures, which own no atoms, out of 0/0.
            counts = jnp.maximum(
                jax.ops.segment_sum(
                    mask, atom_to_structure, num_segments=num_structures
                ),
                1.0,
            )
            energy = energy + 0.5 * (K_s / counts)[atom_to_structure] * Q_i**2

            B_s = jax.ops.segment_sum(
                phi0_i * mask, atom_to_structure, num_segments=num_structures
            )
        else:
            B_s = K_s = None

        energy = energy * mask

        if self.charge_conditioning == "quadratic_film":
            # anharmonic correction on top of the quadratic form. The features
            # feeding the quadratic heads stay charge-free deliberately: FiLMing
            # them would make E0/Phi0/kappa themselves Q-dependent, E(Q) would
            # stop being quadratic, and kappa would quietly stop being d2E/dQ2.
            residual = ChargeConditioning(d)(Q_i, nodes_scalar, atom_mask)
            correction = masked(MLP(features=[d, d, 1]), residual, atom_mask)[..., 0]
            # Zero-initialised gate, so the residual starts at exactly zero and
            # this mode starts identical to "quadratic". Without it the head's
            # final Dense carries its default init, contributing arbitrary
            # curvature in q: measured on a 108-atom slab with lr=True it put
            # d2E/dq2 at -7.95 V/e, a negative capacitance. The gate opens only
            # if the data asks for anharmonicity, which is what makes this a
            # residual rather than a second model.
            gate = self.param("film_residual_gate", nn.initializers.zeros, ())
            energy = energy + gate * correction
            # E(q) is no longer quadratic, so the coefficients no longer
            # describe it and must not be handed out as though they did
            B_s = K_s = None

        return energy, B_s, K_s

    def atoms_to_batch(self, atoms):
        from lorem.batching import to_batch, to_sample

        sample = to_sample(atoms, self.cutoff, energy=False, forces=False, stress=False)
        batch = to_batch([sample], [])

        return jax.tree.map(lambda x: jnp.array(x), batch)

    def dummy_inputs(self):
        from ase.build import bulk

        atoms = bulk("Ar") * [2, 2, 2]
        atoms.info["total_charge"] = 0.0

        return self.atoms_to_batch(atoms)[:-1]

    def energy(self, params, batch):
        sr = batch[1]
        energies = self.apply(
            params,
            batch.atomic_numbers,
            batch.sr,
            batch.nopbc,
            batch.pbc,
            batch.total_charge,
        )
        energies *= sr.atom_mask

        return jnp.sum(energies), energies

    def _energy_and_grads(self, params, batch):
        """One value_and_grad of the energy w.r.t. the whole batch.

        Split out so `LoremQ` can read dE/dq off the same gradient pytree that
        already carries the forces, rather than paying for a second backward
        pass to get it.
        """
        sr = batch[1]

        energy_and_derivatives_fn = jax.value_and_grad(
            self.energy, allow_int=True, has_aux=True, argnums=1
        )
        batch_energy_and_atom_energies, batch_grads = energy_and_derivatives_fn(
            params, batch
        )
        _, energies = batch_energy_and_atom_energies

        energy = (
            jax.ops.segment_sum(energies, sr.atom_to_structure, sr.cell.shape[0])
            * sr.structure_mask
        )
        forces = -batch_grads.sr.positions

        return energy, forces, batch_grads

    def _stress(self, sr, grads):
        return (
            jax.ops.segment_sum(
                jnp.einsum("ia,ib->iab", sr.positions, grads.positions),
                sr.atom_to_structure,
                num_segments=sr.cell.shape[0],
            )
            + jnp.einsum("sAa,sAb->sab", sr.cell, grads.cell)
        ) * sr.structure_mask[:, None, None]

    def predict(self, params, batch, stress=False):
        """Energy, forces and optionally stress -- the plain MLIP contract.

        Charge derivatives live on `LoremQ`. They are meaningless here: this
        class applies `ChargeConditioning` unconditionally and `total_charge`
        defaults to 0, so a model trained where q never varied would otherwise
        report an unconstrained dE/dq alongside energy and forces, with the
        same apparent standing.
        """
        sr = batch[1]
        energy, forces, batch_grads = self._energy_and_grads(params, batch)

        results = {"energy": energy, "forces": forces}

        if stress:
            results["stress"] = self._stress(sr, batch_grads.sr)

        return results


class LoremQ(Lorem):
    """`Lorem` for systems where the total charge is a real, varied input.

    Identical architecture -- it inherits `__call__` untouched, so a `LoremQ`
    checkpoint is weight-compatible with a `Lorem` one. The only difference is
    what `predict` exposes: the derivatives with respect to `total_charge`,
    which are only meaningful when q actually varied during training.

    - `work_function` = dE/dq. Free: it falls out of the same backward pass as
      the forces.
    - `bec_z` = -(A eps0) d2E/(dr dq), the Born effective charge. Costs a
      forward-over-reverse pass, so it stays behind `predict_bec`.
    - `d2Edq2` = d2E/dq2, the inverse frozen-nuclei capacitance up to 1/A. It
      rides along on the `bec_z` pass for free, so it shares that flag.

    In the exactly-quadratic modes E(q) = A + B q + 1/2 K q^2 identically, so
    `predict` takes the analytic route instead: dE/dq is B + K q and d2E/dq2 is
    K, both without autodiff, and `bec_z` is a reverse sweep over B + K q
    rather than a forward-over-reverse pass over the energy.

    This is about the order of the derivative, not about speed. Measured on a
    36-atom Pt slab the two routes cost the same (8.4 vs 8.3 ms), because the
    backbone in these modes is charge-free: the q-tangent through it is
    identically zero and XLA prunes the forward pass down to the readout
    anyway. What the analytic route buys is that Z* comes out as a *first*
    derivative, as razor and the SEBEC formalism take it, rather than as a
    mixed second derivative in float32 -- and that dE/dq and d2E/dq2 are exact
    and free even with `predict_bec` off. `tests/test_conditioning.py` pins the
    two routes against each other; they agree because the identity is exact.

    For `film` and `quadratic_film` E(q) is not quadratic and the derivative is
    the only honest answer, so those keep the autodiff route.
    """

    # off by default: unlike the work function this is not free, so only runs
    # that actually supervise it should pay for it
    predict_bec: bool = False

    def _charge_second_derivatives(self, params, batch):
        """Born effective charges, as the mixed second derivative.

        The charge sets a surface charge density q/A, hence a field q/(A eps0)
        along the slab normal, and an atom with Born effective charge Z* feels
        F = Z* q/(A eps0). So the dimensionless Z* the datasets carry is

            Z* = (A eps0) dF/dq = -(A eps0) d2E/(dr dq)

        Verified against razor's 3-point stencil: a finite-difference dF/dq
        regressed on the `bec_z` label gives slope 1/(A eps0) = 2.19839 for
        A = 82.3108 A^2, intercept ~1e-11, correlation 1.00000.

        Structures in a batch don't interact, so a tangent of ones on the
        per-structure charge vector hands each atom its own structure's
        derivative -- the same argument that makes the work function a single
        backward pass. This one costs a forward-over-reverse pass on top.
        """
        sr = batch[1]

        def first_derivatives(total_charge):
            def energy_of(positions, charge):
                shifted = batch._replace(
                    sr=sr._replace(positions=positions),
                    total_charge=charge,
                )
                return self.energy(params, shifted)[0]

            # one reverse pass for both gradients: jax handles a tuple of
            # argnums in a single backward sweep
            return jax.grad(energy_of, argnums=(0, 1))(sr.positions, total_charge)

        # forward-over-reverse. Linearising BOTH first derivatives in the charge
        # costs one extra tangent component in an already-linearised
        # computation, so d2E/dq2 is free once bec_z is being computed.
        _, (d2E_drdq, d2E_dq2) = jax.jvp(
            first_derivatives,
            (batch.total_charge,),
            (jnp.ones_like(batch.total_charge),),
        )

        bec_z = -d2E_drdq * self._bec_scale(sr) * sr.atom_mask[:, None]
        return bec_z, d2E_dq2 * sr.structure_mask

    @staticmethod
    def _bec_scale(sr):
        """A eps0, per atom -- what turns d2E/(dr dq) into a dimensionless Z*.

        The in-plane cell area only: the out-of-plane vector varies per
        structure and must not enter (pbc = T T F).
        """
        area = jnp.linalg.norm(jnp.cross(sr.cell[:, 0, :], sr.cell[:, 1, :]), axis=-1)
        return (area * EPSILON_0)[sr.atom_to_structure][:, None]

    @property
    def _exactly_quadratic(self):
        """Whether E(q) is a quadratic the model states rather than one autodiff
        has to uncover."""
        return self.charge_conditioning in ("quadratic", "quadratic_const")

    def _predict_quadratic(self, params, batch, stress=False):
        """The analytic route, for the modes where E(q) is exactly quadratic.

        One `jax.vjp` linearises the coefficient fields once; the forces and
        the BECs are then two backward sweeps over that same linearisation,
        seeded on the energy and on dE/dq respectively. dE/dq and d2E/dq2 are
        read straight off B and K.
        """
        sr = batch[1]

        def outputs(positions, cell):
            shifted = batch._replace(
                sr=sr._replace(positions=positions, cell=cell),
            )
            energies, B_s, K_s = self.apply(
                params,
                shifted.atomic_numbers,
                shifted.sr,
                shifted.nopbc,
                shifted.pbc,
                shifted.total_charge,
                method=Lorem.coefficients,
            )
            phi = (B_s + K_s * shifted.total_charge) * sr.structure_mask
            return (jnp.sum(energies), jnp.sum(phi)), (energies, phi, K_s)

        (_, _), vjp_fn, (energies, phi, K_s) = jax.vjp(
            outputs, sr.positions, sr.cell, has_aux=True
        )

        # seed on the energy: the forces, and the cell derivative for stress
        dE_dr, dE_dcell = vjp_fn((1.0, 0.0))

        energy = (
            jax.ops.segment_sum(energies, sr.atom_to_structure, sr.cell.shape[0])
            * sr.structure_mask
        )
        results = {
            "energy": energy,
            "forces": -dE_dr,
            "work_function": phi,
        }

        if self.predict_bec:
            # seed on dE/dq instead: d/dr of the work function is the mixed
            # second derivative, no second-order autodiff involved
            dphi_dr, _ = vjp_fn((0.0, 1.0))
            results["bec_z"] = -dphi_dr * self._bec_scale(sr) * sr.atom_mask[:, None]
            results["d2Edq2"] = K_s * sr.structure_mask

        if stress:
            results["stress"] = self._stress(
                sr, sr._replace(positions=dE_dr, cell=dE_dcell)
            )

        return results

    def predict(self, params, batch, stress=False):
        if self._exactly_quadratic:
            return self._predict_quadratic(params, batch, stress=stress)

        sr = batch[1]
        energy, forces, batch_grads = self._energy_and_grads(params, batch)

        # dE/dq. `energy` sums over structures that do not interact, so the
        # derivative w.r.t. the per-structure total_charge vector is already
        # each structure's own value. The per-species baseline is
        # charge-independent and drops out, so no offset correction applies
        # here (unlike for the energy itself).
        results = {
            "energy": energy,
            "forces": forces,
            "work_function": batch_grads.total_charge * sr.structure_mask,
        }

        if self.predict_bec:
            # both come out of the same forward-over-reverse pass
            bec_z, d2Edq2 = self._charge_second_derivatives(params, batch)
            results["bec_z"] = bec_z
            results["d2Edq2"] = d2Edq2

        if stress:
            results["stress"] = self._stress(sr, batch_grads.sr)

        return results
