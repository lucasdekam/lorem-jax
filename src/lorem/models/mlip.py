import jax
import jax.numpy as jnp

import e3x
import flax.linen as nn
from jaxpme.batched_mixed import Ewald

from lorem.models.backbone import (
    MLP,
    FiLM,
    Initial,
    RadialCoefficients,
    Update,
    EPSILON_0,
    degree_wise_repeat_last_axis,
    field_from_charge,
    field_harmonics,
    spherical_norm_last_axis,
)
from lorem.models.backbone import (
    _masked as masked,
)
from lorem.transforms import ToBatch, ToSample

COULOMB = 1.0 / (4.0 * jnp.pi * EPSILON_0)  # eV A / e^2, 14.3996


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
    num_message_passing: int = 0
    equivariant_message_passing: bool = True
    initialize_node_features: bool = True
    charge_conditioning: bool = False
    # external_field (V/A, a 3-vector per structure): FiLM on |E|^2, plus a
    # tensor product of the spherical features with E, linear in E
    field_conditioning: bool = False
    # with charge_conditioning instead of field_conditioning: couple the spherical
    # features to the field the charge itself makes, E = Q / (eps0 A) along
    # slab_normal_axis (an electrode charged Q, its counter charge -Q on the +axis
    # side). FiLM stays on Q alone, since |E|^2 is a function of Q, and dE/dQ, the
    # work function, then includes the field path. The axis is explicit, never
    # inferred from the cell.
    field_from_charge: bool = False
    slab_normal_axis: int = 2
    # with either field path: scale the field coupling by a learned scalar, zero at
    # init, so the model starts as the one without it. Ungated, field_from_charge on
    # stepped Pt took over the water's share of dE/dQ from the start, and the
    # long-range route that carries the step never formed (lorem-q-work,
    # experiments/water_pt_abacus, "Diagnostics of LOREM-V's flat step")
    field_gate: bool = False
    # also predict bec_z, the Born effective charges' z column (one forward-mode
    # pass over the force computation): dF/dE_z with field_conditioning; with
    # charge conditioning alone (A eps0) dF/dq, and d2E/dq2 on the same pass
    predict_bec: bool = False
    # with lr: add the scalar charges' own Coulomb energy to the readout, gated,
    # g * q_i phi_i per atom with a learned scalar g ("coulomb_gate", 0 at init; 1 is
    # the bare Coulomb energy). Without it the long-range potentials reach the energy
    # only through the MLP of the updated features, which leaves e.g. a water's
    # interaction with its lateral images out entirely (lorem-q-work,
    # water_Ne_single). The gate starts the model exactly where it is without the
    # term: ungated, the free, net-charged charges make it eV-sized from step 0 and
    # training stalled. The charges are not constrained: their sum is the atoms' net
    # charge, which is not always 0 (an electrode facing a gate plate). The gate is
    # the only new parameter, so checkpoints trained without the term load unchanged.
    coulomb_energy: bool = False

    @property
    def inputs(self):
        # properties read into batch.inputs
        inputs = ()
        if self.charge_conditioning:
            inputs += ("total_charge",)
        if self.field_conditioning:
            inputs += ("external_field",)
        return inputs

    @property
    def to_batch(self):
        return ToBatch

    @property
    def to_sample(self):
        return ToSample

    @nn.compact
    def __call__(
        self,
        Z_i,
        sr,
        nopbc,
        pbc,
        inputs,
    ):
        R = sr.positions
        i = sr.centers
        j = sr.others
        cell = sr.cell
        cell_shifts = sr.cell_shifts
        pair_mask = sr.pair_mask
        atom_mask = sr.atom_mask

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

        if self.field_from_charge and (
            not self.charge_conditioning or self.field_conditioning
        ):
            raise ValueError(
                "field_from_charge needs charge_conditioning and replaces field_conditioning"
            )
        if self.coulomb_energy and not self.lr:
            raise ValueError("coulomb_energy needs lr: it is the long-range charges' energy")
        couple_field = self.field_conditioning or self.field_from_charge
        if self.field_gate and not couple_field:
            raise ValueError("field_gate needs field_conditioning or field_from_charge")

        conditions = []
        if self.charge_conditioning:
            conditions.append(inputs["total_charge"][sr.atom_to_structure])
        if self.field_conditioning:
            E_i = inputs["external_field"][sr.atom_to_structure] * atom_mask[..., None]
            # |E|^2, not |E|: smooth at E=0, so -dE/dE (the dipole) is defined there
            conditions.append(jnp.sum(E_i**2, axis=-1))
        if self.field_from_charge:
            E_s = field_from_charge(inputs["total_charge"], cell, self.slab_normal_axis)
            E_i = E_s[sr.atom_to_structure] * atom_mask[..., None]
        if conditions:
            c_i = jnp.stack(conditions, axis=-1)
            nodes_scalar = FiLM(d)(c_i, nodes_scalar, atom_mask)

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

        if couple_field:
            # -- couple the field into the spherical features --
            # residual, so everything downstream (norms, message passing, LR
            # charges) sees it; linear in E, so it is zero at zero field
            field_coupled = e3x.nn.Tensor(
                max_degree=max_degree, include_pseudotensors=False
            )(nodes_spherical, e3x.nn.Dense(s, use_bias=False)(field_harmonics(E_i)))
            if self.field_gate:
                field_coupled *= self.param("field_coupling_gate", nn.initializers.zeros, ())
            nodes_spherical = nodes_spherical + field_coupled

        # -- mix equivariant information into scalar node features --
        norms = spherical_norm_last_axis(nodes_spherical, max_degree)
        updates = (norms * l_factors[None, None, :, None]).reshape(num_atoms, -1)
        if couple_field:
            # the l=0 output, E . (l=1 features): signed in E, which a norm is not
            updates = jnp.concatenate([updates, field_coupled[:, 0, 0, :]], axis=-1)

        nodes_scalar = Update(d)(nodes_scalar, updates, atom_mask)

        # -- initial prediction --
        energy = masked(MLP(features=[d, d, 1]), nodes_scalar, atom_mask)[..., 0]

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
            energy += masked(MLP(features=[d, d, 1]), nodes_scalar, atom_mask)[..., 0]

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

            if self.coulomb_energy:
                # jaxpme's potentials carry the 1/2 of the pair sum: sum_i q_i phi_i is
                # the Coulomb energy, in e^2/A for charges in e
                gate = self.param("coulomb_gate", nn.initializers.zeros, ())
                energy += gate * COULOMB * scalar_charges[..., 0] * potentials[..., 0]

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
            energy += masked(MLP(features=[d, d, 1]), nodes_scalar, atom_mask)[..., 0]

        return energy

    def atoms_to_batch(self, atoms):
        from lorem.batching import to_batch, to_sample

        sample = to_sample(atoms, self.cutoff, keys=())
        batch = to_batch([sample], [])

        return jax.tree.map(lambda x: jnp.array(x), batch)

    def dummy_inputs(self):
        from ase.build import bulk

        atoms = bulk("Ar") * [2, 2, 2]
        batch = self.atoms_to_batch(atoms)

        mask = batch.sr.structure_mask
        inputs = {}
        if self.charge_conditioning:
            inputs["total_charge"] = jnp.zeros_like(mask, dtype=float)
            inputs["total_charge_mask"] = mask
        if self.field_conditioning:
            inputs["external_field"] = jnp.zeros((*mask.shape, 3), dtype=float)
            inputs["external_field_mask"] = mask

        return batch._replace(inputs=inputs)[:5]

    def energy(self, params, batch):
        sr = batch[1]
        energies = self.apply(
            params,
            batch.atomic_numbers,
            batch.sr,
            batch.nopbc,
            batch.pbc,
            batch.inputs,
        )
        energies *= sr.atom_mask

        return jnp.sum(energies), energies

    def predict(self, params, batch, stress=False):
        sr = batch[1]

        energy_and_derivatives_fn = jax.value_and_grad(
            self.energy, allow_int=True, has_aux=True, argnums=1
        )
        batch_energy_and_atom_energies, batch_grads = energy_and_derivatives_fn(
            params, batch
        )
        _, energies = batch_energy_and_atom_energies

        grads = batch_grads.sr

        energy = (
            jax.ops.segment_sum(energies, sr.atom_to_structure, sr.cell.shape[0])
            * sr.structure_mask
        )

        forces = -grads.positions

        results = {"energy": energy, "forces": forces}

        # derivatives w.r.t. the conditioning inputs, from the same backward pass
        if self.charge_conditioning:
            # dE/dQ: the work function, for a charged electrode
            results["work_function"] = (
                batch_grads.inputs["total_charge"] * sr.structure_mask
            )
            if self.predict_bec and not self.field_conditioning:
                results["bec_z"], results["d2Edq2"] = self._charge_second_derivatives(
                    params, batch
                )
        if self.field_conditioning:
            # mu = -dE/dE, the cell dipole (e A) conjugate to the field
            dipole = -batch_grads.inputs["external_field"] * sr.structure_mask[:, None]
            results["dipole"] = dipole
            results["dipole_z"] = dipole[:, 2]
            if self.predict_bec:
                results["bec_z"] = self._bec_z(params, batch)

        if stress:
            stress = (
                jax.ops.segment_sum(
                    jnp.einsum("ia,ib->iab", sr.positions, grads.positions),
                    sr.atom_to_structure,
                    num_segments=sr.cell.shape[0],
                )
                + jnp.einsum("sAa,sAb->sab", sr.cell, grads.cell)
            ) * sr.structure_mask[:, None, None]
            results["stress"] = stress

        return results

    def _bec_z(self, params, batch):
        # Z*_{i,az} = dF_ia/dE_z = -d2E/(dr_ia dE_z): forward mode along E_z
        # through the force computation, one structure's field per structure
        sr = batch.sr
        field = batch.inputs["external_field"]

        def forces_of(field):
            b = batch._replace(inputs={**batch.inputs, "external_field": field})

            def total_energy(positions):
                return self.energy(params, b._replace(sr=sr._replace(positions=positions)))[
                    0
                ]

            return -jax.grad(total_energy)(sr.positions)

        tangent = jnp.zeros_like(field).at[:, 2].set(1.0)
        _, bec_z = jax.jvp(forces_of, (field,), (tangent,))
        return bec_z * sr.atom_mask[:, None]

    def _charge_second_derivatives(self, params, batch):
        # Z*_{i,az} = (A eps0) dF_ia/dq = -(A eps0) d2E/(dr_ia dq): a charge q sets the
        # field q/(A eps0), so this is the dimensionless Born charge that dF/dE_z is
        # under field_conditioning, and razor's bec_z. d2E/dq2 comes on the same
        # forward-over-reverse pass, one tangent per structure's charge.
        sr = batch.sr

        def first_derivatives(charge):
            def total_energy(positions, charge):
                b = batch._replace(
                    sr=sr._replace(positions=positions),
                    inputs={**batch.inputs, "total_charge": charge},
                )
                return self.energy(params, b)[0]

            return jax.grad(total_energy, argnums=(0, 1))(sr.positions, charge)

        charge = batch.inputs["total_charge"]
        _, (d2E_drdq, d2E_dq2) = jax.jvp(
            first_derivatives, (charge,), (jnp.ones_like(charge),)
        )
        area = jnp.linalg.norm(jnp.cross(sr.cell[:, 0], sr.cell[:, 1]), axis=-1)
        scale = (area * EPSILON_0)[sr.atom_to_structure][:, None]
        return -d2E_drdq * scale * sr.atom_mask[:, None], d2E_dq2 * sr.structure_mask
