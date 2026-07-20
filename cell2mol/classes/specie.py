from __future__ import annotations

from typing import Any, TYPE_CHECKING, cast

import numpy as np
from pydantic import Field
from typing_extensions import deprecated

from cell2mol.charge.protonation_enumerator import enumerate_protonation_states
from cell2mol.charge.charge_state_resolver import enumerate_possible_charge_states
from cell2mol.classes.atom import Atom
from cell2mol.classes.charge_state import ChargeState
from cell2mol.classes.metal import Metal
from cell2mol.classes.protonation import Protonation
from cell2mol.element_utils import (
    get_metal_idxs,
    get_alkali_alkaline_earth_metal_idxs,
    get_post_transition_metal_idxs,
    get_element_count,
    get_radii,
    labels2electrons,
    labels2formula,
)
from cell2mol.connectivity import build_adjacency, get_adjacency_types
from cell2mol.elementdata import ElementData
from cell2mol.my_types import NDArray, RDKitObject, SubType
from cell2mol.operations import compute_centroid, extract_from_list
from cell2mol.utils import BaseModel, config
import logging

if TYPE_CHECKING:
    from cell2mol.classes.ligand import Ligand
    from cell2mol.classes.molecule import Molecule
    from cell2mol.classes.cell import Cell

elemdatabase = ElementData()
logger = logging.getLogger(__name__)


class Specie(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    # Positional arguments
    labels: list[str]
    coord: NDArray
    frac_coord: NDArray | None = None
    radii: NDArray | None = None

    # Optional arguments - parents is a cross-reference to parent Species
    parents: list["Specie | Cell"] = Field(default_factory=list)
    parents_indices: list[list[int]] = Field(default_factory=list)
    cov_factor: float = Field(default=config.COV_FACTOR)
    metal_factor: float = Field(default=config.METAL_FACTOR)

    # Defined in other methods
    adj_types: NDArray | None = None  # TOFIX romaingrx: NDarray not pydantic compatible
    adjmat: NDArray | None = None
    adjnum: NDArray | None = None
    atnums: list[int] | None = None
    atom_site_labels: list[str] | None = None
    atomic_charges: list[int] | NDArray | None = None
    atoms: list[Atom] | None = None
    centroid: NDArray | None = None
    element_count: NDArray | None = None
    frac_centroid: NDArray | None = None
    madjmat: NDArray | None = None
    madjnum: NDArray | None = None
    protonation_states: list[Protonation] | None = None
    # True when the structure *contains* a porphyrin-family N4 macrocycle
    # (containment, not identity: a fused / ring-modified system such as EFISEV
    # is not a porphyrin but still has a porphyrin core, so this is True there).
    has_porphyrin: bool | None = None
    # True when the structure *contains* a fullerene cage (C20, C60, C70, ...),
    # including when embedded in a larger substituted derivative.
    has_fullerene: bool | None = None
    # True when the structure *contains* a closo- or nido-type borane/carborane
    # deltahedral cage (B12H12^2-, o-carborane, a dicarbollide, ...).
    has_borane: bool | None = None
    # Set when protonation-state enumeration deliberately declines to handle a
    # hard cases (expanded k>=5, fused/ring-modified k=4, or a detector-rejected
    # N4 pocket). The string records why, so the charge result can be flagged
    # for later review.
    protonation_warning: str | None = None
    rdkit_obj: RDKitObject | None = Field(default=None)
    smiles: str | None = None
    subtype: SubType | None = None
    totcharge: int | None = None

    charge_state: ChargeState | None = None
    possible_cs: list[ChargeState] | list[int] | None = Field(default=None)
    origin: str | None = None

    # Frozen fields
    type: str = Field(default="specie", frozen=True)

    @property
    def formula(self) -> str:
        return labels2formula(self.labels)

    @property
    def eleccount(self) -> int:
        # Assuming neutral specie (so basically this is the sum of atomic numbers)
        return labels2electrons(self.labels)

    @property
    def natoms(self) -> int:
        return len(self.labels)

    @property
    def iscomplex(self) -> bool:
        """True if the structure contains d- or f-block metals."""
        return bool(get_metal_idxs(self.labels))

    @property
    def is_non_complex_molecule(self):
        """
        Return True if the specie is a non-complex molecule,
        meaning it contains no metals or metal-like elements.
        """
        return (
            self.subtype == "molecule"
            and not self.iscomplex
            and not self.has_ia_iia
            and not self.has_post_transition_metal
        )

    @property
    def has_ia_iia(self) -> bool:
        """True if the structure contains Group 1 or Group 2 metals (excluding H/D)."""
        return bool(get_alkali_alkaline_earth_metal_idxs(self.labels))

    @property
    def has_post_transition_metal(self) -> bool:
        """True if the structure contains post-transition metals only."""
        return (
            not self.iscomplex
            and not self.has_ia_iia
            and bool(get_post_transition_metal_idxs(self.labels))
        )

    @property
    def indices(self) -> list[int]:
        # Indices might be the atom ordering within a given specie.
        # e.g. 1st, 2nd, 3rd atom of a specie.
        return list(range(self.natoms))

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        assert len(self.labels) == len(self.coord)
        if self.frac_coord is not None:
            assert len(self.coord) == len(self.frac_coord)
        if self.radii is not None:
            # Handle both list and scalar radii (scalar can happen during deserialization)
            if isinstance(self.radii, (list, tuple)):
                assert len(self.labels) == len(self.radii)
            # If radii is a scalar and we have multiple atoms, expand it or recompute
            elif len(self.labels) > 1:
                self.radii = np.asarray(get_radii(self.labels))
        else:
            self.radii = np.asarray(get_radii(self.labels))

    @classmethod
    @deprecated("Use specie() with the keyword arguments instead.")
    def from_positional(
        cls,
        labels: list[str],
        coord: NDArray | list[list[float]],
        frac_coord: NDArray | list[list[float]] | None = None,
        radii: NDArray | list[float] | None = None,
    ) -> "Specie":
        return cls(
            labels=labels,
            coord=np.asarray(coord),
            frac_coord=np.asarray(frac_coord) if frac_coord is not None else None,
            radii=np.asarray(radii) if radii is not None else None,
        )

    def set_subtype(self, subtype: SubType):
        self.subtype = subtype

    def set_origin(self, origin: str):
        self.origin = origin

    def add_parent(
        self, parent: "Specie | Cell", indices: list[int], overwrite: bool = True
    ):
        ## associates a parent specie to self. The atom indices of self in parent are given in "indices"
        ## if parent of the same subtype already in self.parent then it is overwritten
        ## this is to avoid having a substructure (e.g. a ligand) in more than one superstructure (e.g. a molecule)

        # 1st-evaluates parent
        append = True
        for idx, p in enumerate(self.parents):
            if p.subtype == parent.subtype:
                if overwrite:
                    self.parents[idx] = parent
                    self.parents_indices[idx] = indices
                append = False
        if append:
            self.parents.append(parent)
            self.parents_indices.append(indices)

        # 2nd-evaluates parents of parent
        if isinstance(parent, Specie):
            for jdx, p2 in enumerate(parent.parents):
                append = True
                for idx, p in enumerate(self.parents):
                    if p.subtype == p2.subtype:
                        if overwrite:
                            self.parents[idx] = p2
                            self.parents_indices[idx] = (
                                parent.get_parent_indices(p2.subtype or "") or []
                            )
                        append = False
                if append:
                    self.parents.append(p2)
                    self.parents_indices.append(
                        parent.get_parent_indices(p2.subtype or "") or []
                    )

    def check_parent(self, subtype: str):
        ## checks if parent of a given subtype exists
        for p in self.parents:
            if p.subtype == subtype:
                return True
        return False

    def get_parent(self, subtype: str):
        ## retrieves parent of a given subtype
        for p in self.parents:
            if p.subtype == subtype:
                return p
        return None

    def get_parent_indices(self, subtype: str):
        ## retrieves parent of a given subtype
        for idx, p in enumerate(self.parents):
            if p.subtype == subtype:
                return self.parents_indices[idx]
        return None

    def get_centroid(self):
        self.centroid = np.asarray(compute_centroid(np.array(self.coord)))
        # If fractional coordinates exists, then also computes their centroid
        if self.frac_coord is not None:
            self.frac_centroid = np.asarray(compute_centroid(np.array(self.frac_coord)))
        return self.centroid

    def set_fractional_coord(self, frac_coord: NDArray | list[list[float]]) -> None:
        assert len(frac_coord) == len(self.coord)
        self.frac_coord = np.asarray(frac_coord)

    def get_atomic_numbers(self):
        if self.atoms is None:
            self.set_atoms()
        self.atnums = []
        for at in self.atoms or []:
            self.atnums.append(at.atnum)
        return self.atnums

    def evaluate_has_porphyrin(self) -> bool:
        """
        Detect whether the structure *contains* a porphyrin/phthalocyanine- or
        corrole/corrin-type N4 macrocycle (applies to both ligands and
        non-complex molecules) and cache the result on self.has_porphyrin.

        Named for containment, not identity: a fused / ring-modified system
        (e.g. EFISEV) is not itself a porphyrin but still *has* a porphyrin
        core, so the flag is True there too.
        """
        from cell2mol.charge.special_cases import is_porphyrin_macrocycle

        has_porphyrin, _ = is_porphyrin_macrocycle(
            self.get_atomic_numbers(), self.adjmat
        )
        self.has_porphyrin = has_porphyrin
        return self.has_porphyrin

    def evaluate_has_fullerene(self) -> bool:
        """
        Detect whether the structure *contains* a fullerene cage (C20, C60,
        C70, ...) purely from connectivity -- including when embedded in a
        larger substituted derivative (applies to both ligands and non-complex
        molecules) and cache the result on self.has_fullerene.
        """
        from cell2mol.charge.special_cases import has_fullerene

        is_cage, _ = has_fullerene(self.get_atomic_numbers(), self.adjmat)
        self.has_fullerene = is_cage
        return self.has_fullerene

    def evaluate_has_borane(self) -> bool:
        """
        Detect whether the structure *contains* a closo- or nido-type
        borane/carborane deltahedral cage (closo-B12H12^2-, o-carborane,
        a dicarbollide, ...) purely from connectivity (applies to both
        ligands and non-complex molecules) and cache the result on
        self.has_borane.
        """
        from cell2mol.charge.special_cases import is_borane_cage

        is_cage, _ = is_borane_cage(self.get_atomic_numbers(), self.adjmat)
        self.has_borane = is_cage
        return self.has_borane

    def set_element_count(self, heavy_only: bool = False):
        self.element_count = get_element_count(self.labels, heavy_only=heavy_only)
        return self.element_count

    def build_adjmatrix(
        self,
        use_bond_info: bool | None = None,
        metal_only: bool = False,
    ):
        refcell = self.get_parent("reference")
        bond_data = getattr(refcell, "geom_bond_cif", None) if refcell else None
        cov_factor = getattr(self, "cov_factor", config.COV_FACTOR)
        metal_factor = getattr(self, "metal_factor", config.METAL_FACTOR)

        if use_bond_info is None:
            use_bond_info = config.USE_BOND_INFO

        adjmat = build_adjacency(
            labels=self.labels,
            positions=self.coord,
            atom_site_labels=self.atom_site_labels,
            bond_data=bond_data,
            use_bond_info=use_bond_info,
            cov_factor=cov_factor,
            metal_factor=metal_factor,
            metal_only=metal_only,
            warn_on_mismatch=False,
            detail=False,
        )

        if adjmat is None:
            if metal_only:
                self.madjmat = None
                self.madjnum = None
                return self.madjmat, self.madjnum
            else:
                self.adjmat = None
                self.adjnum = None
                return self.adjmat, self.adjnum

        adjnum = adjmat.sum(axis=1)

        if metal_only:
            self.madjmat = adjmat
            self.madjnum = adjnum
            return self.madjmat, self.madjnum
        else:
            self.adjmat = adjmat
            self.adjnum = adjnum
            return self.adjmat, self.adjnum

    def set_adj_types(self, use_bond_info: bool | None = None):
        if use_bond_info is None:
            use_bond_info = config.USE_BOND_INFO
        if self.adjmat is None:
            self.build_adjmatrix(use_bond_info=use_bond_info, metal_only=False)
        if self.adjmat is not None:
            self.adj_types = get_adjacency_types(self.labels, self.adjmat)
        return self.adj_types

    def set_adjacency_parameters(self, cov_factor: float, metal_factor: float) -> None:
        # Stores the covalentradii factor and metal factor that were used to generate the molecule
        self.cov_factor = cov_factor
        self.metal_factor = metal_factor

    def get_adjacency_parameters(self) -> tuple[float, float]:
        return self.cov_factor, self.metal_factor

    def set_atoms(
        self,
        atomlist: list[Atom] | None = None,
        create_adjacencies: bool = False,
        atom_site_labels: list[str] | None = None,
        use_bond_info: bool | None = None,
    ):
        if use_bond_info is None:
            use_bond_info = config.USE_BOND_INFO
        if atomlist is not None:
            self.atoms = atomlist.copy()
            for idx, at in enumerate(self.atoms):
                at.add_parent(self, index=idx)
        else:
            self.atoms = []

            metal_idxs = get_metal_idxs(self.labels)
            alkali_alkaline_earth_metal_idxs = get_alkali_alkaline_earth_metal_idxs(
                self.labels
            )

            assert self.radii is not None
            for idx, label in enumerate(self.labels):
                ## For each label in labels, create an atom class object.
                ismetal = (
                    elemdatabase.elementblock[label] == "d"
                    or elemdatabase.elementblock[label] == "f"
                )
                if len(get_alkali_alkaline_earth_metal_idxs([label])) > 0:
                    ismetal = True
                if len(metal_idxs) == 0 and len(alkali_alkaline_earth_metal_idxs) == 0:
                    if len(get_post_transition_metal_idxs([label])) > 0:
                        ismetal = True

                if self.frac_coord is not None:
                    if ismetal:
                        newatom = Metal.from_positional(
                            label,
                            self.coord[idx],
                            self.frac_coord[idx],
                            radii=self.radii[idx],
                        )
                    else:
                        newatom = Atom.from_positional(
                            label,
                            self.coord[idx],
                            self.frac_coord[idx],
                            radii=self.radii[idx],
                        )
                else:
                    if ismetal:
                        newatom = Metal.from_positional(
                            label, self.coord[idx], radii=self.radii[idx]
                        )
                    else:
                        newatom = Atom.from_positional(
                            label, self.coord[idx], radii=self.radii[idx]
                        )
                newatom.add_parent(self, index=idx)
                self.atoms.append(newatom)

        if atom_site_labels is not None:
            self.atom_site_labels = atom_site_labels.copy()
            for idx, at in enumerate(self.atoms):
                at.set_atom_site_label(atom_site_labels[idx])

        if create_adjacencies:
            if self.adjmat is None:
                self.build_adjmatrix(use_bond_info=use_bond_info, metal_only=False)

            if self.madjmat is None:
                self.build_adjmatrix(use_bond_info=use_bond_info, metal_only=True)

            if self.adjmat is not None and self.madjmat is not None:
                assert self.adjnum is not None
                assert self.madjnum is not None
                for idx, at in enumerate(self.atoms):
                    at.set_adjacencies(
                        self.adjmat[idx],
                        self.madjmat[idx],
                        self.adjnum[idx],
                        self.madjnum[idx],
                    )

    def set_inherit_adjmatrix(self, parent_subtype: str):
        exists = self.check_parent(parent_subtype)
        if not exists:
            logger.debug(f"{parent_subtype=} does not exist")
            return None
        parent = cast("Specie", self.get_parent(parent_subtype))
        indices = self.get_parent_indices(parent_subtype)
        assert parent is not None
        assert indices is not None
        if parent.madjnum is None:
            logger.debug(f"{parent_subtype=} does not have madjnum")
            return None
        assert parent.madjmat is not None
        assert parent.adjmat is not None
        assert parent.adjnum is not None

        self.madjmat = np.stack(
            extract_from_list(indices, parent.madjmat.tolist(), dimension=2), axis=0
        )
        self.madjnum = np.stack(
            extract_from_list(indices, parent.madjnum.tolist(), dimension=1), axis=0
        )
        self.adjmat = np.stack(
            extract_from_list(indices, parent.adjmat.tolist(), dimension=2), axis=0
        )
        self.adjnum = np.stack(
            extract_from_list(indices, parent.adjnum.tolist(), dimension=1), axis=0
        )

    def get_protonation_states(self):
        """
        Enumerate protonation states for this Specie.

        Protonation states are only defined for ligands and
        non-complex molecules. For all other species,
        protonation_states is None.
        """
        self.protonation_states = None
        self.protonation_warning = None

        if not (self.subtype == "ligand" or self.is_non_complex_molecule):
            return self.protonation_states

        if self.subtype == "ligand":
            ligand_self = cast("Ligand", self)
            if ligand_self.is_haptic is None:
                ligand_self.get_hapticity()
            if ligand_self.denticity is None:
                ligand_self.get_denticity()
            if ligand_self.is_nitrosyl is None:
                ligand_self.evaluate_as_nitrosyl()

        self.protonation_states = enumerate_protonation_states(self)
        return self.protonation_states

    def get_possible_cs(self):
        """
        Enumerate possible charge states for this Specie.

        Possible charge states are only defined for ligands and
        non-complex molecules. Final charge selection is handled
        later at the cell level.
        """
        if self.possible_cs is not None:
            return self.possible_cs

        # Default behavior
        self.possible_cs = None

        if not (self.subtype == "ligand" or self.is_non_complex_molecule):
            return self.possible_cs

        if self.protonation_states is None:
            self.get_protonation_states()
            # logger.debug(
            #     "Obtained protonation states for %s: %s",
            #     self.formula,
            #     self.protonation_states,
            # )
        logger.debug("Enumerating charge states for %s", self.formula)

        self.possible_cs = enumerate_possible_charge_states(self)
        return self.possible_cs

    def set_charges(
        self,
        totcharge: int | None = None,
        atomic_charges: list[int] | NDArray | None = None,
        smiles: str | None = None,
        rdkit_obj: object = None,
    ) -> None:
        ## Sets total charge
        if totcharge is not None:
            self.totcharge = totcharge
        elif totcharge is None and atomic_charges is not None:
            self.totcharge = int(np.sum(atomic_charges))
        elif totcharge is None and atomic_charges is None:
            self.totcharge = None
        ## Sets atomic charges
        if atomic_charges is not None:
            self.atomic_charges = atomic_charges
            if self.atoms is None:
                self.set_atoms()
            for idx, a in enumerate(self.atoms or []):
                a.set_charge(self.atomic_charges[idx])
        if smiles is not None:
            self.smiles = smiles
        if rdkit_obj is not None:
            self.rdkit_obj = rdkit_obj

    def reset_charge(self):
        self.totcharge = None
        self.atomic_charges = None
        self.smiles = None
        self.rdkit_obj = None
        self.possible_cs = None

        for a in self.atoms or []:
            a.reset_charge()

    def print_xyz(self):
        print(self.natoms)
        print("")
        for idx, label in enumerate(self.labels):
            print(
                "%s  %.6f  %.6f  %.6f"
                % (label, self.coord[idx][0], self.coord[idx][1], self.coord[idx][2])
            )

    ## This defines the sum operation between two species. To be implemented
    def __add__(self, other):
        if not isinstance(other, type(self)):
            return self
        return self

    def __str__(self):
        # This will make print(object) behave like before
        return self.__repr__()

    def __repr__(self, indirect: bool = False):
        to_print = ""
        if not indirect:
            to_print += "------------- Cell2mol SPECIE Object --------------\n"
        to_print += f" Type                         = {self.type}\n"
        if self.subtype is not None:
            to_print += f" Sub-Type                     = {self.subtype}\n"
        to_print += f" Number of Atoms              = {self.natoms}\n"
        to_print += f" Formula                      = {self.formula}\n"
        to_print += f" Covalent Radii Factor        = {self.cov_factor}\n"
        to_print += f" Metal Radii Factor           = {self.metal_factor}\n"
        if self.adjmat is not None:
            to_print += " Has Adjacency Matrix         = YES\n"
        else:
            to_print += " Has Adjacency Matrix         = NO \n"
        if self.totcharge is not None:
            to_print += f" Total Charge                 = {self.totcharge}\n"
        if self.subtype == "molecule" and cast("Molecule", self).spin is not None:
            to_print += (
                f" Spin                         = {cast('Molecule', self).spin}\n"
            )
        if self.smiles is not None:
            to_print += f" Smiles                       = {self.smiles}\n"
        if self.origin is not None:
            to_print += f" Origin                       = {self.origin}\n"
        if not indirect:
            to_print += "---------------------------------------------------\n"
        return to_print
