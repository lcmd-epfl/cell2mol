from __future__ import annotations

import numpy as np
from pydantic import Field
from typing_extensions import deprecated
from cell2mol.classes.atom import Atom
from cell2mol.classes.group import Group
from cell2mol.classes.metal import Metal
from cell2mol.classes.specie import Specie
from cell2mol.connectivity import build_adjacency
from cell2mol.utils import config
from cell2mol.operations import get_angle
from cell2mol.elementdata import ElementData
from cell2mol.my_types import HapticType, NOType, OptionalRefList, SubType
import logging

elemdatabase = ElementData()
logger = logging.getLogger(__name__)


class Ligand(Specie):
    model_config = {"arbitrary_types_allowed": True}

    NO_type: NOType | None = None
    # Cross-reference: points to atoms already in self.atoms
    connected_atoms: OptionalRefList[Atom] = None
    connected_idx: list[int] | None = None
    denticity: int | None = None
    # Ownership: groups are children of this ligand
    groups: list[Group] | None = Field(default=None)
    haptic_type: HapticType | None = None
    is_haptic: bool | None = None
    is_nitrosyl: bool | None = None
    is_silylyne: bool | None = None
    # Cross-reference: points to metals in parent Molecule
    metals: OptionalRefList[Metal] = None
    unique_index: int | None = None

    subtype: SubType = Field(default="ligand")

    @classmethod
    @deprecated("Use ligand(**kwargs) with keyword arguments")
    def from_positional(
        cls, labels: list, coord: list, frac_coord: list = None, radii: list = None
    ) -> None:
        return cls(labels=labels, coord=coord, frac_coord=frac_coord, radii=radii)
        # self.evaluate_as_nitrosyl() ### move to the split_complexes function

    def __repr__(self):
        to_print = ""
        to_print += "------------- Cell2mol LIGAND Object --------------\n"
        to_print += Specie.__repr__(self, indirect=True)
        if self.groups is not None:
            to_print += f" Number of Groups             = {len(self.groups)}\n"
        to_print += "---------------------------------------------------\n"
        return to_print

    def get_connected_metals(self, use_bond_info: bool | None = None):
        self.metals = []
        refcell = self.get_parent("reference")
        bond_data = getattr(refcell, "geom_bond_cif", None) if refcell else None
        cov_factor = getattr(self, "cov_factor", config.COV_FACTOR)
        metal_factor = getattr(self, "metal_factor", config.METAL_FACTOR)
        mol = self.get_parent("molecule")

        if use_bond_info is None:
            use_bond_info = config.USE_BOND_INFO

        for met in mol.metals:
            tmplabels = list(self.labels.copy())
            tmpcoord = list(self.coord.copy())
            atom_site_labels = [atom.atom_site_label for atom in self.atoms]
            tmplabels.append(met.label)
            tmpcoord.append(met.coord)
            atom_site_labels.append(met.atom_site_label)
            tmp_adjmat = build_adjacency(
                labels=tmplabels,
                positions=tmpcoord,
                atom_site_labels=atom_site_labels,
                bond_data=bond_data,
                use_bond_info=use_bond_info,
                cov_factor=cov_factor,
                metal_factor=metal_factor,
                metal_only=True,
            )
            if tmp_adjmat is None:
                continue
            else:
                tmp_adjnum = tmp_adjmat.sum(axis=1)
                if any(tmp_adjnum) > 0:
                    self.metals.append(met)
                    logger.debug(
                        "Ligand %s is connected to %s", self.formula, met.label
                    )

    def evaluate_as_nitrosyl(self):
        self.is_nitrosyl = False
        if self.natoms == 2 and "N" in self.labels and "O" in self.labels:
            self.is_nitrosyl = True
            self.get_nitrosyl_geom()
        return self.is_nitrosyl

    def get_nitrosyl_geom(self: object, thres: float = 160) -> str:
        """Get the geometry of a Nitrosyl ligand."""
        # Determines whether the M-N-O angle of a Nitrosyl "ligand" is "Bent" or "Linear"
        # Each case is treated differently
        # return NO_type: "Linear" or "Bent"
        if self.atoms is None:
            self.set_atoms()
        if self.metals is None:
            self.get_connected_metals()

        for idx, a in enumerate(self.atoms):
            if a.label == "N":
                central = a.coord.copy()
            if a.label == "O":
                extreme = a.coord.copy()

        dist = []
        for idx, met in enumerate(self.metals):
            metal = np.array(met.coord)
            dist.append(np.linalg.norm(central - metal))
        tgt = np.argmin(dist)
        metal = self.metals[tgt].coord.copy()
        vector1 = np.subtract(np.array(central), np.array(extreme))
        vector2 = np.subtract(np.array(central), np.array(metal))

        angle = get_angle(vector1, vector2)
        if np.degrees(angle) > float(thres):
            self.NO_type = "Linear"
        else:
            self.NO_type = "Bent"

        return self.NO_type

    def get_connected_idx(self):
        # Remember madjmat should not be computed at the ligand level.
        # Since the metal is not there.
        # Operate at the molecular level.
        # We get the parent molecule, and the indices of the ligand atoms in the molecule
        self.connected_idx = []
        if self.madjnum is None:
            self.set_inherit_adjmatrix("molecule")
        for idx, con in enumerate(self.madjnum):
            if con > 0:
                self.connected_idx.append(idx)
        return self.connected_idx

    def get_connected_atoms(self):
        if self.atoms is None:
            self.set_atoms()
        if self.connected_idx is None:
            self.get_connected_idx()
        self.connected_atoms = []
        for idx, at in enumerate(self.atoms):
            if idx in self.connected_idx and at.mconnec > 0:
                self.connected_atoms.append(at)
            elif idx in self.connected_idx and at.mconnec == 0:
                logger.warning("Atom appears in connected_idx, but has mconnec=0")
        return self.connected_atoms

    def get_denticity(self):
        if self.groups is None:
            return None
        self.denticity = 0
        for g in self.groups:
            self.denticity += g.get_denticity()
        return self.denticity

    def get_hapticity(self):
        if self.groups is None:
            return None
        self.is_haptic = False
        self.haptic_type = []
        for gr in self.groups:
            if gr.is_haptic is None:
                gr.get_hapticity()
            for entry in gr.haptic_type:
                self.haptic_type.append(entry)
        if len(self.haptic_type) > 0:
            self.is_haptic = True

        return self.haptic_type
