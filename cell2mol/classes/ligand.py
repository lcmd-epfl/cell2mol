from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from pydantic import Field
from typing_extensions import deprecated
from cell2mol.classes.group import Group
from cell2mol.classes.metal import Metal
from cell2mol.classes.specie import Specie
from cell2mol.connectivity import build_adjacency
from cell2mol.utils import config
from cell2mol.operations import get_angle
from cell2mol.my_types import NOType, OptionalRefList, SubType, NDArray
import logging

if TYPE_CHECKING:
    from cell2mol.classes.molecule import Molecule

logger = logging.getLogger(__name__)


class Ligand(Specie):
    """Ligand with coordinating groups, hapticity, and nitrosyl geometry."""

    NO_type: NOType | None = None
    denticity: int | None = None
    # Groups belong to this ligand; metals belong to the parent molecule.
    groups: list[Group] | None = Field(default=None)
    haptic_type: list[str] | None = None
    is_haptic: bool | None = None
    is_nitrosyl: bool | None = None
    metals: OptionalRefList[Metal] = None
    unique_index: int | None = None

    subtype: SubType | None = Field(default="ligand")

    def determine_connected_metals(self, use_bond_info: bool | None = None) -> None:
        """Rebuild the bonded-metal list using ligand–metal adjacency."""
        self.metals = []
        refcell = self.get_parent("reference")
        bond_data = getattr(refcell, "geom_bond_cif", None) if refcell else None
        cov_factor = self.cov_factor
        metal_factor = self.metal_factor
        mol = cast("Molecule", self.get_parent("molecule"))

        if use_bond_info is None:
            use_bond_info = config.USE_BOND_INFO

        for met in mol.metals or []:
            tmplabels = list(self.labels.copy())
            tmpcoord = list(self.coord.copy())
            atom_site_labels = [atom.atom_site_label for atom in self.atoms or []]
            tmplabels.append(met.label)
            tmpcoord.append(met.coord)
            atom_site_labels.append(met.atom_site_label)
            tmp_adjmat = build_adjacency(
                labels=tmplabels,
                positions=np.asarray(tmpcoord),
                atom_site_labels=cast("list[str]", atom_site_labels)
                if all(lbl is not None for lbl in atom_site_labels)
                else None,
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

    def detect_special_moieties(self) -> None:
        """Detect special specie motifs and classify nitrosyl ligands."""
        super().detect_special_moieties()

        self.detect_nitrosyl()
        if self.is_nitrosyl:
            logger.debug("Ligand %s is a nitrosyl", self.formula)

    def detect_nitrosyl(self) -> bool:
        """Detect an NO ligand and compute its geometry when present."""
        self.is_nitrosyl = False
        if self.natoms == 2 and "N" in self.labels and "O" in self.labels:
            self.is_nitrosyl = True
            self.compute_nitrosyl_geometry()
        return self.is_nitrosyl

    def compute_nitrosyl_geometry(self, angle_threshold: float = 160) -> NOType:
        """Classify the nearest metal's M–N–O angle using a threshold in degrees."""
        if self.atoms is None:
            self.set_atoms()
        if self.metals is None:
            self.determine_connected_metals()

        central = None
        extreme = None
        for a in self.atoms or []:
            if a.label == "N":
                central = a.coord.copy()
            if a.label == "O":
                extreme = a.coord.copy()
        assert central is not None
        assert extreme is not None

        metals = self.metals or []
        dist = []
        for met in metals:
            metal = np.array(met.coord)
            dist.append(np.linalg.norm(central - metal))
        tgt = int(np.argmin(dist))
        metal = metals[tgt].coord.copy()
        vector1 = np.subtract(np.array(central), np.array(extreme))
        vector2 = np.subtract(np.array(central), np.array(metal))

        angle = get_angle(vector1, vector2)
        if np.degrees(angle) > float(angle_threshold):
            self.NO_type = "Linear"
        else:
            self.NO_type = "Bent"

        return self.NO_type

    def update_denticity(self):
        """Sum group denticities; return None when groups are unavailable."""
        if self.groups is None:
            return None
        self.denticity = 0
        for g in self.groups:
            self.denticity += g.update_denticity()
        return self.denticity

    def update_hapticity(self):
        """Collect group haptic modes; return None when groups are unavailable."""
        if self.groups is None:
            return None
        self.is_haptic = False
        self.haptic_type = []
        for gr in self.groups:
            if gr.is_haptic is None:
                gr.update_hapticity()
            if gr.is_haptic and gr.haptic_type is not None:
                self.haptic_type.append(gr.haptic_type)
        if len(self.haptic_type) > 0:
            self.is_haptic = True

        return self.haptic_type

    def __repr__(self, indirect: bool = False):
        """Return specie details and the number of coordinating groups."""
        to_print = ""
        to_print += "------------- Cell2mol LIGAND Object --------------\n"
        to_print += Specie.__repr__(self, indirect=True)
        if self.groups is not None:
            to_print += f" Number of Groups             = {len(self.groups)}\n"
        to_print += "---------------------------------------------------\n"
        return to_print

    @classmethod
    @deprecated("Use ligand(**kwargs) with keyword arguments")
    def from_positional(
        cls,
        labels: list[str],
        coord: NDArray | list[list[float]],
        frac_coord: NDArray | list[list[float]] | None = None,
        radii: NDArray | list[float] | None = None,
    ) -> "Ligand":
        return cls(
            labels=labels,
            coord=np.asarray(coord),
            frac_coord=np.asarray(frac_coord) if frac_coord is not None else None,
            radii=np.asarray(radii) if radii is not None else None,
        )
