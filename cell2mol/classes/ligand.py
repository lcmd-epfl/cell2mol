from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from pydantic import Field
from typing_extensions import deprecated
from cell2mol.classes.group import Group
from cell2mol.classes.metal import Metal
from cell2mol.classes.species import Species
from cell2mol.connectivity import build_adjacency
from cell2mol.utils import config
from cell2mol.operations import get_angle
from cell2mol.my_types import (
    CoordinationMode,
    NOType,
    OptionalRefList,
    SubType,
    NDArray,
)
import logging

if TYPE_CHECKING:
    from cell2mol.classes.molecule import Molecule

logger = logging.getLogger(__name__)


class Ligand(Species):
    """Ligand with coordinating groups, hapticity, and nitrosyl geometry."""

    NO_type: NOType | None = None
    denticity: int | None = None
    # Keys are metal atom indices in the parent Molecule.
    denticity_by_metal: dict[int, int] | None = None
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
        """Detect special species motifs and classify nitrosyl ligands."""
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

    @property
    def coordination(self) -> dict[int, CoordinationMode]:
        """View existing contacts as kappa donors and eta sites per metal.

        Metal keys are parent Molecule atom indices; donor indices are local
        to this ligand. Metal-specific groups take precedence over merged
        ligand groups. This view neither perceives bonds nor stores new state.
        Unavailable connectivity yields an empty mapping.
        """
        atoms = self.atoms or []
        local_indices = {id(atom): index for index, atom in enumerate(atoms)}
        mol = cast("Molecule | None", self.get_parent("molecule"))
        metals = (
            self.metals if self.metals is not None else (mol.metals if mol else None)
        )
        result: dict[int, CoordinationMode] = {}
        for metal in metals or []:
            metal_index = metal.get_parent_index("molecule")
            if metal_index is None:
                raise ValueError(
                    "Coordination requires metal indices in the parent molecule"
                )
            contacts = {
                index
                for index, atom in enumerate(atoms)
                if metal_index in atom.metal_contact_indices
            }
            if not contacts:
                continue
            eta_sites: set[tuple[int, ...]] = set()
            groups = metal.groups if metal.groups is not None else self.groups
            for group in groups or []:
                if not group.is_haptic:
                    continue
                site = tuple(
                    sorted(
                        {
                            local_indices[id(atom)]
                            for atom in group.atoms or []
                            if id(atom) in local_indices
                            and local_indices[id(atom)] in contacts
                        }
                    )
                )
                # A single contact to a ring is localized, not an eta site.
                if len(site) > 1:
                    eta_sites.add(site)
            eta_atoms = {index for site in eta_sites for index in site}
            result[metal_index] = {
                "kappa_atoms": sorted(contacts - eta_atoms),
                "eta_groups": [list(site) for site in sorted(eta_sites)],
            }
            # logger.debug(
            #     "Ligand %s: metal index=%d, coordination=%s",
            #     self.formula,
            #     metal_index,
            #     result[metal_index],
            # )
        return result

    @property
    def coordinated_metals(self) -> list[int]:
        """Parent Molecule atom indices of metals with ligand contacts."""
        return list(self.coordination)

    @property
    def mu(self) -> int:
        """Number of coordinated metals; terminal ligands have mu=1."""
        return len(self.coordinated_metals)

    @property
    def is_bridging(self) -> bool:
        """Whether this ligand contacts more than one metal."""
        return self.mu > 1

    @property
    def kappa_per_metal(self) -> dict[int, int]:
        """Number of localized donor atoms at each metal, excluding eta atoms."""
        return {
            metal: len(info["kappa_atoms"]) for metal, info in self.coordination.items()
        }

    @property
    def eta_per_metal(self) -> dict[int, list[int]]:
        """Hapticities of all eta sites at each metal, including empty lists."""
        return {
            metal: [len(site) for site in info["eta_groups"]]
            for metal, info in self.coordination.items()
        }

    @property
    def denticity_per_metal(self) -> dict[int, int]:
        """Denticity equals kappa count, never eta atom or site count."""
        return self.kappa_per_metal

    @property
    def donor_atoms(self) -> list[int]:
        """Unique ligand-local kappa donor indices; eta atoms are excluded."""
        return sorted(
            {
                index
                for info in self.coordination.values()
                for index in info["kappa_atoms"]
            }
        )

    @property
    def has_mixed_coordination(self) -> bool:
        """Whether both kappa and eta interactions occur, at any metals."""
        modes = list(self.coordination.values())
        return any(info["kappa_atoms"] for info in modes) and any(
            info["eta_groups"] for info in modes
        )

    def update_denticity(self) -> dict[int, int] | None:
        """Refresh legacy summaries from kappa counts, excluding eta sites.

        The scalar is populated only for a single coordinated metal.
        """
        self.denticity = None
        self.denticity_by_metal = None
        logger.debug("Updating denticity for ligand %s", self.formula)
        if self.groups is None or self.metals is None:
            logger.debug(
                "Skipping denticity: groups available=%s, metals available=%s",
                self.groups is not None,
                self.metals is not None,
            )
            return None
        self.denticity_by_metal = self.denticity_per_metal
        if len(self.denticity_by_metal) == 1:
            self.denticity = next(iter(self.denticity_by_metal.values()))
        logger.debug(
            "Ligand %s: denticity_by_metal=%s, scalar denticity=%s",
            self.formula,
            self.denticity_by_metal,
            self.denticity,
        )
        return self.denticity_by_metal

    def update_hapticity(self):
        """Collect group haptic modes; return None when groups are unavailable."""
        logger.debug("Updating hapticity for ligand %s", self.formula)
        if self.groups is None:
            logger.debug(
                "Skipping hapticity for ligand %s: groups unavailable", self.formula
            )
            return None
        self.is_haptic = False
        self.haptic_type = []
        for group_index, gr in enumerate(self.groups):
            logger.debug(
                "Ligand group %d (%s): is_haptic=%s, mode=%s, recompute=%s",
                group_index,
                gr.formula,
                gr.is_haptic,
                gr.haptic_type,
                gr.is_haptic is None,
            )
            if gr.is_haptic is None:
                gr.update_hapticity()
            logger.debug(
                "Ligand group %d: is_haptic=%s, mode=%s, include=%s",
                group_index,
                gr.is_haptic,
                gr.haptic_type,
                bool(gr.is_haptic and gr.haptic_type is not None),
            )
            if gr.is_haptic and gr.haptic_type is not None:
                self.haptic_type.append(gr.haptic_type)
        if len(self.haptic_type) > 0:
            self.is_haptic = True
        logger.debug(
            "Ligand %s: is_haptic=%s, haptic_type=%s",
            self.formula,
            self.is_haptic,
            self.haptic_type,
        )

        return self.haptic_type

    def __repr__(self, indirect: bool = False):
        """Return species details and the number of coordinating groups."""
        to_print = ""
        to_print += "------------- Cell2mol LIGAND Object --------------\n"
        to_print += Species.__repr__(self, indirect=True)
        if self.groups is not None:
            to_print += f" Number of Groups             = {len(self.groups)}\n"
        to_print += "---------------------------------------------------\n"
        return to_print

    @classmethod
    @deprecated("Use ligand(**kwargs) with keyword arguments")
    def from_positional(
        cls,
        symbols: list[str],
        coord: NDArray | list[list[float]],
        frac_coord: NDArray | list[list[float]] | None = None,
        radii: NDArray | list[float] | None = None,
    ) -> "Ligand":
        return cls(
            symbols=symbols,
            coord=np.asarray(coord),
            frac_coord=np.asarray(frac_coord) if frac_coord is not None else None,
            radii=np.asarray(radii) if radii is not None else None,
        )
