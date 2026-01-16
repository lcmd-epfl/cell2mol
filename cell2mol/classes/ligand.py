from __future__ import annotations

import numpy as np
from pydantic import Field
from typing_extensions import deprecated
from cell2mol.classes.atom import Atom
from cell2mol.classes.group import Group
from cell2mol.classes.metal import Metal
from cell2mol.classes.specie import Specie
from cell2mol.connectivity import build_adjacency, split_species, split_group
from cell2mol.utils import config
from cell2mol.operations import extract_from_list, get_angle
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

    def get_connected_idx(self, debug: int = 0):
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
            self.split_ligand()
        self.denticity = 0
        for g in self.groups:
            self.denticity += g.get_denticity()
        return self.denticity

    def split_ligand(self, use_bond_info: bool | None = None):
        """
        Split a ligand into coordination groups.
        """

        def _is_single_sublist(lst):
            return isinstance(lst, list) and len(lst) == 1 and isinstance(lst[0], list)

        if use_bond_info is None:
            use_bond_info = config.USE_BOND_INFO

        # Reference / bond information
        refcell = self.get_parent("reference")
        bond_data = getattr(refcell, "geom_bond_cif", None) if refcell else None
        cov_factor = getattr(self, "cov_factor", config.COV_FACTOR)

        # Coordination groups
        logger.info("Splitting Ligand %s into groups", self.formula)
        self.groups = []

        # Identify Connected and Unconnected atoms (to the metal)
        if self.connected_idx is None:
            self.get_connected_idx()
        connected_idx = self.connected_idx

        # logger.debug(f"Ligand indices: {self.indices=}")
        # logger.debug(f"{connected_idx=}")

        conn_labels = extract_from_list(connected_idx, self.labels, dimension=1)
        conn_coord = extract_from_list(connected_idx, self.coord, dimension=1)
        if self.frac_coord is not None:
            conn_frac_coord = extract_from_list(
                connected_idx, self.frac_coord, dimension=1
            )
        conn_radii = extract_from_list(connected_idx, self.radii, dimension=1)
        conn_atoms = extract_from_list(connected_idx, self.atoms, dimension=1)
        if self.atom_site_labels is not None:
            conn_atom_site_labels = extract_from_list(
                connected_idx, self.atom_site_labels, dimension=1
            )
        else:
            conn_atom_site_labels = None

        logger.debug(
            "  coordinating atoms: %s %s",
            conn_labels,
            conn_atom_site_labels if conn_atom_site_labels else "",
        )

        blocklist = split_species(
            labels=conn_labels,
            positions=conn_coord,
            radii=conn_radii,
            indices=None,  # rest_indices
            atom_site_labels=conn_atom_site_labels,
            bond_data=bond_data,
            use_bond_info=use_bond_info,
            cov_factor=cov_factor,
            apply_graph=True,
        )

        ## Arranges Groups
        for b in blocklist:
            gr_indices = extract_from_list(b, connected_idx, dimension=1)
            gr_labels = extract_from_list(b, conn_labels, dimension=1)
            gr_coord = extract_from_list(b, conn_coord, dimension=1)
            if self.frac_coord is not None:
                gr_frac_coord = extract_from_list(b, conn_frac_coord, dimension=1)
            gr_radii = extract_from_list(b, conn_radii, dimension=1)
            gr_atoms = extract_from_list(b, conn_atoms, dimension=1)
            if self.atom_site_labels is not None:
                gr_atom_site_labels = extract_from_list(
                    b, conn_atom_site_labels, dimension=1
                )
            else:
                gr_atom_site_labels = None
            # Create Group Object
            if self.frac_coord is not None:
                newgroup = Group.from_positional(
                    gr_labels, gr_coord, gr_frac_coord, radii=gr_radii
                )
            else:
                newgroup = Group.from_positional(gr_labels, gr_coord, radii=gr_radii)

            # For debugging
            newgroup.origin = "split_ligand"
            # Define the ligand as parent of the group. Bottom-Up hierarchy
            newgroup.add_parent(self, indices=gr_indices)
            # Pass the ligand atoms to the groud
            newgroup.set_atoms(
                atomlist=gr_atoms,
                create_adjacencies=False,
                atom_site_labels=gr_atom_site_labels,
                use_bond_info=use_bond_info,
            )
            # Inherit the adjacencies from molecule
            newgroup.set_inherit_adjmatrix("ligand")
            # Associate the Groups with the Metals
            newgroup.get_connected_metals()
            newgroup.get_closest_metal()
            newgroup.get_hapticity()

            (
                newgroup,
                final_group_indices,
                final_ligand_indices,
                group_metals_indices,
            ) = newgroup.check_coordination(use_bond_info=use_bond_info)

            # atoms in new group are connected to different metals
            if not _is_single_sublist(final_group_indices):
                logger.debug(
                    "Enterting SPLIT_GROUP for the GROUP  %s with %s %s",
                    newgroup.formula,
                    final_group_indices,
                    [met.label for met in newgroup.metals],
                )
                for kdx, (conn_idx, metal_idx) in enumerate(
                    zip(final_group_indices, group_metals_indices)
                ):
                    group_metals = [newgroup.metals[midx] for midx in metal_idx]
                    logger.debug(
                        "Enterting SPLIT_GROUP for the GROUP  %s with %s %s",
                        newgroup.formula,
                        final_group_indices,
                        [met.label for met in newgroup.metals],
                    )
                    splitted_groups = split_group(
                        newgroup,
                        conn_idx,
                        final_ligand_indices[kdx],
                        group_metals,
                    )
                    for g in splitted_groups:
                        self.groups.append(g)
            else:
                logger.debug(
                    "GROUP %s with %s connected to %s",
                    newgroup.formula,
                    final_group_indices,
                    [met.label for met in newgroup.metals],
                )
                conn_idx = final_group_indices[0]
                group_metals = [newgroup.metals[kdx] for kdx in group_metals_indices[0]]
                if len(conn_idx) == len(newgroup.atoms):
                    logger.debug("  LIGAND.SPLIT_LIGAND: new group is found")
                    newgroup.get_denticity()
                    # Top-down hierarchy
                    self.groups.append(newgroup)
                elif len(conn_idx) == 0:
                    logger.debug("  LIGAND.SPLIT_LIGAND: no group is found")
                    continue
                else:
                    logger.debug(
                        "  Enterting SPLIT_GROUP for the GROUP %s with %s",
                        newgroup.formula,
                        conn_idx,
                    )
                    splitted_groups = split_group(
                        newgroup,
                        conn_idx,
                        final_ligand_indices[0],
                        group_metals,
                    )
                    for g in splitted_groups:
                        self.groups.append(g)
        logger.info("  Found groups %s", [group.formula for group in self.groups])

        return self.groups

    def get_hapticity(self):
        if self.groups is None:
            self.split_ligand()
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
