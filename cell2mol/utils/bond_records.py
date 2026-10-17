"""Upgrade legacy atom-owned Bond graphs to species-local inline records."""

from __future__ import annotations

from typing import Any


def _upgrade(models, atom_bonds, missing_species):
    from cell2mol.classes.atom import Atom
    from cell2mol.classes.species import Species

    for species in models:
        if not isinstance(species, Species) or species.id not in missing_species:
            continue
        atoms = species.atoms or []
        positions = {id(atom): i for i, atom in enumerate(atoms)}
        records = {}
        for atom in atoms:
            for bond in atom_bonds.get(atom.id, []):
                i = positions.get(id(bond.atom1))
                j = positions.get(id(bond.atom2))
                # Ligands contain their internal bonds; the parent molecule
                # also contains metal contacts and other ligand bonds.
                if i is None or j is None:
                    continue
                pair = tuple(sorted((i, j)))
                record = {
                    "atom_indices": list(pair),
                    "order": float(bond.order),
                    "distance": float(bond.distance)
                    if bond.distance is not None
                    else species.make_bond_record(list(pair), bond.order)["distance"],
                }
                if pair in records and records[pair] != record:
                    raise ValueError(f"Conflicting legacy bonds for atom pair {pair}")
                records[pair] = record
        species.bonds = list(records.values())

    for obj in models:
        if isinstance(obj, Atom):
            obj.__dict__.pop("bonds", None)
            obj.__pydantic_fields_set__.discard("bonds")


def upgrade_store_bonds(store: dict[str, Any], registry: dict[str, Any]) -> None:
    """Migrate after references and legacy Bond distances have been resolved."""
    atom_bonds = {
        uid: [registry[ref] for ref in data.get("bonds", [])]
        for uid, data in store.items()
        if data.get("_type") in ("Atom", "Metal")
    }
    missing_species = {uid for uid, data in store.items() if "bonds" not in data}
    _upgrade(list(registry.values()), atom_bonds, missing_species)


def upgrade_pickle_bonds(root: Any) -> None:
    """Migrate a fully unpickled graph; nested objects may have cycles."""
    from cell2mol.classes.atom import Atom
    from cell2mol.utils.pydantic import BaseModel

    models = []
    seen = set()
    pending = [root]
    atom_bonds = {}
    while pending:
        obj = pending.pop()
        if isinstance(obj, BaseModel):
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            models.append(obj)
            pending.extend(
                getattr(obj, name)
                for name in type(obj).model_fields
                if name != "bonds" or name in obj.__dict__
            )
            if isinstance(obj, Atom) and "bonds" in obj.__dict__:
                atom_bonds[obj.id] = obj.__dict__["bonds"]
                pending.extend(atom_bonds[obj.id])
        elif isinstance(obj, (list, tuple)):
            pending.extend(obj)
        elif isinstance(obj, dict):
            pending.extend(obj.values())
    missing_species = {obj.id for obj in models if "bonds" not in obj.__dict__}
    _upgrade(models, atom_bonds, missing_species)
