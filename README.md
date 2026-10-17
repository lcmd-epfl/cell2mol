cell2mol: Unit Cell to Molecule Interpretation
==============================================

![cell2mol logo](./images/cell2mol_logo.png)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lcmd-epfl/cell2mol/dev?labpath=App.ipynb)

## Contents

* [About](#about)
* [Install](#install)
* [Examples](#examples)
* [Documentation](#documentation)

## About

cell2mol converts atomic coordinates into molecular structures with assigned
connectivity, bond orders, formal charges, and metal oxidation states. It
supports molecular crystals supplied as CIF files and individual molecules or
collections of molecules supplied as XYZ files.

For CIF input, cell2mol builds reference molecules from the listed atom sites
and can reconstruct molecules in the unit cell. Metal complexes are separated
into metal atoms and ligands, with coordinating donor groups identified within
each ligand. Coordination analysis describes how ligands bind to each metal,
including denticity, hapticity, and bridging between metals, and characterizes
metal coordination geometries.

Charge assignment generates candidate structures for ligands and metal-free
molecules, assigns bond orders and atom-level formal charges, and checks their
chemical validity. Certain chemical moieties require careful treatment when assigning bond
orders and formal charges. Candidate species charges and metal oxidation
states are then combined to satisfy the system charge: a neutral unit cell for
CIF processing or a user-supplied total charge for XYZ input.

Results retain the relationships between cells, molecules, ligands, donor groups,
and atoms, together with coordinates, molecular representations, and processing
flags. These structured models can be saved and loaded for further analysis,
quantum chemistry calculations, and machine learning workflows. See the
[documentation guides](#documentation) for the data models and charge-assignment
workflow.

## Install

PyPI publication is planned. For now, clone the repository and install with pip:

```bash
git clone https://github.com/lcmd-epfl/cell2mol.git
cd cell2mol
pip install .
```

Alternatively, use uv from the repository root:

```bash
uv sync
uv run python -m cell2mol --help
```

## Examples

Interpret molecules built from atom sites listed in a CIF:

```bash
python -m cell2mol -i structure.cif -t reference
```

Reconstruct and interpret the full unit cell:

```bash
python -m cell2mol -i structure.cif -t unitcell
```

Interpret an XYZ input with a supplied total charge:

```bash
python -m cell2mol -i molecules.xyz -t molecule --charge 0
```

For an XYZ input containing multiple molecules, `--charge` is the total charge of all molecules in the input. In a uv project, use `uv run python -m cell2mol` with the same arguments.

Show available options:

```bash
python -m cell2mol --help
```

Processing writes JSON models and legacy pickle files, depending on the workflow.

## Documentation

| Guide | Covers |
| --- | --- |
| [Class structure](cell2mol/classes/README.md) | Model inheritance, object relationships, file roles, parent links, and saving/loading results |
| [Charge assignment](cell2mol/charge/README.md) | Candidate generation, bond orders, chemical-family rules, charge balancing, and module roles |
