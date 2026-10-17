# Charge assignment

Assign bond orders and formal charges for ligands and metal-free molecules.
A charge state is a candidate molecular structure with a specific total charge,
bond orders, and atom-level formal charges.

Generate plausible charge states, then select one per species using the total
system charge and metal oxidation states. The target system charge is 0 for a
unitcell and the user-supplied charge for XYZ input.

## Flow

```text
Donor protonation patterns
  → Candidate charges / chemical-family builders
  → Bond orders → repair → validation
  → Select one structure per charge → prune artifacts
  → Balance charges → apply structures to symmetry copies
```

## Modules

| Module | Role |
| --- | --- |
| `protonation_states.py` | Generate protonation patterns for metal-coordinating donor atoms |
| `state_generation.py` | Generate plausible charge states for ligands and metal-free molecules |
| `bond_orders.py` | Assign candidate bond orders with RDKit and modified AC2mol, then repair and validate structures |
| `molecule_builder.py` | Correct artificial charge separation, assemble bonds, and generate SMILES |
| `validation.py` | Check the chemical validity of candidate structures |
| `state_ranking.py` | Select one structure per charge and prune artifact states |
| `aromaticity.py` | Measure aromaticity and identify artifacts in perceived aromatic rings |
| `special_cases/` | Generate charge states using chemical-family and manual rules |
| `oxidation_states.py` | Supply plausible metal oxidation states |
| `balancing.py` | Choose charge distributions that sum to the target system charge |
| `charge_assignment.py` | Apply selected bond orders and atom charges to each species and its symmetry copies |
| `xyz2mol.py` | Provide the modified AC2mol algorithm for bond-order assignment |

## Key rules

- **System charge:** `balance_unitcell_charge` targets 0;
  `balance_molecule_charge` targets the user-supplied `input_charge` for XYZ input.
- **Protonated frame:** species charge = protonated charge − added proton count + ligand donor electrons.
- **Repair and validate candidate structures:** correct artificial charge
  separation and associated bond orders, then check chemical validity.
- **Rank within a charge:** choose one representative structure for each charge
  and prune artifacts; the balancer chooses among surviving charges using the
  target system charge and metal oxidation states.
- **Use chemical-family rules when needed:** apply dedicated builders for
  recognised chemical families.
- **Transfer solved structures:** match corresponding atoms in copies of the
  same species by graph isomorphism, then transfer the selected bond orders and
  atom charges.
