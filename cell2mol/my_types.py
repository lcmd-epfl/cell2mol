from typing import Any, Literal

Spin = Literal[0, 1]
HapticType = str
Type = Literal["cell", "specie", "protonation", "charge_state", "atom", "bond"]
SubType = Literal["reference", "unitcell", "molecule", "ligand", "metal", "group"]
NOType = Literal["Linear", "Bent"]
NDArray = Any
ChargeState = object
