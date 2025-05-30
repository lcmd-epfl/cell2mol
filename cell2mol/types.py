from typing import Any, Literal

Spin = Literal[0, 1]
HapticType = str
Type = Literal["cell", "specie", "bond", "protonation", "charge_state"]
SubType = Literal["specie", "reference", "molecule", "cell", "group", "ligand"]
NOType = Literal["Linear", "Bent"]
NDArray = Any
ChargeState = object
