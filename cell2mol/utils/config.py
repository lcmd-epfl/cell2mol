from .defaults import (
    DEFAULT_VERSION,
    DEFAULT_CUTOFF,
    DEFAULT_COV_FACTOR,
    DEFAULT_METAL_FACTOR,
)

# Application exit codes; detailed domain errors remain on result models.
ERR_CELL2MOL = 50
ERR_INPUT = 95
ERR_ASE_PARSE = 96
ERR_GENERAL = 97
ERR_TIMEOUT = 98
ERR_MEMORY = 99

# runtime-configurable values
VERSION = DEFAULT_VERSION
CUTOFF = DEFAULT_CUTOFF
COV_FACTOR = DEFAULT_COV_FACTOR
METAL_FACTOR = DEFAULT_METAL_FACTOR

USE_BOND_INFO = False
MAX_METALS = 4
TIMEOUT = 600  # seconds
MAX_MEM_GB = 12  # GB

# Metal oxidation-state GNN fallback for unit-cell charge balancing (see
# cell2mol.charge.balancing._resolve_with_gnn). Off by default.
GNN_METAL_OS = False  # True: use cell2mol/models/metal_ox_gnn.pt to resolve metal OS
GNN_METAL_OS_CHECKPOINT = None  # None: cell2mol/models/metal_ox_gnn.pt
GNN_METAL_OS_MIN_CONFIDENCE = 0.9


def dump():
    """Return runtime configuration as a formatted string."""
    lines = [
        "cell2mol runtime configuration:",
        f"  VERSION         = {VERSION}",
        f"  USE_BOND_INFO   = {USE_BOND_INFO}",
        f"  CUTOFF          = {CUTOFF}",
        f"  COV_FACTOR      = {COV_FACTOR}",
        f"  METAL_FACTOR    = {METAL_FACTOR}",
        f"  MAX METALS      = {MAX_METALS}",
        f"  TIMEOUT LIMIT   = {TIMEOUT} seconds",
        f"  MEMORY LIMIT    = {MAX_MEM_GB} GB",
        f"  GNN METAL OS    = {GNN_METAL_OS} (min confidence {GNN_METAL_OS_MIN_CONFIDENCE}, "
        f"checkpoint {GNN_METAL_OS_CHECKPOINT or 'default'})",
    ]
    return "\n".join(lines)
