from .defaults import (
    DEFAULT_VERSION,
    DEFAULT_CUTOFF,
    DEFAULT_COV_FACTOR,
    DEFAULT_METAL_FACTOR,
)

# Error Codes
ERR_GENERAL = 97
ERR_TIMEOUT = 98
ERR_MEMORY = 99

# runtime-configurable values
VERSION = DEFAULT_VERSION
CUTOFF = DEFAULT_CUTOFF
COV_FACTOR = DEFAULT_COV_FACTOR
METAL_FACTOR = DEFAULT_METAL_FACTOR

USE_BOND_INFO = False
MAX_METALS = 6
TIMEOUT = 300  # seconds
MAX_MEM_GB = 5  # GB


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
    ]
    return "\n".join(lines)
