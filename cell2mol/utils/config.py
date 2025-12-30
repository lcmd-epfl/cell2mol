from .defaults import (
    DEFAULT_VERSION,
    DEFAULT_CUTOFF,
    DEFAULT_COV_FACTOR,
    DEFAULT_METAL_FACTOR,
)

# runtime-configurable values
VERSION = DEFAULT_VERSION
CUTOFF = DEFAULT_CUTOFF
COV_FACTOR = DEFAULT_COV_FACTOR
METAL_FACTOR = DEFAULT_METAL_FACTOR

USE_BOND_INFO = False


def dump():
    """Return runtime configuration as a formatted string."""
    lines = [
        "cell2mol runtime configuration:",
        f"  VERSION         = {VERSION}",
        f"  USE_BOND_INFO   = {USE_BOND_INFO}",
        f"  CUTOFF          = {CUTOFF}",
        f"  COV_FACTOR      = {COV_FACTOR}",
        f"  METAL_FACTOR    = {METAL_FACTOR}",
    ]
    return "\n".join(lines)
