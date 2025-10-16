#!/usr/bin/env python
from __future__ import absolute_import

import os
import sys

# from cell2mol import c2m_driver
# from cell2mol import new_c2m_driver
from cell2mol import final_c2m_driver  # noqa: F401

if __package__ == "":
    path = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, path)

if __name__ == "__main__":
    sys.exit()
