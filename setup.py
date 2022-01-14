from setuptools import setup
import io

# Read the contents of your README file
from os import path

this_directory = path.abspath(path.dirname(__file__))
with io.open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="cell2mol",
    packages=["cell2mol"],
    version="1.0",
    description="Generator cell object from a cif file",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="svela-bs, rlaplaza, choglass, lcmd-epfl",
    url="https://github.com/lcmd-epfl/cell2mol/",
    classifiers=["Programming Language :: Python :: 3"],
    include_package_data=True,
    package_dir={"cell2mol": "cell2mol"},
    entry_points={"console_scripts": ["cell2mol = cell2mol.c2m_driver:main"]},
)
