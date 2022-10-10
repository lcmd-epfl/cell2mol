# Compute installation prefix relative to this file
get_filename_component(_prefix "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

# Report other info
set(RDKit_INCLUDE_DIRS "${_prefix}/include/rdkit")

set(RDKit_HAS_PYTHON_WRAPPERS ON)
set(RDKit_HAS_COMPRESSED_SUPPLIERS OFF)
set(RDKit_HAS_CAIRO_SUPPORT ON)
set(RDKit_HAS_FREETYPE_SUPPORT ON)
set(RDKit_HAS_INCHI_SUPPORT ON)
set(RDKit_HAS_AVALON_SUPPORT ON)
set(RDKit_HAS_THREADSAFE_SSS ON)
set(RDKit_HAS_SLN_SUPPORT ON)
set(RDKit_HAS_DESCRIPTORS3D ON)
set(RDKit_HAS_FREESASA_SUPPORT ON)
set(RDKit_HAS_COORDGEN_SUPPORT ON)
set(RDKit_HAS_MAEPARSER_SUPPORT ON)
set(RDKit_HAS_MOLINTERCHANGE_SUPPORT ON)
set(RDKit_HAS_YAEHMOP_SUPPORT ON)
set(RDKit_HAS_STRUCTCHECKER_SUPPORT OFF)
set(RDKit_HAS_URF_SUPPORT ON)

set(RDKit_USE_OPTIMIZED_POPCNT ON)
set(RDKit_USE_STRICT_ROTOR_DEFINITION ON)
set(RDKit_USE_BOOST_VERSION 1.74.0)
set(RDKit_USE_BOOST_SERIALIZATION ON)
set(RDKit_USE_BOOST_REGEX OFF)
set(RDKit_USE_BOOST_IOSTREAMS ON)

# Find the RDKit dependencies
include(CMakeFindDependencyMacro)

find_dependency(Threads REQUIRED)

find_dependency(Boost 1.74.0 REQUIRED)

if(RDKit_HAS_DESCRIPTORS3D)
    find_dependency(Eigen3 REQUIRED NO_MODULE)
endif()

# Import the targets
include("${CMAKE_CURRENT_LIST_DIR}/rdkit-targets.cmake")
