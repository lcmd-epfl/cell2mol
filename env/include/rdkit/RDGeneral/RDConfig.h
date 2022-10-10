//
// Copyright (c) 2018 greg Landrum
//
//   @@ All Rights Reserved  @@
//  This file is part of the RDKit.
//  The contents are covered by the terms of the BSD license
//  which is included in the file license.txt, found at the root
//  of the RDKit source tree.
//

// RDKit configuration options
#define RDK_USE_BOOST_SERIALIZATION
#define RDK_USE_BOOST_IOSTREAMS
/* #undef RDK_USE_BOOST_STACKTRACE */

#define RDK_OPTIMIZE_POPCNT
#ifdef RDK_OPTIMIZE_POPCNT
#define USE_BUILTIN_POPCOUNT
#endif

#define RDK_BUILD_THREADSAFE_SSS
#ifdef RDK_BUILD_THREADSAFE_SSS
#define RDK_THREADSAFE_SSS
#endif

#define RDK_TEST_MULTITHREADED

#define RDK_USE_STRICT_ROTOR_DEFINITION

#define RDK_BUILD_DESCRIPTORS3D
#ifdef RDK_BUILD_DESCRIPTORS3D
#define RDK_HAS_EIGEN3
#endif

#define RDK_BUILD_COORDGEN_SUPPORT

#define RDK_BUILD_MAEPARSER_SUPPORT

#define RDK_BUILD_AVALON_SUPPORT

#define RDK_BUILD_INCHI_SUPPORT

#define RDK_BUILD_SLN_SUPPORT

#define RDK_BUILD_CAIRO_SUPPORT

/* #undef RDK_BUILD_FREETYPE_SUPPORT */

#define RDK_USE_URF
