# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/Users/seanburdges/Desktop/final kel/build_plugin/_deps/rtneural-src")
  file(MAKE_DIRECTORY "/Users/seanburdges/Desktop/final kel/build_plugin/_deps/rtneural-src")
endif()
file(MAKE_DIRECTORY
  "/Users/seanburdges/Desktop/final kel/build_plugin/_deps/rtneural-build"
  "/Users/seanburdges/Desktop/final kel/build_plugin/_deps/rtneural-subbuild/rtneural-populate-prefix"
  "/Users/seanburdges/Desktop/final kel/build_plugin/_deps/rtneural-subbuild/rtneural-populate-prefix/tmp"
  "/Users/seanburdges/Desktop/final kel/build_plugin/_deps/rtneural-subbuild/rtneural-populate-prefix/src/rtneural-populate-stamp"
  "/Users/seanburdges/Desktop/final kel/build_plugin/_deps/rtneural-subbuild/rtneural-populate-prefix/src"
  "/Users/seanburdges/Desktop/final kel/build_plugin/_deps/rtneural-subbuild/rtneural-populate-prefix/src/rtneural-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/seanburdges/Desktop/final kel/build_plugin/_deps/rtneural-subbuild/rtneural-populate-prefix/src/rtneural-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/seanburdges/Desktop/final kel/build_plugin/_deps/rtneural-subbuild/rtneural-populate-prefix/src/rtneural-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
