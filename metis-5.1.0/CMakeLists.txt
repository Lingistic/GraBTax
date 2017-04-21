cmake_minimum_required(VERSION 2.8)
project(METIS)

# Fix GKlib path:
if(NOT WIN32)
	SET(GKLIB_PATH "${${PROJECT_NAME}_SOURCE_DIR}/GKlib" CACHE INTERNAL "Path to GKlib (for METIS)" FORCE)
else()
    set(GKLIB_PATH "GKlib" CACHE PATH "path to GKlib")
endif()--

set(SHARED FALSE CACHE BOOL "build a shared library")

if(MSVC)
  set(METIS_INSTALL FALSE)
else()
  set(METIS_INSTALL TRUE)
endif()

# Configure libmetis library.
if(SHARED)
  set(METIS_LIBRARY_TYPE SHARED)
else()
  set(METIS_LIBRARY_TYPE STATIC)
endif(SHARED)

include(${GKLIB_PATH}/GKlibSystem.cmake)
# Add include directories.
include_directories(${GKLIB_PATH})
include_directories(include)
# Recursively look for CMakeLists.txt in subdirs.
add_subdirectory("include")
add_subdirectory("libmetis")
add_subdirectory("programs")


