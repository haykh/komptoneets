cmake_minimum_required(VERSION 3.16)
project(KKomptoneets)

include(FetchContent)
FetchContent_Declare(
  kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG master
)
FetchContent_MakeAvailable(kokkos)

add_subdirectory(src)