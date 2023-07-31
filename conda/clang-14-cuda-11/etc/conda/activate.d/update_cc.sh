export CMAKE_PREFIX_PATH_base="$CMAKE_PREFIX_PATH"
export CMAKE_LIBRARY_PATH_base="$CMAKE_LIBRARY_PATH"
export CC_base="$CC"
export CXX_base="$CXX"
export CPATH_base="$CPATH"
export LIBRARY_PATH_base="$LIBRARY_PATH"

export CC=$(which clang)
export CXX=$(which clang++)

# 1. Ensure Conda environment is searched by preference.
# 2. CUDA requires librt, which is in the sysroot directory.
export CMAKE_PREFIX_PATH="$CONDA_PREFIX:$(x86_64-conda-linux-gnu-gcc -print-sysroot)/usr${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
# CUDA puts some .so files under a "stubs" drectory.
export CMAKE_LIBRARY_PATH="$CONDA_PREFIX/lib/stubs${CMAKE_LIBRARY_PATH:+:$CMAKE_LIBRARY_PATH}"
# TODO: Clang-16 problem with include paths, maybe not necessary for Clang-14.
# clang's default include search path doesn't include `./include`, causing problems finding `omp.h`
export CPATH="$CONDA_PREFIX/include${CPATH:+:$CPATH}"
# TODO: Clang-14 fixes (Clang-16 seemed fine) - remove when upgrading:
# * clang's default include search path doesn't include ./lib, causing problems finding libomp.
export LIBRARY_PATH="$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
# * Add sysroot for librt so CUDA can find it at runtime.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
