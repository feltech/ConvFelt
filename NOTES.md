# Notes

Attempting to use Conda.

Starting with clang-16 ... but see later - downgraded to clang-14, so some of these earlier steps
may not be necessary

### 1
>   Expecting to find librt for libcudart_static, but didn't find it.
> Call Stack (most recent call first):
> cmake/FindcuBLAS.cmake:20 (find_package)
> src/blas/backends/cublas/CMakeLists.txt:22 (find_package)
1. Ensure Conda environment is searched by preference.
2. CUDA requires librt, which is in the sysroot directory.
```
export CMAKE_PREFIX_PATH="$CONDA_PREFIX:$CONDA_PREFIX/$(clang++ -print-target-triple)/sysroot/usr:$CMAKE_PREFIX_PATH"
```
but once clang downgraded to 14 (see later), the target triple no longer matches, so
```
export CMAKE_PREFIX_PATH="$CONDA_PREFIX:$(x86_64-conda-linux-gnu-gcc -print-sysroot)/usr${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
```


### 2
> -- Found CUDA: /opt/conda/envs/sycl (found suitable version "12.2", minimum required is "10.0")
> CMake Error at /snap/clion/248/bin/cmake/linux/x64/share/cmake-3.26/Modules/FindPackageHandleStandardArgs.cmake:230 (message):
> Could NOT find cuBLAS (missing: CUDA_CUDA_LIBRARY)
> Call Stack (most recent call first):
> /snap/clion/248/bin/cmake/linux/x64/share/cmake-3.26/Modules/FindPackageHandleStandardArgs.cmake:600 (_FPHSA_FAILURE_MESSAGE)
> cmake/FindcuBLAS.cmake:40 (find_package_handle_standard_args)
> src/blas/backends/cublas/CMakeLists.txt:23 (find_package)

CUDA puts some .so files under a "stubs" drectory.
```
export CMAKE_LIBRARY_PATH="$CONDA_PREFIX/lib/stubs:$CMAKE_LIBRARY_PATH"
```

### 3

> Could NOT find OpenMP_C (missing: OpenMP_C_FLAGS OpenMP_C_LIB_NAMES)

OpenMP `omp.h` gets installed in an odd location, as well as the top-level `./include`, neither of
which are in the default include search path for clang.

Temporary fix until https://github.com/conda-forge/openmp-feedstock/issues/24 (my comment lower 
down):
```sh
export CPATH="$CONDA_PREFIX/include:$CPATH"
```

### 4
```
../shared_ptr_base.h:196:22: error: use of undeclared identifier 'noinline'; did you mean 'inline'?
__attribute__((__noinline__))
```
Because https://github.com/llvm/llvm-project/issues/57544#issuecomment-1238812567

Actual fix available in clang 17, but clang 17 not yet in conda.

Must downgrade to libstdc++ 11.3, which in conda means also downgrading clang to 14.

Though once downgraded, building OpenSYCL works, but then oneMKL

> .../clang/14.0.0/include/__clang_cuda_texture_intrinsics.h:696:13: error: no template named 'texture'
            texture<__DataT, __TexT, cudaReadModeNormalizedFloat> __handle,

Presumably too new CUDA - downgrade to 11.5 seems to fix it.

### 5

> lib/libOpenImageIO_Util.so.2.3.7: undefined reference to `boost::filesystem::path::operator/=(boost::filesystem::path const&)'

Presumably because clang-14 is detected as not supporting std::filesystem so OIIO tries to use boost
but has some incompatibility  - must downgrade boost to 1.71.

### 6

Assertion with `cudaCreate : CUBLAS_STATUS_NOT_INITIALIZED`.

Downgrade libcublas to same major version as cuda (frustrating conda allowed incompatible versions 
- seems to assume backward compatibility).


## Older:
* Ubuntu 20.04
* C++20 coroutines, which requires `libc++` for clang support, which brings a lot of trouble (see
  below).
* Salient CMake config, assumes dependencies installed under `dist` in build directory.
  > -DhipSYCL_ROOT=/path/to/hipSYCL/cmake-build-release-clang-14/dist
  > -DSYCL_ROOT=/path/to/hipSYCL/cmake-build-release-clang-14/dist
  > -DoneMKL_ROOT=/path/to/oneMKL/cmake-build-release-clang-14/dist
  > -DCPM_Felt_SOURCE=/path/to/Felt
  > -DCONVFELT_CONAN_PROFILE=clang-14
* `Felt` patched locally [TODO(DF): push relevant changes].
* `hipSyCL`
  - Tested target `cuda.integrated-multipass:sm_70`.
  - Attempting to include as a subproject (e.g. CPM package) is tricky. It must be installed to
    give the CMake helper function. But it must also be patched to allow `libc++`.
  - Patched to
    * Build clang plugin with `libstdc++` (and rest with `libc++`).
    * Add a `sycl-config.cmake` so oneMKL can find it.
  - Hence CMake config
    > -DLLVM_DIR=/usr/lib/llvm-14/cmake -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.7
    > -DCMAKE_CXX_FLAGS=-stdlib=libc++ -DHIPSYCL_CLANG_PLUGIN_FORCE_LIBSTDCPP=ON
    > -DHIPSYCL_ADD_SYCL_CMAKE_CONFIG=ON
  - Back months later: attempting to use newly installed `clang-17`, but
    > Could NOT find OpenMP_C (missing: OpenMP_C_FLAGS OpenMP_C_LIB_NAMES)

    So installed `libomp5-17 libomp-17-dev` and CMake configure works!
  - But then compiler error
    > include/hipSYCL/compiler/Frontend.hpp:38:10: fatal error: 'clang/AST/ASTContext.h' file not found

    So installed `libclang-17-dev`
  - But then compiler errors
    > error: 'getGlobalList' is a private member of 'llvm::Module
  
    So replaced occurences with `globals()`, which is public.
  
  - But then linker errors
    > /usr/bin/ld: src/compiler/llvm-to-backend/libllvm-to-backend.so: undefined reference to `llvm::PassBuilder::PassBuilder(llvm::TargetMachine*, llvm::PipelineTuningOptions, std::__1::optional<llvm::PGOOptions>, llvm::PassInstrumentationCallbacks*)'
    /usr/bin/ld: src/compiler/llvm-to-backend/libllvm-to-backend.so: undefined reference to `llvm::Linker::linkModules(llvm::Module&, std::__1::unique_ptr<llvm::Module, std::__1::default_delete<llvm::Module> >, unsigned int, std::__1::function<void (llvm::Module&, llvm::StringSet<llvm::MallocAllocator> const&)>)'
    /usr/bin/ld: src/compiler/llvm-to-backend/libllvm-to-ptx.so: undefined reference to `llvm::MemoryBuffer::getFile(llvm::Twine const&, bool, bool, bool, std::__1::optional<llvm::Align>)'
    /usr/bin/ld: src/compiler/llvm-to-backend/libllvm-to-ptx.so: undefined reference to `llvm::WriteBitcodeToFile(llvm::Module const&, llvm::raw_ostream&, bool, llvm::ModuleSummaryIndex const*, bool, std::__1::array<unsigned int, 5ul>*)'
    /usr/bin/ld: src/compiler/llvm-to-backend/libllvm-to-ptx.so: undefined reference to `llvm::sys::ExecuteAndWait(llvm::StringRef, llvm::ArrayRef<llvm::StringRef>, std::__1::optional<llvm::ArrayRef<llvm::StringRef> >, llvm::ArrayRef<std::__1::optional<llvm::StringRef> >, unsigned int, unsigned int, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >*, bool*, std::__1::optional<llvm::sys::ProcessStatistics>*, llvm::BitVector*)'
    /usr/bin/ld: src/compiler/llvm-to-backend/libllvm-to-backend.so: undefined reference to `llvm::GlobalVariable::GlobalVariable(llvm::Module&, llvm::Type*, bool, llvm::GlobalValue::LinkageTypes, llvm::Constant*, llvm::Twine const&, llvm::GlobalVariable*, llvm::GlobalValue::ThreadLocalMode, std::__1::optional<unsigned int>, bool)'
    clang++-17: error: linker command failed with exit code 1 (use -v to see invocation)

    * So installed `libclang-cpp17-dev` - nope!, uninstalled.
    * So installed `libclang-rt-17-dev` - nope! already installed
    * So removed `-DCMAKE_CXX_FLAGS=-stdlib=libc++ -DHIPSYCL_CLANG_PLUGIN_FORCE_LIBSTDCPP=ON` from
      CMake flags, that worked!  But there was a reason for them...
 
  - Attempting to build ConvFelt with then gives
    > CMake Error at conanbuildinfo.cmake:584 (message):
      Incorrect 'clang', is not the one detected by CMake: 'GNU'
    
    So added CC and CXX env vars to Conan profile, such that they point to Clang.

* `clang-14` compiler using `libc++`
    - `libstdc++` is used by hipSyCL apt package. Attempting to use `libc++` whilst lnking to that
      gives
      > `undefined reference to ``hipsycl::rt::execution_hints::overwrite_with(
      std::__1::shared_ptr<hipsycl::rt::execution_hint>)'`

      So we must build (patched, see above) hipSyCL.
    - `gcc-10`'s `libstdc++` works with hipSyCL apt package, but clang does not support its
      coroutine library. We can trick the `-fcoroutines` vs `-fcoroutine-ts` error by `#define`ing
      `__cpp_impl_coroutine=201902L`, but that hack just reveals deeper issues, i.e.
      > error: std::experimental::coroutine_traits type was not found; include
      > <experimental/coroutine> before defining a coroutine
    - `gcc-11`'s `libstdc++` is incompatible, we get the following issue:
      https://github.com/ROCmSoftwarePlatform/rocBLAS/issues/1191. The hipSyCL apt packages bundle
      clang-11 headers/lib. We can patch the headers using the fix found in the above link. However, we
      then run into the same coroutine issues as `gcc-10`.
    - Attempting to use `gcc-11` directly fails during hipSyCL device compilation with
      > `clang-11: error: unknown argument: '-fcoroutines'`
* `libomp14-dev` apt package
* `cuda-11.7` apt package
* Intel `oneAPI` apt package.
* `oneMKL` built from CMake source with `libc++`, CUDA and matching hipSYCL targets. Hence CMake
  config:
  > -DCMAKE_VERBOSE_MAKEFILE=ON -DENABLE_CUBLAS_BACKEND=True -DENABLE_MKLCPU_BACKEND=OFF
  -DENABLE_MKLGPU_BACKEND=OFF -DTARGET_DOMAINS=blas -DONEMKL_SYCL_IMPLEMENTATION=hipSYCL
  -DCUDA_ROOT=/usr/local/cuda-11.7
  -DhipSYCL_ROOT=/home/dave/workspace/hipSYCL/cmake-build-release-dpc-trunk/dist
  -DHIPSYCL_TARGETS=cuda.integrated-multipass:sm_70 -DCMAKE_CXX_FLAGS="--target=x86_64-linux-gnu
  -stdlib=libc++" -DBUILD_FUNCTIONAL_TESTS=OFF -DBUILD_EXAMPLES=OFF -DCMAKE_FIND_DEBUG_MODE=ON
* Conan profile:
    ```
    [settings]
    os=Linux
    os_build=Linux
    arch=x86_64
    arch_build=x86_64
    compiler=clang
    compiler.version=14
    compiler.libcxx=libc++
    build_type=Release
    [options]
    [build_requires]
    [env]
    ```
* Attempts to build Intel LLVM (DPC++).
  - cmake 3.16 switch to 3.24 caused complete rebuild - so there is a fundamental difference
  - After building. attempting to use for building oneMKL results in
    >   Expecting to find librt for libcudart_static, but didn't find it.
    > Call Stack (most recent call first):
    > cmake/FindcuBLAS.cmake:20 (find_package)
    > src/blas/backends/cublas/CMakeLists.txt:22 (find_package)

  - Tweaking the build flags to use libstdc++ everywhere...
    > python $DPCPP_HOME/llvm/buildbot/configure.py --cuda -o $DPCPP_HOME/build --cmake-gen Ninja
    --llvm-external-projects openmp --cmake-opt='-DLLVM_INSTALL_TOOLCHAIN_ONLY=OFF'
    --cmake-opt='-DLLVM_ENABLE_RUNTIMES=libcxx;libunwind;libcxxabi'
    --cmake-opt='-DLIBCXX_CXX_ABI=libstdc++'
    --cmake-opt='-DLIBCXX_CXX_ABI_INCLUDE_PATHS=/usr/include/x86_64-linux-gnu/c++/9'
    --cmake-opt='-DLIBCXX_CXX_ABI_LIBRARY_PATH=/usr/lib/gcc/x86_64-linux-gnu/9'
    --cmake-opt='-DCLANG_DEFAULT_CXX_STDLIB=libstdc++' --cmake-opt='-DCLANG_DEFAULT_RTLIB=libgcc'
    --cmake-opt='-DLIBCXX_USE_COMPILER_RT=OFF' --cmake-opt='-DLIBCXX_ENABLE_ABI_LINKER_SCRIPT=OFF'
    --cmake-opt='-DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR:BOOL=OFF'
    --cmake-opt='-DCMAKE_FIND_DEBUG_MODE=OFF' --cmake-opt='-DCMAKE_VERBOSE_MAKEFILE=ON'
    --cmake-opt='-DCMAKE_CXX_COMPILER_LAUNCHER=ccache' --cmake-opt='-DLLVM_INCLUDE_RUNTIMES=ON'

  - Generated CMake command by `configure.py`
    > cmake -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLVM_ENABLE_ASSERTIONS=ON
    > -DLLVM_TARGETS_TO_BUILD=X86;NVPTX
    > -DLLVM_EXTERNAL_PROJECTS=sycl;llvm-spirv;opencl;libdevice;xpti;xptifw;openmp
    > -DLLVM_EXTERNAL_SYCL_SOURCE_DIR=/home/dave/workspace/sycl_workspace/llvm/sycl
    > -DLLVM_EXTERNAL_LLVM_SPIRV_SOURCE_DIR=/home/dave/workspace/sycl_workspace/llvm/llvm-spirv
    > -DLLVM_EXTERNAL_XPTI_SOURCE_DIR=/home/dave/workspace/sycl_workspace/llvm/xpti
    > -DXPTI_SOURCE_DIR=/home/dave/workspace/sycl_workspace/llvm/xpti
    > -DLLVM_EXTERNAL_XPTIFW_SOURCE_DIR=/home/dave/workspace/sycl_workspace/llvm/xptifw
    > -DLLVM_EXTERNAL_LIBDEVICE_SOURCE_DIR=/home/dave/workspace/sycl_workspace/llvm/libdevice
    > -DLLVM_ENABLE_PROJECTS=clang;sycl;llvm-spirv;opencl;libdevice;xpti;xptifw;openmp;libclc
    > -DLIBCLC_TARGETS_TO_BUILD=nvptx64--;nvptx64--nvidiacl
    > -DLIBCLC_GENERATE_REMANGLED_VARIANTS=ON
    > -DSYCL_BUILD_PI_HIP_PLATFORM=AMD -DLLVM_BUILD_TOOLS=ON -DSYCL_ENABLE_WERROR=OFF
    > -DCMAKE_INSTALL_PREFIX=/home/dave/workspace/sycl_workspace/build-dbg/install
    > -DSYCL_INCLUDE_TESTS=ON -DLLVM_ENABLE_DOXYGEN=OFF -DLLVM_ENABLE_SPHINX=OFF
    > -DBUILD_SHARED_LIBS=OFF -DSYCL_ENABLE_XPTI_TRACING=ON -DLLVM_ENABLE_LLD=OFF
    > -DXPTI_ENABLE_WERROR=OFF -DSYCL_CLANG_EXTRA_FLAGS=
    > -DSYCL_ENABLE_PLUGINS=level_zero;cuda;opencl -DLLVM_INSTALL_TOOLCHAIN_ONLY=OFF
    > -DLLVM_ENABLE_RUNTIMES=libcxx;libunwind;libcxxabi -DLIBCXX_CXX_ABI=libstdc++
    > -DLIBCXX_CXX_ABI_INCLUDE_PATHS=/usr/include/x86_64-linux-gnu/c++/9
    > -DLIBCXX_CXX_ABI_LIBRARY_PATH=/usr/lib/gcc/x86_64-linux-gnu/9
    > -DCLANG_DEFAULT_CXX_STDLIB=libstdc++ -DCLANG_DEFAULT_RTLIB=libgcc
    > -DLIBCXX_USE_COMPILER_RT=OFF -DLIBCXX_ENABLE_ABI_LINKER_SCRIPT=OFF
    > -DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR:BOOL=ON -DCMAKE_FIND_DEBUG_MODE=OFF
    > -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DLLVM_INCLUDE_RUNTIMES=OFF
    > -DLLVM_HOST_TRIPLE=x86_64-linux-gnu /home/dave/workspace/sycl_workspace/llvm/llvm

  - The `include/x86_64-unknown-linux-gnu/c++/v1` (see `get_host_triple` in LLVM) directory is not
    included in `clang++`'s default include search paths, meaning most includes (e.g. omp.h) fail, since
    they ultimately try to include `__config_site`. `LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF` doesn't
    help, since it's overridden to ON by `llvm_ExternalProject_Add` calls in `llvm/runtimes`.
    - A rebuild with this set to ON (by default) magically seems to have fixed this. Now see the
      missing `-internal-isystem /home/dave/workspace/sycl_workspace/dist/bin/../include/x86_64-unknown-linux-gnu/c++/v1`
      switch in the `clang++ -v` output.
      - With the above magic fix oneMKL still complains about CUDA, as above.
    - That doesn't seem to be the case, looking a few months later
    - Set  `--cmake-opt='-DLLVM_HOST_TRIPLE=x86_64-linux-gnu'` seemed to work (for RelWithDebInfo,#
      see below) - though warning/errors still contain "-unknown-" so :confused:.
  - `configure.py` using `RelWithDebInfo` then `cmake --build`
    * 16 concurrent build processes hit OOM killer.  4 concurrent was fine.
    * Build is 125GB(!)
    * 88GB for `--install` until failed at
      > CMake Error at tools/libdevice/cmake_install.cmake:46 (file):
        file INSTALL cannot find
        "/home/dave/workspace/sycl_workspace/build-dbg/lib/libsycl-itt-stubs.o": No
        such file or directory.
        Call Stack (most recent call first):
        tools/cmake_install.cmake:146 (include)
        cmake_install.cmake:78 (include)
      - Must `cmake --build . --target sycl-toolchain`
  - Why build DPC++?  I think because __assert_fail is supposedly implemented, but
    > ptxas /tmp/third_party-6d890a/third_party-sm_70.s, line 51508; fatal   : Parsing error near '.': syntax error
    > ptxas fatal   : Ptx assembly aborted due to errors
    > clang-15: error: ptxas command failed with exit code 255 (use -v to see invocation)
    * `sycl` headers are different for hipSYCL vs DPC++
    * Needs more investigation. Same error is reported when using system clang,
      so what did I mean by sycl headers being different - isn't it a clang
      feature?
  - Attempting to build DPC++ clang months later
    * Using CMake variables as in "Generated CMake command" pont above - though
      the LIBCXX_CXX_ABI variables were unused, according to CMake configure
      output.
    - Using CLion to build from clean in Release only the
      `deploy-sycl-toolchain` target using `-j 13`
    - When attempting to use for ConvFelt
      > Could NOT find OpenMP_C (missing: OpenMP_C_FLAGS OpenMP_C_LIB_NAMES)
      * Attempt to build just openmp
        ```
        cmake --build . --target omp
        cmake --install projects/openmp
        ```
        complains that libompd is not available (building target omp provides 
        libomp but not libompd)
      * Do `cmake --build .` - i.e. not limited to sycl-toolchain... then 
        `cmake --install projects/openmp` works. And the Could NOT find error goes away
    - With the above bullet, building hipSYCL against this then fails
      > In file included from /home/dave/workspace/hipSYCL/src/compiler/../../include/hipSYCL/compiler/FrontendPlugin.hpp:31:
        /home/dave/workspace/hipSYCL/src/compiler/../../include/hipSYCL/compiler/Frontend.hpp:38:10: fatal error: 'clang/AST/ASTContext.h' file not found
      * Try `cmake --install . --component libclang-headers` nope
      * Try `cmake --install . --component clang-resource-headers` nope (and perhaps superfluous)
      * Try `cmake --install tools/clang` nope (similar files installed, though)
      * Try `cmake --install . --component clang-headers` installs the header, but hipSYCL is finding
        system llvm, not DPC++. 
        - LLVMConfig.cmake not installed!
        - `cmake --install . --component cmake-exports`
          * but then
            > CMake Error at /home/dave/workspace/sycl_workspace/dist/lib/cmake/llvm/LLVMExports.cmake:970 (message):
            > The imported target "LLVMDemangle" references the file
            > "/home/dave/workspace/sycl_workspace/dist/lib/libLLVMDemangle.a"
      * Tests were being build so `SYCL_INCLUDE_TESTS` + `LLVM_INCLUDE_TESTS` = `OFF`
      * `LLVM_ENABLE_PER_TARGET_RUNTIME_DIR` not used ... ?
      * Try `cmake --build . --target install-llvm-libraries`
      * > CMake Error at /home/dave/workspace/sycl_workspace/dist/lib/cmake/llvm/LLVMExports.cmake:970 (message):
        > The imported target "llvm-tblgen" references the file
        > "/home/dave/workspace/sycl_workspace/dist/bin/llvm-tblgen"
        > but this file does not exist.  Possible reasons include:
      * Try `cmake --build . --target install-llvm-tblgen`
        - Similar error to above, for `llvm-omp-device-info`, so build `install-llvm-omp-device-info`
        - Similar error to above, for `llvm-config`, `llvm-lto`, ...
      * Give up, and just `cmake --install .` - worked! perhaps because removed unit tests.
      * Still no `assert` support, so try force libc using https://stackoverflow.com/questions/72138579/how-to-manually-disable-libstdc-in-cmake-for-clang
        > lib/linux/libclang_rt.builtins-x86_64.a: No such file or directory
        - So add compiler_rt to `LLVM_ENABLE_RUNTIMES` and `LLVM_ENABLE_PROJECTS`
        - Installed under `x86_64-linux-gnu` but searched under `linux` so 
          `-DLLVM_HOST_TRIPLE=x86_64-linux-gnu` in ConvFelt - nope, unused
