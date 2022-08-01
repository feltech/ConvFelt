# Prerequisites

* Ubuntu 20.04
* `hipSyCL` patched to build clang plugin with `libstdc++` and rest with `libc++`.
   Tested target `cuda.integrated-multipass:sm_70`.
  - Attempting to include as a subproject (e.g. CPM package) is tricky. It must be installed to
    give the CMake helper function. But it must also be patched to allow `libc++`.
* `clang-14` compiler using `libc++`
    - `libstdc++` is used by hipSyCL apt package. Attempting to use `libc++` whilst lnking to that
      gives
      > `undefined reference to ``hipsycl::rt::execution_hints::overwrite_with(
      std::__1::shared_ptr<hipsycl::rt::execution_hint>)'`

      So we must build (patched, see above) hipSyCL.
    - `gcc-10`'s `libstdc++` works with hipSyCL, but clang does not support its coroutine library.
      We can trick the `-fcoroutines` vs `-fcoroutine-ts` error by `#define`ing
      `__cpp_impl_coroutine=201902L`, but that hack just reveals deeper issues, i.e.
      > `error: std::experimental::coroutine_traits type was not found; include
      > <experimental/coroutine> before defining a coroutine`
    - `gcc-11`'s `libstdc++` is incompatible, we get the following issue:
      https://github.com/ROCmSoftwarePlatform/rocBLAS/issues/1191
      The pre-built hipSyCL apt packages bundle clang-11 headers/lib. We can patch the headers using
      the fix found in the above link. However, we then run into the same coroutine issues
      as `gcc-10`.
    - Attempting to use `gcc-11` directly fails during hipSyCL device compilation with
      > `clang-11: error: unknown argument: '-fcoroutines'`
* `libomp14-dev` apt package
* `cuda-11.7` apt package
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