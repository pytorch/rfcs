# RFC-0060 asset: artifact inventory

Evidence behind the per-directory table in RFC-0060. Everything here comes
from file listings of real artifacts, collected 2026-09-08 and 2026-09-09.

## Sources and method

| source | how the listing was taken |
|---|---|
| `torch-2.14.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl` | zip central directory, read over HTTP range requests with `list-wheel-contents.py` |
| `torch-2.14.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl` | same |
| `torch-2.14.0-cp312-cp312-macosx_14_0_arm64.whl` | same |
| `torch-2.14.0+cpu-cp312-cp312-win_amd64.whl` | same |
| `libtorch-shared-with-deps-2.14.0+cpu.zip` (Linux) | same |
| `libtorch-macos-arm64-2.14.0.zip` | same |
| `libtorch-win-shared-with-deps-2.14.0+cpu.zip` | same |
| editable install, Linux, CUDA build, Python 3.10, scikit-build-core 1.0.0 | `find` over `site-packages/torch` and the checkout |
| editable install, macOS arm64, CPU build, scikit-build-core 1.0.3 | same |
| editable install, Windows amd64, CPU build, scikit-build-core 1.0.0 | directory probe on a test VM (July 2026 build) |
| python-only modifier | `BUILD_PYTHON_ONLY=ON` configure; install destinations read out of every generated `cmake_install.cmake` |

Wheels come from `https://download.pytorch.org/whl/cpu/torch/` (the macOS
wheel from `/whl/torch/`), zips from
`https://download.pytorch.org/libtorch/cpu/`.
The script regenerates the file lists behind every wheel and zip column for
any release; the per-category counts were derived from those lists by path
pattern.

## Files per category

Counts are files; sizes are uncompressed. Editable rows have no sizes.
`-` means absent.

| category | wheel linux x86_64 | wheel linux aarch64 | wheel macos arm64 | wheel win amd64 | editable linux (cuda) | editable macos | libtorch linux | libtorch macos | libtorch win |
|---|---|---|---|---|---|---|---|---|---|
| metadata (dist-info, licenses) | 204 (2 MB) | 204 (2 MB) | 95 (2 MB) | 99 (2 MB) | 109 | 123 | - | - | - |
| build stamp (build-version, build-hash) | - | - | - | - | - | - | 2 | 2 | 2 |
| headers | 9783 (40 MB) | 9974 (40 MB) | 10129 (39 MB) | 9464 (40 MB) | 9501 | 10166 | 9517 (40 MB) | 10129 (39 MB) | 9464 (40 MB) |
| cmake config (share/cmake, lib/cmake, share/cpuinfo) | 39 | 39 | 30 | 28 | 59 | 37 | 30 | 30 | 28 |
| pkg-config | - | - | - | - | 5 | - | - | - | - |
| share data (Declarations.yaml, share/doc) | 1 | 1 | - | - | 4 | 1 | - | - | - |
| executables (bin/) | 42 (53 MB) | 42 (51 MB) | 3 (8 MB) | 1 (3 MB) | 3 | 3 | - | - | 9 (329 MB) |
| shared libraries (lib/) | 11 (476 MB) | 15 (347 MB) | 7 (419 MB) | 9 (329 MB) | 10 | 7 | 10 (443 MB) | 6 (388 MB) | 9 (329 MB) |
| import libraries (.lib) | - | - | - | 13 (45 MB) | - | - | - | - | 13 (45 MB) |
| static libraries (.a) | - | - | - | - | 17 | 14 | - | - | - |
| debug symbols (.pdb) | - | - | - | - | - | - | - | - | - |
| lib/ other (libshm sources, misc) | 8 | 8 | 5 | 5 | - | - | 5 | 5 | 5 |
| extension modules (_C, _nccl_ep, ...) | 1 | 1 | 1 | 1 | 1 | 1 | - | - | - |
| python sources (.py) | 2437 (51 MB) | 2437 (51 MB) | 2437 (51 MB) | 2437 (52 MB) | 4 | 4 | - | - | - |
| type stubs (.pyi, py.typed) | 44 (3 MB) | 44 (3 MB) | 44 (3 MB) | 44 (3 MB) | 44 | 44 | - | - | - |
| codegen inputs (torchgen/packaged) | 75 | 75 | 69 | 69 (1 MB) | - | - | - | - | - |
| inductor/aoti C++ assets (codegen .h/.cpp, aoti_runtime, script.ld) | 7 | 7 | 4 | 4 | 4 | 4 | - | - | - |
| templates (.jinja) | 20 | 20 | 20 | 20 | 23 | 23 | - | - | - |
| serde schemas (.yaml/.thrift) | 3 | 3 | 2 | 2 | 2 | 2 | - | - | - |
| web assets (.js/.mjs/.html) | 4 | 4 | 4 | 4 | 4 | 4 | - | - | - |
| benchmark/valgrind C++ (utils/benchmark) | 5 | 5 | 5 | 5 | 5 | 5 | - | - | - |
| json/yaml data | 1 | 1 | 1 | 1 | 1 | 1 | - | - | - |
| testing assets | 18 | 18 | - | - | 1 | 1 | - | - | - |
| other | 472 (89 MB) | 472 (96 MB) | 4 | 4 | 8 | 3 | - | - | - |

Notes on the cells: the Linux wheels' "other" is `torch/test/` (114 C++ test
binaries, 89 MB) plus zip directory entries; the editable listings covered
`torch/` only, and `torchgen/packaged` is present in those trees as well (93
files); the Windows editable install was probed by directory rather than
listed per file, so it has no column (its `bin\` holds `protoc.exe` only,
its `lib\` nine DLLs plus import libraries, `include\` and
`share\cmake\Torch` are present); the Windows zip's `bin/` holds the DLLs,
its `lib/` the import libraries and a copy of the DLLs.

## Groups by consumer

| group | contents | produced by | wheel | editable (site-packages) | libtorch zip |
|---|---|---|---|---|---|
| A. Python payload | `.py` sources, `_C` (+ `_nccl_ep`), `.pyi` + `py.typed`, `version.py`, `_rocm_init.py`, vendored packages with LICENSE files | scikit-build-core package walk; `torch/CMakeLists.txt` for `_C` and `version.py` | yes | `.py` from the checkout; `_C`, stubs, `version.py` in site-packages | no |
| A2. Python package data | jinja templates, `_export/serde` schemas, web assets (`model_dump`, `viz`), `graph_break_registry.json`, inductor codegen `.h/.cpp` + `aoti_runtime/model.h` + `script.ld`, valgrind wrapper `.cpp/.h`, `testing/_internal/generated`, `torchgen/packaged/{ATen,autograd}` | `cmake/PackageData.cmake`, `cmake/FileMirroring.cmake` (SKBUILD only) | yes | yes, in site-packages | no |
| B. Native runtime | `lib/`: torch, torch_cpu, c10, torch_python, torch_global_deps, shm, GPU variants; bundled third-party runtime (Linux `libgomp`; aarch64 adds `libopenblas`, `libarm_compute*`, `libgfortran`; macOS `libomp.dylib`; Windows `libiomp5md`, `libiompstubs5md`, `uv.dll`); Windows `.lib` import libraries | `caffe2/CMakeLists.txt`, `c10/*`, `torch/CMakeLists.txt`, `cmake/PostBuildSteps.cmake`, wheel repair | yes | yes | yes; Windows zip puts DLLs in `bin/`, `.lib` in `lib/` |
| C. C++ development surface | `include/` (about 9,500 headers, 40 MB), `share/cmake/{Torch,Caffe2,ATen}`, `share/ATen/Declarations.yaml` | `caffe2/`, `aten/`, `c10/`, `torch/headeronly`, `cmake/Codegen.cmake`, `CMakeLists.txt` | yes | yes | yes |
| D. Tools | `bin/torch_shm_manager` (non-MSVC), `bin/protoc` (+ versioned copy; `protoc.exe` on Windows), `ptxas` when bundled | libshm, vendored protobuf, `CMakeLists.txt` | yes | yes | dropped by the zip packaging |
| E. Test artifacts | `torch/test/*` (114 binaries), `bin/test_*`, `bin/*Test`, `bin/upgrader_models/*.ptl`, `lib/lib{jitbackend,torchbind}_test.so`, `libbackend_with_compiler.so`, `libaoti_custom_ops.so` | `BUILD_TEST=ON` (the default) with `INSTALL_TEST` | Linux wheels only (about 150 MB); macOS and Windows CD jobs build with `BUILD_TEST` off | no | Linux zip inherits the test `.so`s in `lib/`; `test/` and `bin/` dropped |
| F. Third-party build residue | static `.a` libraries (14 to 17), `lib/cmake/{dnnl,fmt,ittapi,protobuf,sleef}`, `lib/pkgconfig`, `share/doc/dnnl`, `share/cmake/{fbgemm,kineto}`, `share/cpuinfo`, protobuf `.inc` files, `include/fp16/*.py` | vendored third-party install rules | excluded by `[tool.scikit-build.wheel].exclude` | present: the wheel exclude list does not apply to the editable staging tree | no |
| G. Packaging metadata | `*.dist-info/{METADATA,RECORD,...}` and about 100 to 200 bundled license files; zip: `build-version`, `build-hash` | scikit-build-core, `PostBuildSteps` license bundling, CD zip packaging | yes | yes | build stamps only |
| H. Debug symbols | `.pdb` | `OPTIONAL` install rules on Windows | no (release) | no | no |

## Where each directory comes from

* `bin/protoc(.exe)`: the vendored protobuf's own install rules
  (`protobuf_INSTALL`, default ON, never overridden), whenever
  `BUILD_CUSTOM_PROTOBUF=ON` (default); `USE_SYSTEM_LIBS` turns it off.
* `bin/torch_shm_manager`: `torch/lib/libshm/CMakeLists.txt`, non-MSVC only,
  gated on `BUILD_PYTHON`; `libshm_windows` installs only `lib/` and
  `include/`. MSVC with system protobuf has no `bin/` at all.
* `BUILD_TEST` (default ON; installed while `INSTALL_TEST` follows it) puts
  test executables under `bin/` and `test/` and test libraries under `lib/`.
  The other `bin/` producers are off by default: `BUILD_BINARY`,
  `BUILD_BUNDLE_PTXAS` (CUDA manywheels only).
* `lib/`: present in every shape. `lib/torch_python` is installed by
  `torch/CMakeLists.txt` whenever `BUILD_PYTHON` is on, python-only modifier
  included; `BUILD_PYTHON=OFF` skips that file, so the standalone shape has
  none.
* `share/cmake/Torch/TorchConfig.cmake`: `caffe2/CMakeLists.txt` inside
  `if(NOT BUILD_LIBTORCHLESS)`; absent under the python-only modifier.
* `include/google` (protobuf headers) ships in every wheel; only the `.inc`
  files are excluded. `include/THC/` exists on CUDA builds only.

## Python files the build generates

Measured on the 2.14.0 Linux wheel against the `v2.14.0` tag, cross-checked
with the editable trees.

Generated into the checkout (gitignored) and installed, so present in both
trees under an editable install:

* `torch/version.py` (from `tools.generate_torch_version`; explicit install
  rule because scikit-build-core skips gitignored files)
* `torch/testing/_internal/generated/annotated_fn_args.py` (autograd codegen)
* seven type stubs generated from `.pyi.in`: `torch/_C/__init__.pyi`,
  `torch/_C/_nn.pyi`, `torch/_C/_VariableFunctions.pyi`, `torch/_VF.pyi`,
  `torch/return_types.pyi`, `torch/nn/functional.pyi`,
  `torch/utils/data/datapipes/datapipe.pyi`
* CUDA builds: the CUPTI stub module under `torch/profiler/_cuspy/`
* ROCm builds: `torch/_rocm_init.py`

Never in the checkout, install tree only:

* `torch/_inductor/kernel/vendored_templates/cutedsl/kernels/` (two files
  mirrored from `third_party/cutlass`, with an empty `__init__.py`)
* `torchgen/packaged/` (twelve `.py` files from `tools/autograd` plus the
  ATen yaml and templates), mirrored so torchgen works standalone

For imports both classes resolve deterministically: the editable finder maps
every CMake-installed module by name to the install tree (relative path in
its mapping), tracked modules to the checkout (absolute path), and the
gitignored checkout copies never enter the package walk. For
`importlib.resources` the first class is the both-trees case; see the RFC's
resolution rules.

## Linux wheel test artifacts

`torch/test/` (114 C++ test binaries, 89 MB), 19 test executables and the
`upgrader_models/*.ptl` fixtures under `bin/` (53 MB with `protoc`), and four
test libraries under `lib/`, about 150 MB in total. Absent from the macOS and
Windows wheels, whose CD jobs set `BUILD_TEST` off. The Linux libtorch zip
inherits the four libraries; its `test/` and `bin/` are dropped by the
extraction.

## CI consumers of the `tools/build_libtorch.py` install prefix

Checked 2026-09-09 for the prefix change proposed in the RFC.

| caller | build dir | reads the install prefix? |
|---|---|---|
| Linux libtorch trunk jobs (`.ci/pytorch/build.sh`) | `/tmp/cpp-build/caffe2` | no; build-only (`build-generates-artifacts: false`) |
| macOS C++ API test (`.ci/pytorch/macos-test.sh`) | `../cpp-build/caffe2` | no; runs `${CPP_BUILD}/caffe2/bin` |
| lightweight-dispatch test (`test/mobile/lightweight_dispatch/build.sh`) | `build/custom_test_artifacts` | no; runs `build/bin/test_codegen_unboxing` |
| Windows arm64 (`.ci/pytorch/windows/arm64/build_libtorch.bat`) | checkout | yes; moves `torch\{bin,cmake,include,lib,share,test}` into `libtorch\`, DLLs from `bin\` to `lib\` |
| s390x `BUILD_PYTHONLESS` branch (`.ci/wheel/linux/build_common.sh`) | `build/build` | copies `build/build/lib`; no caller since the libtorch package type left CD |

## Observations

1. Groups A and A2 are the only ones that differ between the wheel and the
   editable shape, and A2 is the surprising part: mirrored package data lands
   in site-packages next to `_C`, not in the checkout.
2. The editable tree is a superset of the wheel's native tree (group F), so
   probes that key on extra files or counts misclassify it. A `lib/` probe is
   safe.
3. Group E is a CD configuration gap (Linux manywheel jobs leave `BUILD_TEST`
   at its default), not a layout property.
4. The Windows libtorch zip is a different shape from the Windows wheel
   (DLLs in `bin/`); `TorchConfig.cmake` handles it, `__file__`-style code
   would not. The zip also carries `torch_python.dll` because the extraction
   filters on the `libtorch_python` prefix that Windows names lack.
5. Platform-specific bundled runtimes (OpenMP flavours, OpenBLAS and Arm
   Compute on aarch64, libuv on Windows) live in `lib/` beside torch's own
   libraries and are what the global-deps preload and the Windows DLL
   directory must find; they are why `lib/` must be resolvable from `_C`.
6. Tools (group D) are the least consistent group: in wheels and editable
   installs on every platform, absent from every zip.
