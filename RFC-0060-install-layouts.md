# PyTorch install layouts: what goes where, and how code finds it

**Authors:**
* @zklaus


## **Summary**

PyTorch is installed in one of three shapes, reached by six commands, and the
documentation describes the result of one of them. The shapes differ in where
the compiled artifacts (`lib/`, `include/`, `bin/`, `share/`, the `_C`
extension) sit relative to the Python sources, so code that derives paths from
`__file__` is right in some shapes and wrong in others. This RFC proposes to
(1) document the shapes and the commands that produce them, from listings of
the shipped 2.14.0 artifacts and of real editable installs; (2) state two
resolution rules for in-tree code, one for resources it reads and one for
directories it hands to external tools; (3) settle where the documentation
lives; and (4) align the standalone `cmake --install` recipe and
`tools/build_libtorch.py` with the libtorch zips through three build
defaults.


## **Motivation**

The scikit-build-core migration (pytorch/pytorch#180247) made editable
installs use redirect mode. Its commit message states the consequence:

> Redirect mode is the first layout to split the Python sources (checkout)
> from the compiled artifacts (install tree); in wheel and `setup.py develop`
> installs they sat together, so those `dirname(__file__)` lookups happened to
> be right.

Three `__file__` lookups were shimmed at migration time. The rule was not
written down, and the tree was not swept for other sites of the same shape. A
month later pytorch/pytorch#195293 reported `torch.mps.compile_shader`
failing under an editable install, and a grep found seven more call sites
(pytorch/pytorch#195887).

A survey of `docs/`, `CONTRIBUTING.md` and `README.md` (2026-08-31) found
one sentence describing any install layout:

| Layout | Where documented | What it says |
|---|---|---|
| Wheel (`site-packages/torch/`) | nowhere | nothing |
| Editable (redirect mode) | `CONTRIBUTING.md` | rebuild semantics only; the mechanism sentence was wrong (`.egg-link`) until pytorch/pytorch#195737 |
| Standalone `cmake --install` | `docs/libtorch.rst` | the command, then stops |
| In-checkout `tools/build_libtorch.py` | `docs/libtorch.rst` | "installed into `<pytorch_root>/torch/{lib,include,share}`" (accurate) |
| Consuming a libtorch zip | `docs/cpp/source/installing.md` | `CMAKE_PREFIX_PATH` usage; never shows the tree |

Wheel layout also leaks into unrelated docs: `torch.compiler_aot_inductor.md`
hardcodes `/path/to/python/install/site-packages/torch/share/cmake`, wrong
for an editable install.

Every failure is a conflation of layouts, so the layouts want documenting
together, with the resolution rule stated beside them.


## **Proposed Implementation**

### Three shapes

What matters to code is where the compiled artifacts sit relative to the
Python sources, not which command ran. There are three answers.

**Shape 1: artifacts beside the sources (wheel shape).** `_C` sits next to
`torch/__init__.py`, with `lib/`, `include/`, `bin/` and `share/` in the same
package directory. `__file__`-relative lookups are correct. Reached by
installing a published wheel; by `pip install .` from a checkout, which builds
a wheel under PEP 517 and installs it, so "installed from source" is still
this shape; and by installing a locally built wheel, as the CD pipelines do.

**Shape 2: artifacts split from the sources (editable shape).** Reached by
`pip install -e .`, which scikit-build-core performs in redirect mode: "Python
files come from the source tree, CMake-built artifacts from the build tree"
(`pyproject.toml`). The `.py` files are imported from the checkout; everything
CMake installs (`include/`, `lib/`, `bin/`, `share/`, `_C`) is staged beside
the installed distribution in site-packages. The checkout's `torch/lib` holds
only the tracked libshm sources and the checkout has no `torch/include`.
`torch.__path__` names both trees, in an order that is not currently
guaranteed (see the rules). `__file__`-relative lookups resolve into the
checkout and are wrong.

**Shape 3: artifacts without Python sources (standalone shape).** Reached by
a `cmake --install` that scikit-build-core does not drive. With
`BUILD_PYTHON=OFF`, `torch/CMakeLists.txt` returns before defining
`torch_python`, so neither `_C` nor `libtorch_python` exists. The libtorch
zips are this shape as shipped: on Linux and macOS `include/`, `share/`,
`lib/`, `build-version`, `build-hash`, no `bin/`, no Python files. The
Windows zip follows the platform convention instead, DLLs in `bin/` and
`.lib` import libraries in `lib/`, so the Windows zip and the Windows wheel
(DLLs in `torch/lib/`) are different layouts of the same binaries.
`TorchConfig.cmake` absorbs that; path-deriving code would not. (The Windows
zip also carries `torch_python.dll`: the wheel-to-zip extraction filters on
the `libtorch_python` prefix Windows names lack. A packaging bug.) A raw
`cmake --install` with today's defaults additionally installs `bin/`
(`torch_shm_manager`, `protoc`), `libtorch_python` and the C++ test
binaries; the standalone proposals below remove them so recipe and zips
agree. Two commands produce this shape:

* the `cmake` recipe in `docs/libtorch.rst`, with a prefix of the user's
  choosing; it leaves `BUILD_PYTHON` at its default today;
* `tools/build_libtorch.py`: the same build with `BUILD_PYTHON=OFF` and
  `CMAKE_INSTALL_PREFIX=<checkout>/torch` fixed. That prefix puts shape 3's
  content in shape 1's location, the checkout's `torch/{lib,include,share}`,
  next to the Python sources. It is the layout the one existing doc sentence
  describes and the libtorch CI jobs build. Its only in-tree users are CI
  scripts; the Windows arm64 one moves the artifacts straight back out into a
  `libtorch/` directory.

The in-checkout variant of shape 3 and shape 2 are hostile in the same tree:
a checkout that ran `tools/build_libtorch.py` holds exactly the artifacts a
redirect-mode install keeps out of it, and nothing says so. The standalone
proposals move the script's prefix out of the source tree.

**Modifier: python-only (`BUILD_LIBTORCHLESS`, forced on by
`BUILD_PYTHON_ONLY=ON` in `cmake/EnvVarForwarding.cmake`).** Not a shape. It
applies to shapes 1 and 2 and changes contents, not locations: `lib/` holds
`torch_python` and `libshm` but no `libtorch`, `c10` or `torch_cpu`;
`include/` is partial; `share/cmake/` lacks `Torch/TorchConfig.cmake`. No such
wheel is published and no CI exercises it (pytorch/pytorch#159990). It is
recorded so that code probing for an installed tree assumes no more than it
provides.

### What each shape puts under `torch/`

From listings of the 2.14.0 CPU wheels and libtorch zips for Linux, macOS
and Windows, editable installs on the same three platforms, and a
`BUILD_PYTHON_ONLY=ON` configure. Sources, the per-category counts, the
consumer groups and the provenance of each directory are in
[`RFC-0060-assets/artifact-inventory.md`](RFC-0060-assets/artifact-inventory.md);
[`RFC-0060-assets/list-wheel-contents.py`](RFC-0060-assets/list-wheel-contents.py)
regenerates the wheel and zip listings for any release.

| directory | shape 1, wheel | shape 2, installed tree in site-packages | shape 2, checkout `torch/` | shape 3, libtorch zip | python-only modifier (derived) |
|---|---|---|---|---|---|
| `lib/` | torch's own libraries plus the bundled runtime: `libgomp` on Linux (aarch64 adds OpenBLAS, Arm Compute, `libgfortran`), `libomp` on macOS, `libiomp5md` + `libuv` on Windows; Windows adds `.lib` import libraries. Linux wheels also carry four test libraries (see note) | **superset of the wheel**: everything the wheel has, plus 14-17 static `.a` libraries, `lib/cmake/{dnnl,fmt,ittapi,protobuf,sleef}` and `lib/pkgconfig/` that the wheel exclude list removes | libshm **sources** only (tracked) | Linux: 10 libraries, macOS: 6, no `torch_python`. **Windows: DLLs live in `bin/`, `lib/` holds the `.lib` import libraries** (and a copy of the DLLs; `torch_python.dll` included, the filter bug) | `torch_python`, `libshm`, vendored `cmake/`, `pkgconfig/`; no libtorch, c10, torch_cpu |
| `bin/` | `torch_shm_manager` + `protoc` on Linux and macOS; `protoc.exe` only on Windows | same as wheel | absent | Linux and macOS: absent. Windows: the DLLs (see `lib/`) | `torch_shm_manager`, `protoc` |
| `include/` | full (about 9,500 headers, 40 MB, protobuf's `google/` included) | full, plus the protobuf `.inc` files and the fp16 generator scripts the wheel excludes; CUDA builds add `THC/` | absent | full | **partial**: ATen, `torch/headeronly`, `torch/csrc/stable`, two jit dirs, third-party |
| `share/` | `cmake/{Torch,ATen,Caffe2}`; Linux adds `ATen/Declarations.yaml` | same, plus `cmake/{fbgemm,kineto}`, `cpuinfo/`, `doc/dnnl` that the wheel excludes | absent | `cmake/{Torch,ATen,Caffe2}` | `cmake/{ATen,Caffe2,fbgemm,kineto}`; **no `Torch/TorchConfig.cmake`** |
| `_C` extension | beside `__init__.py` | in the installed tree (served by scikit-build-core's redirect loader) | absent | absent | beside `__init__.py` |
| `test/` | Linux wheels only: 114 C++ test binaries (see note) | absent | absent | absent | absent |

Three notes:

* **The editable tree is a strict superset of the wheel's native tree.**
  `[tool.scikit-build.wheel].exclude` trims third-party residue (static
  libraries, vendored CMake and pkg-config files, protobuf `.inc` files, dnnl
  docs) from the wheel but not from the editable staging tree. A probe that
  keys on those files, or on file counts, misclassifies an editable install;
  a probe on `lib/` does not.
* **Some Python files exist only because the build made them.** One class
  is generated into the checkout (gitignored) and installed, so it exists in
  both trees under shape 2: `torch/version.py`, the generated `.pyi` stubs,
  `annotated_fn_args.py`, and the CUDA and ROCm specific modules. The other
  exists in the install tree only: the mirrored cutedsl kernel package and
  `torchgen/packaged/`. Imports of both resolve deterministically, because
  the editable finder maps every CMake-installed module to the install tree
  and gitignored checkout copies never enter the package walk.
  `importlib.resources` on the first class is the both-trees case: a stub
  read as data goes to whichever tree the finder listed first, which today
  varies with the hash seed. Full list in the inventory asset.
* **The Linux wheels ship test artifacts** the other platforms do not,
  about 150 MB under `test/`, `bin/` and `lib/`, because `BUILD_TEST`
  defaults to ON and only the macOS and Windows CD jobs turn it off. The
  Linux zip inherits the four test libraries. A CD configuration gap, out of
  scope here.

### The standalone shape: proposed defaults

The zips are the standalone shape as users receive it: no `bin/` (Windows
aside, where `bin/` holds the DLLs) and, the Windows filter bug aside, no
`libtorch_python`, since they became wheel extractions. The documented recipe
and `tools/build_libtorch.py` produce something else. Three changes close the
gap.

**`BUILD_PYTHON=OFF` and `BUILD_TEST=OFF` are the standalone recipe.**
`torch/CMakeLists.txt` returns early when `BUILD_PYTHON` is off, and it is
what builds `torch_python`, `_C` and libshm (`torch_shm_manager` included).
With the default ON, the recipe in `docs/libtorch.rst` compiles `_C` and
installs nothing from it, installs `libtorch_python` no C++ consumer can use,
and installs a shared-memory manager that exists for `torch.multiprocessing`.
`BUILD_TEST` also defaults to ON and installs test executables and libraries
the zips do not carry. Proposed: the recipe sets both off (as
`tools/build_libtorch.py` already does for `BUILD_PYTHON`), and a configure
that scikit-build-core does not drive emits a `message(STATUS)` when
`BUILD_PYTHON` is on, saying the Python bindings will be built and not
installed, and naming the switch.

**`protobuf_INSTALL=OFF` for the vendored protobuf.** `protoc` compiles the
ONNX schemas at build time and is never used afterwards. It lands in `bin/`
because the vendored protobuf's `protobuf_INSTALL` defaults to ON and
`cmake/ProtoBuf.cmake` never turns it off. The same default installs the
static `libprotobuf`, the `include/google` headers and protobuf's CMake and
pkg-config files; the wheel exclude list strips the library, the CMake and
pkg-config files and the `.inc` files, and ships the headers and `protoc`. A
downstream cannot use the compiler without the library, and the exported
config aliases `protobuf::libprotobuf` to a dummy target when protobuf is
linked locally, so nothing can be built against it. Proposed: set
`protobuf_INSTALL` to OFF before adding the vendored protobuf. Every shape
loses the residue; the wheels lose two copies of a 4 to 5 MB binary on Linux
and macOS (the symlink is materialised when the wheel is zipped) and one on
Windows.

With these, a standalone install has no `bin/` and its `lib/` matches the
zip, which settles "should `bin/` ship" by construction. The wheels keep
`bin/torch_shm_manager`; it is a Python-side tool.

**`tools/build_libtorch.py` installs outside the source tree.** Its fixed
prefix, `<checkout>/torch`, is what makes its output collide with an editable
install. Proposed: the default prefix becomes a `libtorch/` directory beside
the build directory the script creates (`<cwd>/build` and `<cwd>/libtorch`),
`--install-prefix` overrides it, and the script refuses a prefix inside the
source tree. Callers run from a scratch directory, as every CI caller already
does. Of the five in-tree callers (table in the inventory asset), only the
Windows arm64 libtorch script reads the prefix: it moves the checkout's
`torch\{bin,cmake,include,lib,share,test}` into `libtorch\` by hand, and
with the new default reads the prefix directly. The Linux libtorch trunk jobs
are build-only, the macOS and lightweight-dispatch tests run binaries from
their build directories, and the s390x `BUILD_PYTHONLESS` branch has had no
caller since the libtorch package type left CD.

### The resolution rules

Code inside `torch` must not derive the path of a compiled artifact or a data
file from `__file__`, which is correct only when sources and artifacts share
a directory (shape 1). Two kinds of lookup exist and need different
mechanisms. Both go through `torch._utils_internal`, one entry point each,
sharing the anchor logic.

**Rule 1: resources are read through the resource helper, which is
`importlib.resources` underneath.** A resource is a file whose contents the
code consumes: templates, schemas, JSON, web assets, a header read as text,
the Metal shader headers, most of the package data CMake mirrors into the
package. `importlib.resources.files("torch")` works in every shape with a
Python package; under shape 2 scikit-build-core's resource reader spans the
install tree and the checkout, and a file in either resolves to a concrete
path. The call site is `torch._utils_internal.resource(*parts)`, which
returns the stdlib `Traversable` unchanged. Callers open, read or iterate
it; they do not turn it into a path.

The indirection absorbs three things and is designed to disappear:

* Under shape 2 the order of the multiplexed members was not guaranteed
  (scikit-build/scikit-build-core#1565), so a file present in both trees
  resolved to a hash-seed-dependent copy. The fix (install tree first,
  scikit-build/scikit-build-core#1566) and a public `module.__loader__.paths`
  in that order (scikit-build/scikit-build-core#1567) are in review. The
  helper prefers the install tree: via `__loader__.paths` where present, on
  older releases by reading the reader's private member list, in one
  function with a comment naming the removal condition.
* Python 3.10 and 3.11 get scikit-build-core's own multiplexed-path class,
  3.12 and later the stdlib one. The helper hides which.
* python/importlib_resources#310 proposes a public `paths` accessor for the
  stdlib; scikit-build-core#1567 adds the same to the pre-3.12 fallback. When
  both cover the supported range the helper drops the private access.

When all three are resolved the helper is an alias for
`importlib.resources.files("torch").joinpath(...)`, kept as the lint's single
allowed call site or removed.

**Rule 2: locations are resolved through the location helper.** A location
is a directory handed to something outside the interpreter: the include
directory for a compiler, `lib/` for the `ctypes` preload and
`os.add_dll_directory`, `share/cmake` for `torch.utils.cmake_prefix_path`,
`bin/` for `torch_shm_manager`. `importlib.resources` cannot provide these: a
directory that exists in both trees stays multiplexed, on which
`os.fspath()` raises, and `torch/lib` is a tracked source directory, so under
shape 2 it multiplexes on every platform. `as_file()` supports directories
only from Python 3.12, by copying the tree into a temporary directory. The
helper is the existing `torch._utils_internal.get_file_path("torch", ...)`,
returning a filesystem path; `torch.utils.cmake_prefix_path`,
`torch.utils.cpp_extension` and every in-tree location lookup go through it.

The type difference is deliberate: a `Traversable` for resources, a path for
locations. A caller that needs a path for a resource is asking for a
location and uses rule 2.

Both helpers anchor on the import system, in this order:

1. **The editable loader's `paths`.** scikit-build/scikit-build-core#1567 (in
   review) gives every loader a `paths` attribute listing the package's
   search locations in `__path__` order, install tree first after #1566, next
   to the existing `__loader__.rebuild()` hook. Absent on older releases, so
   the chain continues.
2. **The extension module's spec.**
   `importlib.util.find_spec("torch._C").origin` answers "where will the
   extension be imported from" in every shape that has one: the install tree
   under shape 2 (verified on Linux and Windows), beside `__init__.py` for
   shape 1. The native-AOT stage 2 driver chose the same anchor.
3. **`__file__`**, only when `find_spec("torch._C")` returns nothing. Not
   shape 3, which never imports `torch`, but an interpreter without a
   discoverable spec for the extension, such as a frozen or embedded build.

Two constraints:

* **Probe for an installed tree with `lib/`**, never `bin/`, `include/` or
  `share/cmake/Torch`: `bin/` is empty under MSVC without vendored protobuf,
  `include/` is partial under the python-only modifier, `TorchConfig.cmake`
  is absent there, and the editable tree is a superset of the wheel's.
* **Do not rely on the order of `torch.__path__`.** Under shape 2 it varied
  with `PYTHONHASHSEED` (scikit-build/scikit-build-core#1565, fixed in #1566,
  in review). Until every supported scikit-build-core release has the fix,
  nothing outside the resource helper's shim may depend on which tree comes
  first. This is why stale artifacts in a checkout are a correctness problem,
  not clutter.

**Enforcement.** With one allowed call site per rule, the lint is a grep:
flag `__file__`-derived paths and direct `importlib.resources.files("torch")`
calls in `torch/` outside `torch._utils_internal`. It backs the first metric.

### Sequencing

1. **The helpers and the known call sites.** `get_file_path` re-anchored on
   the `_C` spec and the seven `__file__` sites fixed (pytorch/pytorch#195587,
   open; the pytorch/pytorch#195887 sub-issues #195889, #195890, #195891).
   The resource helper lands with the first call site that reads a file.
2. **The editable-install CI job** (pytorch/pytorch#195888), so shape 2 is
   exercised on every PR.
3. **The lint**, once the inventory is at zero.
4. **The documentation page**, the three links into it, and the
   `CONTRIBUTING.md` paragraph.
5. **The standalone defaults** in one PR; the standalone-layout CI job asserts
   the resulting tree.
6. **The `tools/build_libtorch.py` prefix**, with the Windows arm64 script
   adapted.
7. **Upstream, in parallel**: scikit-build/scikit-build-core#1566 and #1567,
   python/importlib_resources#310. Each landing retires a shim; the
   scikit-build-core pair does so once the `build-system.requires` floor
   passes the release that carries them.

### Vocabulary

The component taxonomy proposed in pytorch/pytorch#184873 (`libtorch` /
`torch` / `dev` / `third_party`), once landed, is the vocabulary: `cmake
--install --component libtorch` (plus `dev`) is the C++ distribution, the
`torch` component the wheel's Python payload. Without it the same split gets
described twice.

### Placement

Proposed: **one page**, "Install layouts", holding the three shapes, the
table and the two rules, linked from `CONTRIBUTING.md`, `docs/libtorch.rst`
and `docs/cpp/source/installing.md` where each describes, or fails to
describe, its own shape. The problem is conflation between shapes; one page
that shows them side by side is the direct remedy, and it gives the rules a
single home.

The alternative is a split by audience: the editable shape and the rules in
`CONTRIBUTING.md`, the wheel tree, standalone install and zips in the C++
docs. Each audience already looks in its own place, but the comparison,
which is the content, gets written twice or not at all. The split stays an
option for the discussion; either outcome leaves the rest unchanged.


## **Metrics**

* Zero in-tree `__file__`-derived asset lookups and zero direct
  `importlib.resources.files("torch")` calls outside the helper module; the
  pytorch/pytorch#195887 inventory reaches zero and the lint keeps it there.
* An editable-install CI job (pytorch/pytorch#195888) exists and stays green.
* Every shape is described in exactly one place; every other place links
  there.


## **Drawbacks**

* Documentation of the trees goes stale when install rules change.
  Mitigation: the inventory asset names the CMake rule behind each directory
  and the listing script regenerates the file lists behind the counts, so a
  rule change points at the row and a release refreshes the numbers.
* On scikit-build-core releases without `__loader__.paths`
  (scikit-build/scikit-build-core#1567), the resource helper reads a private
  attribute of the editable reader. Mitigation: one function, a test that
  fails loudly if a release renames it, removal once the build requirement's
  floor carries #1567.
* The rules change no runtime behaviour. The standalone defaults do: wheels
  stop shipping `torch/bin/protoc` (a build-time tool with no library to use
  it against); `tools/build_libtorch.py` stops installing into the checkout
  (the one in-tree consumer, the Windows arm64 script, is adapted); and the
  documented `cmake --install` stops installing `libtorch_python` and the test
  binaries, matching the zips users already receive.


## **Alternatives**

* **Do nothing.** The seven call sites get fixed one by one and the next
  contributor writes the eighth.
* **Collapse the shapes**, e.g. make redirect mode install artifacts into the
  checkout so shape 2 becomes shape 1. Rejected: it reintroduces the in-tree
  artifact problem the migration removed and does nothing for shape 3.
* **Document only the editable layout** in `CONTRIBUTING.md`. It is the shape
  that broke, but the conflation is between shapes.
* **Use `importlib.resources` directly, no helpers.** It stays the
  implementation, but it cannot be the call site: it cannot produce a
  directory for an external tool (a directory in both trees has no single
  path, and `as_file()` copies), and under today's editable finder its
  answer for a file in both trees is not deterministic
  (scikit-build/scikit-build-core#1565). The helpers hold the temporary shims
  and give the lint one call site; both become aliases as
  scikit-build-core#1566/#1567 and python/importlib_resources#310 land.
* **Split the documentation by audience.** See Placement.


## **Prior Art**

The ecosystem documents the editable *mechanism* and its *limitations*, not
the resulting trees, and leaves asset resolution to each project. Checked
2026-09-09:

* **setuptools** ("Development Mode"): non-Python files, data files, binary
  extensions, headers and metadata "may be exposed as a snapshot of the
  version they were at the moment of the installation".
* **meson-python** ("Editable installs") documents its loader-based redirect
  and states rule 1's consequence: data files "need to be accessed using
  `importlib.resources`", because a `__file__`-relative read "would fail when
  the package is installed in editable mode".
* **numpy** ("Building from source"): "editable installs are fundamentally
  incomplete installs. Their only guarantee is that `import numpy` works";
  "headers, entrypoints, and other such things may not be available".
  `numpy.get_include()` is the precedent for rule 2's shape, a function, and
  for the trap: it joins `os.path.dirname(numpy.__file__)` with the include
  subdirectory, with a "running from numpy source directory" branch, which is
  the `__file__` heuristic that breaks under a split layout and the reason for
  the warning.
* **scipy** carries numpy's warning and adds that editable installs "tend to
  hit weird corner cases more frequently than regular installations".
* **scikit-build-core** documents what redirect mode serves live and what it
  snapshots, not what a project ends up with on disk.

None documents the layouts, and none has PyTorch's second requirement:
handing directories to compilers and loaders at runtime as a user-facing
feature (`torch.compile`, `cpp_extension`, `cmake_prefix_path`). Hence a
location rule with a supported anchor rather than numpy's warning.


## **How we teach this**

* Name the shapes consistently ("wheel shape", "editable shape", "standalone
  shape") and say which command produced a tree: `pip install .` is the wheel
  shape, `pip install -e .` the editable shape, `cmake --install` and
  `tools/build_libtorch.py` the standalone shape.
* Two rules, one sentence each: read resources through
  `torch._utils_internal.resource(...)`, `importlib.resources` underneath;
  resolve locations through `get_file_path("torch", ...)`. Never `__file__`,
  and never a path from a resource.
* `CONTRIBUTING.md` gains a short "Where things end up" paragraph in its
  editable-install section: the shape after `pip install -e .`, the two rules
  in a sentence each, and a link to the page. It states no tree of its own, so
  it cannot drift.


## **Unresolved questions**

* **Placement.** One page is proposed; the split by audience stays on the
  table.
* **Sequencing of the `tools/build_libtorch.py` change.** Whether the Windows
  arm64 script is adapted in the same PR or the script keeps a transition
  flag for one release.
* **Helper naming.** `torch._utils_internal.resource(...)` is a placeholder;
  the design is the two-helper split and the anchor order.
* Out of scope: the python-only (`BUILD_LIBTORCHLESS`) modifier, described so
  the rules assume no more than it provides, nothing else decided.


## Resolution
(to be filled in)

### Level of Support
(to be filled in)

#### Additional Context
(to be filled in)

### Next Steps
(to be filled in)

#### Tracking issue
(to be filled in)

#### Exceptions
(to be filled in)
