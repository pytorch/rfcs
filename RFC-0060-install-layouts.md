# PyTorch install layouts: what goes where, and how code finds it

**Authors:**
* @zklaus


## **Summary**

PyTorch is installed in one of three layouts. In the wheel the compiled
artifacts (`lib/`, `include/`, `bin/`, `share/`, the `_C` extension) sit
beside the Python sources; in an editable install scikit-build-core's
redirect mode splits them; a standalone `cmake --install` has no Python
sources at all. Code that derives paths from `__file__` is right in the first
layout and wrong in the other two, and none of the three is documented. This
RFC proposes (1) one documentation page that names the three layouts and
shows what each puts where, from listings of the shipped 2.14.0 artifacts and
of real editable installs; (2) one rule for in-tree code, carried by two
helpers anchored on the import system, one for files the code reads and one
for directories it hands to external tools; and (3) a lint and an
editable-install CI job that keep the rule true.


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

### Changes

**Code**

1. `torch._utils_internal.resource(*parts)`, the single entry point for
   reading package data. It returns
   `importlib.resources.files("torch").joinpath(*parts)`, a stdlib
   `Traversable`. (Rule 1.)
2. `torch._utils_internal.get_file_path()` anchored on the import system
   instead of `__file__`: the editable loader's `paths`, then the spec of
   `torch._C`, then `__file__`. (Rule 2; pytorch/pytorch#195587.)
3. The known call sites converted.
   * Locations, through `get_file_path`: `torch.utils.cmake_prefix_path`, the
     inductor MPS header path and `torch_openreg` (pytorch/pytorch#195587;
     `torch.mps.compile_shader` was done in pytorch/pytorch#195295), the
     remaining inductor sites (pytorch/pytorch#195889), `_nvshmem_triton`
     (pytorch/pytorch#195890), and `find_library_location` with two test-side
     lookups (pytorch/pytorch#195891).
   * Resources, through `resource`: `torch.utils.model_dump`,
     `torch._export.serde.schema_check`, the inductor `codegen/aoti_runtime/`
     sources, `kernel/flex/templates/`, `kernel/templates/` and
     `csrc/inductor/aoti_runtime/model.h`.
4. `torchgen` gets the same two helpers: `torchgen.gen.get_torchgen_root()`
   re-anchored, and a resource entry point for `packaged/`.

**CI**

5. An editable-install job (pytorch/pytorch#195888), so the editable layout
   is exercised on every PR.
6. A lint: no `__file__`-derived paths and no direct `importlib.resources`
   calls in `torch/` outside `torch._utils_internal`.

**Documentation**

7. An "Install layouts" page holding the three layouts, the table and the
   two rules.
8. Links to it from `CONTRIBUTING.md`, `docs/libtorch.rst` and
   `docs/cpp/source/installing.md`, where each describes, or fails to
   describe, its own layout.
9. A "Where things end up" paragraph in `CONTRIBUTING.md`.

Landing order: 2 and 3 are in flight; 5 follows so the editable layout is
exercised on every PR; 1, 4 and 6 once the inventory is at zero; 7 to 9 as
soon as the helpers are the answer they document. Items 1 and 2 require the
`build-system.requires` floor at scikit-build-core 1.1.1, which carries the
install-first `__path__` order and the loader's `paths` attribute
(scikit-build/scikit-build-core#1566, #1567).

### The three layouts

What matters to code is where the compiled artifacts sit relative to the
Python sources, not which command ran. There are three answers.

**Wheel layout: artifacts beside the sources.** `_C` sits next to
`torch/__init__.py`, with `lib/`, `include/`, `bin/` and `share/` in the same
package directory. `__file__`-relative lookups are correct. Reached by
installing a published wheel; by `pip install .` from a checkout, which builds
a wheel under PEP 517 and installs it, so "installed from source" is still
this layout; and by installing a locally built wheel, as the CD pipelines do.

**Editable layout: artifacts split from the sources.** Reached by
`pip install -e .`, which scikit-build-core performs in redirect mode: "Python
files come from the source tree, CMake-built artifacts from the build tree"
(`pyproject.toml`). The `.py` files are imported from the checkout; everything
CMake installs (`include/`, `lib/`, `bin/`, `share/`, `_C`) is staged beside
the installed distribution in site-packages. The checkout's `torch/lib` holds
only the tracked libshm sources and the checkout has no `torch/include`.
`torch.__path__` names both trees, install tree first.
`__file__`-relative lookups resolve into the checkout and are wrong.

**Standalone layout: artifacts without Python sources.** Reached by a
`cmake --install` that scikit-build-core does not drive. With
`BUILD_PYTHON=OFF`, `torch/CMakeLists.txt` returns before defining
`torch_python`, so neither `_C` nor `libtorch_python` exists. The libtorch
zips are this layout as shipped: on Linux and macOS `include/`, `share/`,
`lib/`, `build-version`, `build-hash`, no `bin/`, no Python files. The
Windows zip follows the platform convention instead, DLLs in `bin/` and
`.lib` import libraries in `lib/`, so the Windows zip and the Windows wheel
(DLLs in `torch/lib/`) are different layouts of the same binaries.
`TorchConfig.cmake` absorbs that; path-deriving code would not. The two
commands that produce this layout today, the `cmake` recipe in
`docs/libtorch.rst` and `tools/build_libtorch.py`, install more than the zips
carry and, in the script's case, into the checkout's `torch/` directory,
where the artifacts collide with an editable install. Closing that gap is
split out below.

**Modifier: python-only (`BUILD_LIBTORCHLESS`, forced on by
`BUILD_PYTHON_ONLY=ON` in `cmake/EnvVarForwarding.cmake`).** Not a layout. It
applies to the wheel and editable layouts and changes contents, not
locations: `lib/` holds `torch_python` and `libshm` but no `libtorch`, `c10`
or `torch_cpu`; `include/` is partial; `share/cmake/` lacks
`Torch/TorchConfig.cmake`. No such wheel is published and no CI exercises it
(pytorch/pytorch#159990). It is recorded so that code probing for an
installed tree assumes no more than it provides.

### What each layout puts under `torch/`

From listings of the 2.14.0 CPU wheels and libtorch zips for Linux, macOS
and Windows, editable installs on the same three platforms, and a
`BUILD_PYTHON_ONLY=ON` configure. The per-category counts, the consumer
groups and the CMake rule behind each directory are in
[`RFC-0060-assets/artifact-inventory.md`](RFC-0060-assets/artifact-inventory.md);
[`RFC-0060-assets/list-wheel-contents.py`](RFC-0060-assets/list-wheel-contents.py)
regenerates the wheel and zip listings for any release.

| directory | wheel | editable, installed tree in site-packages | editable, checkout `torch/` | standalone, libtorch zip | python-only modifier |
|---|---|---|---|---|---|
| `lib/` | torch's own libraries plus the bundled runtime (`libgomp`, `libomp`, `libiomp5md` + `libuv` by platform); Windows adds `.lib` import libraries | **superset of the wheel**: everything the wheel has, plus the static libraries, vendored `cmake/` and `pkgconfig/` that the wheel exclude list removes | libshm **sources** only (tracked) | Linux, macOS: the libraries, no `torch_python`. **Windows: DLLs live in `bin/`, `lib/` holds the `.lib` import libraries** | `torch_python`, `libshm`; no libtorch, c10, torch_cpu |
| `bin/` | `torch_shm_manager` + `protoc` on Linux and macOS; `protoc.exe` only on Windows | same as wheel | absent | Linux, macOS: absent. Windows: the DLLs | `torch_shm_manager`, `protoc` |
| `include/` | full (about 9,500 headers) | full, plus files the wheel excludes | absent | full | **partial** |
| `share/` | `cmake/{Torch,ATen,Caffe2}` | same, plus third-party CMake files the wheel excludes | absent | `cmake/{Torch,ATen,Caffe2}` | `cmake/{ATen,Caffe2,...}`; **no `Torch/TorchConfig.cmake`** |
| `_C` extension | beside `__init__.py` | in the installed tree, served by scikit-build-core's redirect loader | absent | absent | beside `__init__.py` |

Two notes bear on the rule:

* **The editable tree is a strict superset of the wheel's native tree.**
  `[tool.scikit-build.wheel].exclude` trims third-party residue from the
  wheel but not from the editable staging tree. A probe that keys on those
  files, or on file counts, misclassifies an editable install; a probe on
  `lib/` does not.
* **Some Python files exist only because the build made them.** One class
  is generated into the checkout (gitignored) and installed, so it exists in
  both trees under the editable layout: `torch/version.py`, the generated
  `.pyi` stubs, `annotated_fn_args.py`, and the CUDA and ROCm specific
  modules. The other exists in the install tree only: the mirrored cutedsl
  kernel package and `torchgen/packaged/`. Imports of both resolve
  deterministically, because the editable finder maps every CMake-installed
  module to the install tree. `importlib.resources` on the first class
  resolves to the install tree's copy, which `__path__` lists first. Full
  list in the inventory asset.

The Linux wheels additionally ship about 150 MB of C++ test artifacts under
`test/`, `bin/` and `lib/`, because `BUILD_TEST` defaults to ON and only the
macOS and Windows CD jobs turn it off. A CD configuration gap, out of scope
here.

### The resolution rule

Code inside `torch` must not derive the path of a compiled artifact or a data
file from `__file__`, which is correct only when sources and artifacts share
a directory. Two kinds of lookup exist and need different mechanisms. Both go
through `torch._utils_internal`, one entry point each, sharing the anchor
logic.

**Rule 1: resources are read through `resource(*parts)`, which is
`importlib.resources` underneath.** A resource is a file whose contents the
code consumes: templates, schemas, JSON, web assets, a header read as text,
the Metal shader headers, most of the package data CMake mirrors into the
package. `importlib.resources.files("torch")` works in every layout with a
Python package; under the editable layout scikit-build-core's resource reader
spans the install tree and the checkout, install tree first, and a file in
either resolves to a concrete path. The helper returns the stdlib
`Traversable` unchanged. Callers open, read or iterate it; they do not turn
it into a path. The helper exists to give the lint one allowed call site and
to hide which multiplexed-path class is in play (scikit-build-core's own on
Python 3.10 and 3.11, the stdlib's from 3.12).

**Rule 2: locations are resolved through `get_file_path("torch", ...)`.** A
location is a directory handed to something outside the interpreter: the
include directory for a compiler, `lib/` for the `ctypes` preload and
`os.add_dll_directory`, `share/cmake` for `torch.utils.cmake_prefix_path`,
`bin/` for `torch_shm_manager`. `importlib.resources` cannot provide these: a
directory that exists in both trees stays multiplexed, on which
`os.fspath()` raises, and `torch/lib` is a tracked source directory, so under
the editable layout it multiplexes on every platform. `as_file()` supports
directories only from Python 3.12, by copying the tree into a temporary
directory. The helper returns a filesystem path; `torch.utils.cmake_prefix_path`,
`torch.utils.cpp_extension` and every in-tree location lookup go through it.

The type difference is deliberate: a `Traversable` for resources, a path for
locations. A caller that needs a path for a resource is asking for a
location and uses rule 2.

**Both helpers anchor on the import system, in this order:**

1. **The editable loader's `paths`.** scikit-build-core 1.1 gives every
   redirect loader a `paths` attribute listing the package's search locations
   in `__path__` order, install tree first, next to the existing
   `__loader__.rebuild()` hook. Absent in a wheel install, so the chain
   continues.
2. **The extension module's spec.**
   `importlib.util.find_spec("torch._C").origin` answers "where will the
   extension be imported from" in every layout that has one: the install tree
   under the editable layout (verified on Linux and Windows), beside
   `__init__.py` for the wheel.
3. **`__file__`**, only when `find_spec("torch._C")` returns nothing. Not the
   standalone layout, which never imports `torch`, but an interpreter without
   a discoverable spec for the extension, such as a frozen or embedded build.

One constraint: **probe for an installed tree with `lib/`**, never `bin/`,
`include/` or `share/cmake/Torch`. `bin/` is empty under MSVC without
vendored protobuf, `include/` is partial under the python-only modifier,
`TorchConfig.cmake` is absent there, and the editable tree is a superset of
the wheel's.

**`torchgen` is a second anchor with the same two rules.** It is a separate
top-level package that must not import `torch` (stage-1 codegen runs before
torch is built), so it cannot use `torch._utils_internal`. Its data,
`torchgen/packaged/` (the ATen yaml, the templates, the autograd files), is
gitignored and mirrored by `cmake/FileMirroring.cmake` straight into the
install tree, so under the editable layout it exists only there.
`files("torchgen")` spans both trees and finds it;
`torchgen.gen.get_torchgen_root()`, the public location helper documented for
out-of-tree users, returns `Path(__file__).parent` today and so points at the
checkout, where `packaged/` is absent. It gets the anchor chain above with
`torchgen` in place of `torch`. `gen_backend_stubs` prefers the checkout's
`aten/src` and is correct as is.

**Enforcement.** With one allowed call site per rule, the lint is a grep:
flag `__file__`-derived paths and direct `importlib.resources` calls (the
`files()` and the functional `read_text()` forms alike) in `torch/` outside
`torch._utils_internal`. It backs the first metric.

### Placement

Proposed: **one page**, "Install layouts", holding the three layouts, the
table and the two rules, linked from `CONTRIBUTING.md`, `docs/libtorch.rst`
and `docs/cpp/source/installing.md` where each describes, or fails to
describe, its own layout. The problem is conflation between layouts; one
page that shows them side by side is the direct remedy, and it gives the
rules a single home.

The alternative is a split by audience: the editable layout and the rules in
`CONTRIBUTING.md`, the wheel tree, standalone install and zips in the C++
docs. Each audience already looks in its own place, but the comparison,
which is the content, gets written twice or not at all. The split stays an
option for the discussion; either outcome leaves the rest unchanged.

### Split out

The inventory also showed that the documented standalone recipe and
`tools/build_libtorch.py` produce more than the zips carry: `libtorch_python`
and the C++ test binaries (`BUILD_PYTHON` and `BUILD_TEST` default to ON),
`bin/protoc` with the vendored protobuf's static library and CMake files
(`protobuf_INSTALL` is never turned off), and, for the script, an install
prefix inside the checkout that collides with an editable install. The fixes
are three build changes: the recipe sets `BUILD_PYTHON=OFF` and
`BUILD_TEST=OFF`, with a configure-time message when `BUILD_PYTHON` is on in
a configure that scikit-build-core does not drive; `cmake/ProtoBuf.cmake`
sets `protobuf_INSTALL=OFF`; and the script's default prefix moves out of the
source tree, with the Windows arm64 build script adapted and a CI job that
asserts the standalone tree. They change what the C++ audience receives and
have their own reviewers, so they follow as PRs that cite the page. The
evidence, the zip-versus-recipe diff and the five in-tree callers of the
script, is in the inventory asset.


## **Metrics**

* Zero in-tree `__file__`-derived asset lookups and zero direct
  `importlib.resources` calls outside the helper module; the
  pytorch/pytorch#195887 inventory reaches zero and the lint keeps it there.
* An editable-install CI job (pytorch/pytorch#195888) exists and stays green.
* Every layout is described in exactly one place; every other place links
  there.


## **Drawbacks**

* Documentation of the trees goes stale when install rules change.
  Mitigation: the inventory asset names the CMake rule behind each directory
  and the listing script regenerates the file lists behind the counts, so a
  rule change points at the row and a release refreshes the numbers.
* The rule adds an indirection in front of `importlib.resources` for
  resources. Mitigation: the helper is a one-line alias; its value is the
  single call site the lint can check.


## **Alternatives**

* **Do nothing.** The seven call sites get fixed one by one and the next
  contributor writes the eighth.
* **Collapse the layouts**, e.g. make redirect mode install artifacts into
  the checkout so the editable layout becomes the wheel layout. Rejected: it
  reintroduces the in-tree artifact problem the migration removed and does
  nothing for the standalone layout.
* **Document only the editable layout** in `CONTRIBUTING.md`. It is the
  layout that broke, but the conflation is between layouts.
* **Use `importlib.resources` directly, no helpers.** It stays the
  implementation for resources, but it cannot be the only call site: it
  cannot produce a directory for an external tool (a directory in both trees
  has no single path, and `as_file()` copies), and a lint needs one place to
  allow.
* **Split the documentation by audience.** See Placement.


## **Prior Art**

The ecosystem documents the editable *mechanism* and its *limitations*, not
the resulting trees, and leaves asset resolution to each project. Checked
2026-09-09:

* **meson-python** ("Editable installs") documents its loader-based redirect
  and states rule 1's consequence: data files "need to be accessed using
  `importlib.resources`", because a `__file__`-relative read "would fail when
  the package is installed in editable mode".
* **numpy** ("Building from source"): "editable installs are fundamentally
  incomplete installs. Their only guarantee is that `import numpy` works".
  `numpy.get_include()` is the precedent for rule 2's shape, a function, and
  for the trap: it joins `os.path.dirname(numpy.__file__)` with the include
  subdirectory, with a "running from numpy source directory" branch, which is
  the `__file__` heuristic that breaks under a split layout and the reason for
  the warning.
* **setuptools**, **scipy** and **scikit-build-core** carry the same warning
  in their own words; none shows what a project ends up with on disk.

None documents the layouts, and none has PyTorch's second requirement:
handing directories to compilers and loaders at runtime as a user-facing
feature (`torch.compile`, `cpp_extension`, `cmake_prefix_path`). Hence a
location rule with a supported anchor rather than numpy's warning.


## **How we teach this**

* Name the layouts consistently ("wheel layout", "editable layout",
  "standalone layout") and say which command produced a tree: `pip install .`
  is the wheel layout, `pip install -e .` the editable layout,
  `cmake --install` and `tools/build_libtorch.py` the standalone layout.
* Two rules, one sentence each: read resources through
  `torch._utils_internal.resource(...)`, `importlib.resources` underneath;
  resolve locations through `get_file_path("torch", ...)`. Never `__file__`,
  and never a path from a resource.
* `CONTRIBUTING.md` gains a short "Where things end up" paragraph in its
  editable-install section: the layout after `pip install -e .`, the two
  rules in a sentence each, and a link to the page. It states no tree of its
  own, so it cannot drift.


## **Unresolved questions**

* **Placement.** One page is proposed; the split by audience stays on the
  table.
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
