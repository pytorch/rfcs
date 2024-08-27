# A PyTorch conda "distribution"

|            |                 |
| ---------- | --------------- |
| Authors    | Ralf Gommers    |
| Status     | Rejected        |
| Type       | Process         |
| Created    | 2020-11-26      |

This proposal addresses the need for a PyTorch conda distribution, meaning a
collection of integration-tested packages that can be installed from a single
channel, to enable package authors to release packages that depend on PyTorch
and let users install them in a reliable way.


## Motivation and Scope

For developers of libraries that depend on PyTorch, it is currently (Nov'20)
quite difficult to express that dependency in a way that makes their package
easily installable with `conda` (or `pip`) by end users. With the PyTorch
ecosystem growing and the dependency graphs of sets of packages users use in
a single environment becoming more complex, streamlining the package
distribution and installation experience is important.

Examples of packages for which there's interest in making them more easily
available to end users:

- [fastai](https://docs.fast.ai/): Jeremy Howard expressed interest, and
  plans to copy `pytorch` and other dependencies of fastai over to the `fastai`
  channel in case this proposal doesn't work out.
- [fairseq](https://github.com/pytorch/fairseq): a fairseq developer inquired
  about being added to the `pytorch` channel
  [here](https://github.com/pytorch/builder/issues/563), and a conda-forge
  contributor wanted to package both PyTorch and fairseq in conda-forge, see
  [here](https://github.com/conda-forge/pytorch-cpu-feedstock/issues/7#issuecomment-688467743).
- [TorchANI](https://github.com/aiqm/torchani): see a TorchANI user's recent
  attempt to add a conda-forge package
  [here](https://github.com/conda-forge/torchani-feedstock/pull/1).

In scope for this proposal are:

- Processes related to adding new packages to the `pytorch` conda channel.
- CI infrastructure needed for integration testing and moving already built
  packages to the `pytorch` channel.

_Note: using the `pytorch` channel seems like the most obvious choice for a
single integration channel; using a new channel is also possible, it won't
change the rest of this proposal materially._

Out of scope are:

- Changes related to how libraries are built or packages for conda are created.
- Updating PyTorch packaging in `defaults` or `conda-forge`.
- Improvements to installing with pip or wheel builds.


### The current state of affairs

PyTorch is packaged in the `pytorch` channel; users must either add that
channel to the channels list globally or in an environment (using, e.g.,
`conda config --env --add channels pytorch`), or add `-c pytorch` to every
`conda` command they run. Note that the channels method is preferred over `-c
pytorch` but installation instructions invariably use the latter, which can
lead to problems when it's forgotten by the user at some point.

PyTorch is also packaged in `defaults`, but it's really outdated (1.4.0 for
CUDA-enabled packages, 1.5.0 for CPU-only). The `conda-forge` channel doesn't
have PyTorch packages - there's a desire to add them, however it's unclear if
and how that will happen.

Authors of _pure Python packages_ tend to use their own conda channel to
distribute their own package. Installation instructions will then have both
the `pytorch` and their own channel in them. For example for fastai and
BoTorch:

```
conda install -c fastai -c pytorch fastai
```

```
conda install botorch -c pytorch -c gpytorch
```

When a user needs multiple packages, that becomes unwieldy quickly with each
package adding its own channel. Note: alternatively, pure Python packages can
choose to distribute on PyPI only (see the _PyPI, pip and wheels_ section
further down) - Kornia is an example of a package that does this.

Authors of _packages containing C++ or CUDA code_ which use the PyTorch C++
API have an additional issue: they need to release new package versions in
sync with PyTorch itself, because there's no stable ABI that would allow
depending on multiple PyTorch versions. For example, the torchvision
`install_requires` dependency is determined like:

```python
pytorch_dep = 'torch'
if os.getenv('PYTORCH_VERSION'):
    pytorch_dep += "==" + os.getenv('PYTORCH_VERSION')

requirements = [
    'numpy',
    pytorch_dep,
]
```
and its build script ensure a one-to-one correspondence of `pytorch` and
`torchvision` versions of packages.

The `pytorch` channel currently already contains other packages that depend
on PyTorch. Those fall into two categories: needed dependencies (e.g.,
`magma-cuda`, `ffmpeg`) , and PyTorch-branded and Facebook-owned projects
like `torchvision`, `torchtext`, `torchaudio`, `captum`, `faiss`, `ignite`, etc.
See https://anaconda.org/pytorch/repo for a complete list.

Those packages maintain their own build and packaging scripts (see
[this comment](https://github.com/pytorch/builder/issues/563#issuecomment-722667815)),
and the integration testing and uploading to the `pytorch` conda channel is done
via scripts in the [pytorch/builder](https://github.com/pytorch/builder) repo.

There's more integration testing happening already:
- The `test_community_repos/` directory in the `builder` repo contains a
  significantly larger set of packages that's tested in addition to the packages
  that are distributed on the `pytorch` conda channel.
- The [pytorch-integration-testing](https://github.com/pytorch/pytorch-integration-testing)
  repo contains tooling to test PyTorch release candidates.
- An overview of integration test results from the `builder` repo (last updated Oct'19,
  so perhaps no longer maintained) can be found
  [here](https://web.archive.org/web/20201222195552/http://ossci-integration-test-results.s3-website-us-east-1.amazonaws.com/test-results.html).


## Usage and Impact

### End users

The intended outcome for end users is that they will be able to install many
of the most commonly packages easily with `conda` from a single channel,
e.g.:

```
conda install pytorch torchvision kornia fastai mmf -c pytorch
```

or, a little more complete:

```
# Use a new environment for a new project
conda create -n myenv
conda activate myenv
# Add channel to env, so all conda commands will now pick up packages
# in the pytorch channel:
conda config --env --add channels pytorch
conda install pytorch torchvision kornia fastai mmf
```

### Maintainers of packages depending on PyTorch

The intended outcome for maintainers is that:

1. They have clear documentation on how to add their package to the `pytorch` channel,
   including the criteria their packages should meet, how to run integration tests,
   and how to release new versions.
2. They can declare their dependencies correctly
3. They will still need their own channel or some staging channel to host packages
   before they get `anaconda copy`'d to the `pytorch` channel.
4. They can provide a single install command to their users, `conda install mypkg -c pytorch`,
   that will work reliably.


## Processes

### Proposing a new package for inclusion

Prerequisites for a package being considered for inclusion in the `pytorch` channel are:

1. The package naturally belongs in the PyTorch ecosystem. I.e., PyTorch is a
   key dependency, and the package is focused on an area like deep learning,
   machine learning or scientific computing.
2. All runtime dependencies of the package are available in the `defaults` or
  `pytorch` channel, or adding them to the `pytorch` is possible with a
  reasonable amount of effort.
3. A working recipe for creating a conda package is available.

A GitHub repository (working name `conda-distro`) will be used for managing
proposals for new packages as well as integration configuration and tooling.
To propose a new package, open an issue and fill out the instructions in the
GitHub issue template. When a maintainer approves the request, the proposer
can open a PR to that same repo to add the package to the integration
testing.


### Integration testing infrastructure

The CI connected to the `conda-distro` repo has to do the following:

1. Trigger on PRs that add or update an individual package, running the tests
   for that package _and_ downstream dependencies of that package.
2. If tests for (1) are successful, sync the conda packages in question to
   the `pytorch` channel with `anaconda copy`.
3. Provide a way to run the tests of all packages together.
4. Send notifications if a package releases requires an update (e.g. a
   version bump) to a downstream package.

The individual packages have to do the following:

1. Ensure there are _upper bounds on dependency versions_, so new releases of
   PyTorch or another dependency cannot break already released versions of
   the individual package in question. Note that that does mean that a new
   PyTorch releases requires version bumps on existing packages - more detail
   in strategy will be needed here.
2. Tests for a package should be _runnable in a standardized way_, via
   `conda-build --test`. This is easy to achieve via either a `test:` section
   in the recipe (`meta.yaml`) or a `run_test.py` file. See [this section of
   the conda-build docs](https://docs.conda.io/projects/conda-build/en/latest/resources/define-metadata.html#test-section)
   for details. An advantage of this method is that `conda-build` is already
   aware of channels and dependencies, so it should work with very little
   extra effort.


### What happens when a new PyTorch release is made?

For minor or major versions of PyTorch, new releases of downstream packages
will also be necessary. A number of packages, such as `torchvision`,
`torchaudio` and `torchtext`, are anyway released in sync. Other packages in
the `pytorch` channel may need to be manually released via a PR to the
`conda-distro` repo).

Version constraints should be set such that a bugfix release of PyTorch does
not require any new downstream package releases.


### Dealing with packages that aren't maintained

Proposing a package for inclusion in the `pytorch` channel implies a
commitment to keep maintaining the package. There wil be a place to list one
or more maintainers for each package so they can be pinged if needed. In case
a package is not up-to-date or broken and it does not get fixed, after a
certain duration (length TBD) it may be removed from the channel.


## Alternatives

### Conda-forge

The main alternative to making the `pytorch` channel an integration channel
that distributes many packages that depend on PyTorch is to have a
(GPU-enabled) PyTorch package in conda-forge, and tell users and package
authors that that is the place to go. It will require working with
conda-forge in order to ensure that the `pytorch` package is of high quality,
either by copying over the binaries from the `pytorch` channel or by
migrating recipes and keeping them in sync. See
[this very long discussion](https://github.com/conda-forge/pytorch-cpu-feedstock/issues/7)
for details (and issues).

Advantages of this alternative are:

- Conda-forge has a lot of packages, so it will be easier to install PyTorch
  in combination with other non-deep learning packages (e.g. the geo-science
  stack).
- Conda-forge already has established tools and processes for adding and
  updating them. Which means it's less likely for there to be issues with
  dependencies (e.g. packages with many or unusual dependencies may not be
  accepted into the `pytorch` channel, while `conda-forge` will be fine with
  them).
- Users are likely already familiar with using the `conda-forge` channel.

Disadvantages of this alternative are:

- As of today, conda-forge doesn't have GPU hardware. Building is stil
  possible using CUDA stubs, however testing cannot really be done inside CI,
  only manually (which is a pain, especially when having to test multiple
  hardware and OS platforms).
  _Note that there are packages that follow this approach (mostly without
  problems so far), for example `arrow-cpp` and `cupy`. To obtain a full list of packages, clone https://github.com/conda-forge/feedstocks and run
  `grep 'compiler(' feedstocks/*/meta.yaml | grep cuda`._
- `conda-forge` and `defaults` aren't guaranteed to be compatible, so
  standardizing on `conda-forge` may cause problems for people who prefer
  `defaults`.
- Exotic hardware support may be difficult. PyTorch has support for TPUs (via
  XLA), AMD ROCm, Linux on ARM64, Vulkan, Metal, Android NNAPI - this list
  will continue to grow. Most of this is experimental and hence not present
  in official binaries (and/or in the C++/Java packages which aren't
  distributed with conda), but this is likely to change and present issues
  with compilers or dependencies not present in conda-forge.
  For more details, see [this comment by Soumith](https://github.com/conda-forge/pytorch-cpu-feedstock/issues/7#issuecomment-538253388).
- Release coordination is more difficult. For a PyTorch release, packages for
  `pytorch`, `torchvision`, `torchtext`, `torchaudio` will all be built
  together and then released. There may be manual quality assurance steps
  before uploading the packages.
  Building a set of packages like that depend on each other and releasing
  them in a coordinated fashion is hard to do on conda-forge, given that if
  everything is in feedstocks, the new pytorch package must already be
  available before the next build can start. It may be possible to do this
  with channel labels (build sequentially, then move all packages to the
  `main` label at once), but either way all the released artifacts will be
  publicly visible before the official release.

Other points:

- If the PyTorch team does not package for conda-forge, someone else will do
  that at some point.
- Conda-forge no longer uses a single compiler toolchain for all packages it
  builds for a given platform - it is now possible to use a newer compiler,
  which itself is built with an older glibc/binutils (that does need to be
  common). See
  [this example](https://github.com/conda-forge/omniscidb-feedstock/blob/master/recipe/conda_build_config.yaml)
  for how to specify using GCC 8. So not having a recent enough compiler
  available is unlikely to be a relevant concern.
- Mirroring packages in the `pytorch` channel to the `conda-forge` channel
  would alleviate worries about the disadvantages here, however there's no
  conda-forge tooling currently to verify ABI compatibility of the packages,
  which is the main worry of the conda-forge team with this approach.


### DIY for every package

Letting authors of every package depending on PyTorch find their own solution
is basically the status quo of today. The most likely outcome longer-term is
that PyTorch plus those packages depending on it will be packaged in
conda-forge independently. At that point there are two competing `pytorch`
packages, one in the `pytorch` and one in the `conda-forge` channel. And
users who need a prebuilt version of other packages not available in the
`pytorch` channel will likely migrate to `conda-forge`.

The advantage is: no need to do any work to implement this proposal. The
disadvantage is: depending on PyTorch will remain difficult for downstream
packages.


## Related work and issues

### Conda channels

Mixing multiple conda channels is rarely a good idea. It isn't even
completely clear what a channel is for, opinions of conda and conda-forge
maintainers differ - see
https://github.com/conda-forge/conda-forge.github.io/issues/883.


### RAPIDS

RAPIDS has a really complex setup for distributing conda packages. Its install instructions currently look like:
```
conda create -n rapids-0.16 -c rapidsai -c nvidia -c conda-forge \
    -c defaults rapids=0.16 python=3.7 cudatoolkit=10.1
```

Depending on a user's config (e.g. having `channel_priority: strict` in
`.condarc`), this may not work even in a clean environment. If one would add
the `pytorch` channel as well, for users that need both PyTorch and RAPIDS,
it's even less likely to work - the conda solver cannot handle that many
channels and will fail to find a solution.


### Cudatoolkit

CUDA libraries are distributed for conda users via the `cudatoolkit` package.
That package is only available in the `nvidia`, `defaults` and `conda-forge`
channels. The license of the package prohibits redistribution, and an
exception is difficult to obtain. Therefore it should not be added to the
`pytorch` channel (also not necessary, obtaining it from `defaults` is fine).


### PyPI, pip and wheels

The experience installing PyTorch with `pip` is suboptimal, mainly because
there's no way to control CUDA versions via `pip`, so the user gets whatever
the default CUDA version is (10.2 at the time of writing) when running `pip
install torch`. In case the user needs a different CUDA version or the
CPU-only package, the install instruction looks like:
```
pip install torch==1.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
There's the [pytorch-pip-shim](https://github.com/pmeier/pytorch-pip-shim)
tool to handle auto-detecting CUDA versions and retrieving the right wheel.
It relies on monkeypatching pip though, so it may break when new versions of
pip are released.

For package authors wanting to add a dependency on PyTorch, the above
usability issue is a serious problem. If they add a runtime dependency on
PyTorch (via `install_requires` in `setup.py` or via `pyproject.toml`), the
only thing they can add is `torch` and there's no good way of signalling to
the user that there's a CUDA version issue or how to deal with it.

Finally note that `pip` and `conda` work together reasonably well, so for
package authors that want to release packages that _do not contain C++ or
CUDA code_, releasing on PyPI only and telling their users to install PyTorch
with `conda` and their package with `pip` will work best. As soon as C++/CUDA
code gets added, that's no longer reliable though.


## Effort estimate

TODO

### Initial setup


### Ongoing effort

