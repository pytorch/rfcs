
# [CUDA version support]

**Authors:**
* @atalman @malfet

## **Motivation**

The proposal is to provide two main benefits

- Aligned decision policy provides transparent decision making;
- Incorporate decision points in the early release process to stagger CUDA migration from the RC release integration work.

## **Proposed Policy and Process**

### We should introduce new version of CUDA when

- It enables new important GPU architecture (For example Blackwell with CUDA-12.8)
- It brings significant performance improvements
- It fixes significant correctness issues
- It significantly reduces binary/memory footprint

### We would deprecate version of CUDA when

As soon as we introduce a new Experimental Version we should consider moving the previous Experimental Version to Stable, and decommission the previous Stable version. Typically we want to support 2-3 versions of CUDA as follows:

- Optional Legacy Version: If we need to have 1 version for backend compatibility or to work around the current limitation. For example: CUDA older driver is incompatible with newer CUDA version
- Stable Version: This is a stable CUDA version that is used most of the time. This is the version we want to upload to PyPI. Please note that if the stable version is equal or slightly older to the version then the version fbcode is using, we probably need to keep it in CI/CD system as Optional Legacy Version.
- Latest Experimental Version: This is the latest version of CUDA that we want to support. Minimal requirement is to have it available in nightly releases (CD)

### Detailed Process of Introducing new CUDA version

1. Evaluate CUDA update necessity
  Goal: When any of this is true:
  - It enables new important GPU architecture (For example Blackwell with CUDA-12.8)
  - It brings significant performance improvements
  - It fixes significant correctness issues
  - It significantly reduces binary/memory footprint

2. Evaluate if we have all packages for update
  When: As soon as Update determined to be necessary. Start by creating RFC [issue](https://github.com/pytorch/pytorch/issues/145544) with possible CUDA matrix to support for next release.
  Goal: Make sure everything is available to perform complete upgrade of CUDA and dependencies

3. Update CUDA in CD (this is necessary condition to qualify for CUDA version be released as Experimental)
  When: Evaluate if we have all packages for update is complete
  Goal: Make sure all binaries Linux/Windes on wheels and conda platforms are produced on nightly

4. Update CUDA in CI
  When: Update CUDA in CD is complete
  Goal: Make sure CI jobs tests on new CUDA versions, all failing tests should be identified and reported

5. Run benchmarks to compare new experimental CUDA version to current stable CUDA version
  When: Update CUDA in CD  is complete
  Goal: Make sure we identify all performance regressions


### Detailed Process of Deprecating CUDA version

1. Evaluate deprecation of legacy CUDA version from CI/CD
When: We completed CUDA update and previous experimental CUDA version is qualified to be stable and we have 3 supported versions (legacy, stable and experimental). Start by creating RFC [issue](https://github.com/pytorch/pytorch/issues/147383) to deprecate legacy CUDA version from CI/CD.
Goal: Make sure we support 2 versions of CUDA, supporting 3 versions can be an exception for certain release where we need to keep legacy version

2. Deprecate legacy CUDA version from CI/CD
When: Evaluate deprecation of legacy CUDA version from CI/CD is complete
Goal: Support for legacy CUDA versions is dropped, starting from PyTorch Domain Libraries and then in PyTorch core. First we drop CD support and then CI support.


# CUDA/CUDNN Upgrade Runbook

So you wanna upgrade PyTorch to support a new CUDA? Follow these steps in order! They are adapted from previous CUDA upgrade processes.

## 0. Before starting upgrade process

### A. Currently Supported CUDA and CUDNN Libraries
Here is the supported matrix for CUDA and CUDNN (versions can be looked up in [here](https://github.com/pytorch/pytorch/blob/main/.ci/docker/common/install_cudnn.sh) . Architecture support can be found [here](https://github.com/pytorch/pytorch/blob/main/.ci/manywheel/build_cuda.sh))

| CUDA | CUDNN | architectures supported | additional details |
| --- | --- | --- | --- |
| 11.8.0 | 9.1.0.70 | Kepler(3.7), Maxwell(5.0), Pascal(6.0), Volta(7.0), Turing(7.5), Ampere(8.0, 8.6), Hopper(9.0) | Legacy CUDA Release |
| 12.6.3 | 9.5.1.17 | Maxwell(5.0), Pascal(6.0), Volta(7.0), Turing(7.5), Ampere(8.0, 8.6), Hopper(9.0)  | Stable CUDA Release |
| 12.8.0 | 9.7.1.26 | Turing(7.5), Ampere(8.0, 8.6), Hopper(9.0), Blackwell(10.0, 12.0+PTX)  | Latest CUDA Release |
|        | 9.8.0.87 | Turing(7.5), Ampere(8.0, 8.6), Hopper(9.0), Blackwell(10.0, 12.0+PTX)  | Latest CUDA Nightly |

### B. Check the package availability

Package availability to validate before starting upgrade process :

1) CUDA and CUDNN is available for Linux and Windows:
https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run (x86)
https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux_sbsa.run (aarch64)
https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/

2) CUDA is available on Docker hub images : https://hub.docker.com/r/nvidia/cuda
   Following example is for cuda 12.4: https://gitlab.com/nvidia/container-images/cuda/-/tree/master/dist/12.4.0/ubuntu2204/devel?ref_type=heads (TODO: Update this for 12.8)
   (Make sure to use version without CUDNN, it should be installed separately by install script)

3) Validate new driver availability: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html. Check following table: Table 3. CUDA Toolkit and Corresponding Driver Versions


## 1. Maintain Progress and Updates
Make an issue to track the progress, for example [#56721: Support 11.3](https://github.com/pytorch/pytorch/issues/56721). This is especially important as many PyTorch external users are interested in CUDA upgrades.

## 2. Modify scripts to install the new CUDA for Manywheel Docker Linux containers.
There are two types of Docker containers we maintain in order to build Linux binaries: `libtorch`, and `manywheel`. They all require installing CUDA and then updating code references in respective build scripts/Dockerfiles.  This step is about manywheel.

1. Follow this [PR 145567](https://github.com/pytorch/pytorch/pull/145567) for all steps in this section
2. Find the CUDA install link [here](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&=Debian&target_version=10&target_type=runfile_local)
3. Get the cudnn link from NVIDIA on the PyTorch Slack
4. Modify [`install_cuda.sh`](common/install_cuda.sh) and [`install_cuda_aarch64.sh`](common/install_cuda_aarch64.sh)
5. Run the `install_128` chunk of code on your devbox to make sure it works.
6. Modify [`build-manywheel-images.yml`](.github/workflows/build-manywheel-images.yml) with the latest CUDA version 12.8 in this case.
7. To test that your code works, from the root builder repo, run something similar to `export CUDA_VERSION=12.8 && .ci/docker/manywheel/build_scripts/build_docker.sh` for the `manywheel` images.
8. Once the PR in step1 is merged, validate manylinux docker hub [manylinux2_28-builder:cuda12.8](https://hub.docker.com/r/pytorch/manylinux2_28-builder/tags?name=12.8) and [manylinuxaarch64-builder:cuda12.8](https://hub.docker.com/r/pytorch/manylinuxaarch64-builder/tags?name=12.8) to see that images have been built and correctly tagged. These images are used in the next step to build Magma for linux.

## 3. Update Magma for Linux
Build Magma for Linux. Our Linux CUDA docker images require magma build, so we need to build magma-cuda<version> and push it to the ossci-linux s3 bucket:
1. The code to build Magma is in the [`pytorch/pytorch` repo](https://github.com/pytorch/pytorch/tree/main/.ci/magma)
2. Currently, this is mainly copy-paste in [`magma/Makefile`](magma/Makefile) if there are no major code API changes/deprecations to the CUDA version. Previously, we've needed to add patches to MAGMA, so this may be something to check with NVIDIA about.
3. To push the package, please update [build-magma-linux workflow](https://github.com/pytorch/pytorch/blob/main/.github/workflows/build-magma-linux.yml)
4. NOTE: This step relies on the `pytorch/manylinux2_28-builder:cuda${DESIRED_CUDA}-main` image (changes to [`.github/workflows/build-manywheel-images.yml`](https://github.com/pytorch/pytorch/blob/7d4f5f7508d3166af58fdcca8ff01a5b426af067/.github/workflows/build-manywheel-images.yml#L52)), so make sure you have pushed the new manywheel-builder prior.

## 4. Modify scripts to install the new CUDA for Libtorch Docker Linux containers. Modify builder supporting scripts
There are two types of Docker containers we maintain in order to build Linux binaries: `libtorch`, and `manywheel`. They all require installing CUDA and then updating code references in respective build scripts/Dockerfiles.  This step is about libtorch containers.

Add setup for our Docker `libtorch`:
1. Follow this PR [PR 145789](https://github.com/pytorch/pytorch/pull/145789) for all steps in this section. For `libtorch`, the code changes are usually copy-paste.
2. Merge the above the PR, and it should automatically push the images to Docker Hub with GitHub Actions. Make sure to update the `cuda_version` to the version you're adding in respective YAMLs, such as `.github/workflows/build-libtorch-images.yml`.
3. Verify that the workflow that pushes the images succeed by selecting and verifying them in the [Actions page](https://github.com/pytorch/pytorch/actions/workflows/build-libtorch-images.yml). Furthermore, check [https://hub.docker.com/r/pytorch/libtorch-cxx11-builder/tags](https://hub.docker.com/r/pytorch/libtorch-cxx11-builder/tags) to verify that the right tags exist for libtorch types of images.

## 5. Generate new Windows AMI, test and deploy to canary and prod.

Please note, since this step currently requires access to corporate AWS, this step should be performed by Meta employee. To be removed, once automated. Also note that Windows AMI takes about a week to build, so start this step early.
1. For Windows you will need to rebuild the test AMI, please refer to this [PR](https://github.com/pytorch/test-infra/pull/6243). After this is done, run the release of Windows AMI using this [proecedure](https://github.com/pytorch/test-infra/tree/main/aws/ami/windows). As time of this writing this is manual steps performed on dev machine. Please note that packer, aws cli needs to be installed and configured!
2. After step 1 is complete and new Windows AMI have been deployed to AWS. We need to deploy the new AMI to our canary environment (https://github.com/pytorch/pytorch-canary) through https://github.com/fairinternal/pytorch-gha-infra example : [PR](https://github.com/fairinternal/pytorch-gha-infra/pull/31) . After this is completed Submit the code for all windows workflows to https://github.com/pytorch/pytorch-canary and make sure all test are passing for all CUDA versions.
3. After that we can deploy the Windows AMI out to prod using the same pytorch-gha-infra repository.

## 6. Modify code to install the new CUDA for Windows and update MAGMA for Windows

1. Follow this [windows Magma and cuda build for cu128](https://github.com/pytorch/pytorch/pull/146653/files) for all steps in this section
2. To get the CUDA install link, just like with Linux, go [here](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local) and upload that `.exe` file to our S3 bucket [ossci-windows](https://s3.console.aws.amazon.com/s3/buckets/ossci-windows?region=us-east-1&tab=objects).
3. Review "Table 3. Possible Subpackage Names" of CUDA installation guide for windows [link](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) to make sure the Subpackage Names have not changed. These are specified in [cuda_install.bat file](https://github.com/pytorch/pytorch/pull/146653/files#diff-0b30eff7a5006465b01be34be60b1b109cf93fb0996de40613a319de309f40db)
4. To get the cuDNN install link, you could ask NVIDIA, but you could also just sign up for an NVIDIA account and access the needed `.zip` file at this [link](https://developer.nvidia.com/rdp/cudnn-download). First click on `cuDNN Library for Windows (x86)` and then upload that zip file to our S3 bucket.
5. NOTE: When you upload files to S3, make sure to make these objects publicly readable so that our CI can access them!
6. If you have to upgrade the driver install for newer versions, which would look like [updating the `windows/internal/driver_update.bat` file](https://github.com/pytorch/pytorch/blob/main/.ci/pytorch/windows/internal/driver_update.bat)
    1. Please check the CUDA Toolkit and Minimum Required Driver Version for CUDA minor version compatibility table  in [the release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) to see if a driver update is necessary.
7. Compile MAGMA with the new CUDA version. Update [`.github/workflows/build-magma-windows.yml`](https://github.com/pytorch/pytorch/pull/146653/files#diff-613791f266f2f7b81148ca8f447b0cd6c6544f824f5f46a78a2794006c78957b) to include new version.
8. Validate Magma builds by going to S3 [ossci-windows](https://s3.console.aws.amazon.com/s3/buckets/ossci-windows?region=us-east-1&tab=objects). And querying for ```magma_```


## 7. Add the new CUDA version to the nightly binaries matrix.
Adding the new version to nightlies allows PyTorch binaries compiled with the new CUDA version to be available to users through `pip` or just raw `libtorch`.
1. If the new CUDA version requires a new driver (see #1 sub-bullet), the CI and binaries would also need the new driver. Find the driver download [here](https://www.nvidia.com/en-us/drivers/unix/) and update the link like [so](https://github.com/pytorch/pytorch/commit/fcf8b712348f21634044a5d76a69a59727756357).
    1. Please check the Driver Version table in [the release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) to see if a driver update is necessary.
2. Follow this [Add CUDA 12.8 manywheel x86 Builds to Binaries Matrix](https://github.com/pytorch/pytorch/pull/145792/files) for steps 2-4 in this section.
3. Once [PR 145792](https://github.com/pytorch/pytorch/pull/145792/files) is created make sure to attach ciflow/binaries, ciflow/nightly labels to this PR. And make sure all the new workflow with new CUDA version terminate successfully.
4. Testing nightly builds is done as follows:
    - Make sure your commit to master passed all the test and there are no failures, otherwise the next step will not work
    - Make sure your changes are promoted to viable/strict branch: https://github.com/pytorch/pytorch/tree/viable/strict . Run viable/strict promotion job to promote from master to viable/strict
    - After your changes are promoted to viable/strict. Run nighly build job.
    - Make sure your changes made to nightly branch https://github.com/pytorch/pytorch/tree/nightly
    - Make sure all nightly build succeeded before continuing to Step #6
5. If Stable CUDA version changes, update latest tag for ghcr.io like so: https://github.com/pytorch/pytorch/blob/main/.github/scripts/generate_binary_build_matrix.py#L20

## 8. Add the new CUDA version to OSS CI.
Testing the new version in CI is crucial for finding regressions and should be done ASAP along with the next step (I am simply putting this one first as it is usually easier).
1. The configuration files will be subject to change, but usually you just have to replace an older CUDA version with the new version you're adding. **Code reference for 12.6**: [PR 140793](https://github.com/pytorch/pytorch/pull/140793/files).
2. IMPORTANT NOTE: the CI is not always automatically triggered when you edit the workflow files! Ensure that the new CI job for the new CUDA version is showing up in the PR signal box.
If it is not there, make sure you add the correct ciflow label (ciflow/periodic, for example) to trigger the test. Just because the CI is green on your pull request does NOT mean
the test has been run and is green.
3. It is likely that there will be tests that no longer pass with the new CUDA version or GPU driver. Disable them for the time being, notify people who can help, and make issues to track them (like [so](https://github.com/pytorch/pytorch/issues/57482)).
4. After merging the CI PR, Please open temporary issues for new builds and tests marking them unstable, example [issue](https://github.com/pytorch/pytorch/issues/127104). These issues should be closed after few days of opening, when newly added CI jobs are constantly green.

## 9. Update Linux Nvidia driver used during runner provisioning
If linux driver update is required. The driver should be updated during the runner provisioning otherwise nightly workflows will fail with multiple Nova workflows.
1. Post and merge [PR 5243](https://github.com/pytorch/test-infra/pull/5243)
2. Run workflow [lambda-release-tag-runners workflow](https://github.com/pytorch/test-infra/actions/workflows/lambda-release-tag-runners.yml) this worklow will create new release [here](https://github.com/pytorch/test-infra/releases)
3. Post and merge [PR 394](https://github.com/pytorch-labs/pytorch-gha-infra/pull/394)
4. Deploy this change by running following workflow [runners-on-dispatch-release](https://github.com/pytorch-labs/pytorch-gha-infra/actions/workflows/runners-on-dispatch-release.yml)

## 10. Add the new version to torchvision and torchaudio CI.
Torchvision and torchaudio is usually a dependency for installing PyTorch for most of our users. This is why it is important to also
propagate the CI changes so that torchvision and torchaudio can be packaged for the new CUDA version as well.
1. Add a change to a binary build matrix in test-infra repo [here](https://github.com/pytorch/test-infra/blob/main/tools/scripts/generate_binary_build_matrix.py#L29)
2. A code sample for torchvision: [PR 7533](https://github.com/pytorch/vision/pull/7533)
3. A code sample for torchaudio: [PR 3284](https://github.com/pytorch/audio/pull/3284)
You can combine all above three steps in one PR: [PR 6244] (https://github.com/pytorch/test-infra/pull/6244/files)
4. Almost every change in the above sample is copy-pasted from either itself or other existing parts of code in the
builder repo. The difficulty again is not changing the config but rather verifying and debugging any failing builds.

This completes CUDA and CUDNN upgrade. Congrats! PyTorch now has support for a new CUDA version and you made it happen!

## Upgrade CUDNN version only

If you require to update CUDNN version for already existing CUDA version, please perform the followin modifications.
1. Add new cudnn vesion to windows AMI: https://github.com/pytorch/test-infra/pull/6290. Rebuild and retest the AMI. Follow step 6  Generate new Windows AMI, test and deploy to canary and prod.
2. Add new cudnn version to linux builds: https://github.com/pytorch/pytorch/pull/148963/files (including installation script and small wheel update)
