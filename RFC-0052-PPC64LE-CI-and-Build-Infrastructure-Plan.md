# Phased Enablement of Power (ppc64le) CI in PyTorch

**Authors:**
* Sandeep Gupta
* IBM Power team

## **Summary**

This RFC describes a phased approach for enabling Power (ppc64le) CI testing and build infrastructure in PyTorch. The plan incrementally introduces upstream CI coverage—starting from fork validation, moving to upstream on-demand pull-request testing, and eventually enabling nightly builds—while minimizing risk and operational overhead.

The goal is to make Power CI visible upstream while keeping ownership, operations, and infrastructure fully vendor-managed by IBM.


## **Motivation**

PyTorch is widely used across multiple hardware architectures, including Power. Today, CI testing and build validation for Power systems are largely maintained outside of upstream PyTorch CI, typically in downstream or vendor-managed forks.

Bringing Power CI upstream provides:

* Earlier detection of architecture-specific regressions
* Increased confidence for users running PyTorch on Power systems
* Reduced long-term maintenance cost of downstream forks
* Alignment with PyTorch's multi-architecture support goals

At the same time, introducing new architecture coverage into upstream CI must be done carefully to avoid instability, unexpected CI noise, or community maintenance burden. This RFC proposes a measured, incremental plan to reach consensus on scope and execution.

### Current Status

As of today:

* ✅ Power build and test jobs are validated on a fork
* ✅ CI workflows and ciflow/ppc64le labels have been updated and validated
* ✅ Jobs are passing in controlled environments
* ❌ No Power CI coverage exists upstream in pytorch/pytorch
* ❌ No nightly artifacts are published for Power

Technical feasibility has been established; upstream integration and governance are the next steps.


## **Proposed Implementation**

Power CI support will be introduced through incremental CI roles, each with clearly scoped responsibilities, triggers, and expectations.

### Guiding Principles

* Start with non-blocking, opt-in CI execution
* Expand scope only after stability and signal quality are demonstrated
* Avoid introducing new merge or release gating
* Keep ownership and operational responsibility explicit
* All Power CI jobs discussed below run on IBM-operated, vendor-managed runner infrastructure

### Technical Implementation: Runners and Infrastructure

This section summarizes how Power CI runners are implemented and operated. The design follows the same general model used for other vendor-managed architectures, with IBM retaining full ownership and operational responsibility.

#### Docker Images Overview

Power CI uses two distinct container images, each serving a separate purpose:

1. **Ephemeral Runner Image**
2. **PyTorch Build and Test Image**

##### Ephemeral Runner Image

The ephemeral runner image is used to dynamically provision GitHub self-hosted runners.

This image contains:
* GitHub Actions runner binaries
* Runner registration and teardown logic
* Minimal system dependencies required for runner initialization

Additional characteristics:
* Maintained outside of the pytorch/pytorch repository
* Owned and operated by the IBM Power team
* Runners register at job start and deregister on completion

##### PyTorch Build and Test Image

A separate container image is used for building and testing PyTorch on Power (ppc64le).

Details:
* Defined by the Dockerfile located at: `.ci/docker/manywheel/Dockerfile_ppc64le`

Includes:
* Required system dependencies for Power
* PyTorch build toolchains
* Runtime and test dependencies used by CI

The Dockerfile and related configuration are versioned directly in the pytorch/pytorch repository as part of the Power CI changes. Keeping the build image definition in-repo ensures the Power build environment is transparent, reviewable, and aligned with upstream CI expectations.

#### Image Maintenance and Updates

Both images are maintained by the IBM Power team.

Updates are performed when:
* Base OS packages require updates
* Build or test dependencies change
* Toolchain updates are needed

Changes to the PyTorch build image are validated via fork-based CI before being used in upstream workflows.

#### Runner Provisioning and Lifecycle

* Runner provisioning, registration, and teardown are handled by IBM-managed automation outside of the PyTorch repository
* Runners are provisioned on demand and destroyed after job completion
* IBM is responsible for runner capacity planning, scaling, and health

#### Integration with Upstream PyTorch

* Runners are integrated with pytorch/pytorch using the existing PyTorch GitHub App, consistent with the s390 setup
* CI jobs are triggered via standard GitHub Actions workflows
* Power CI is opt-in and label-driven (ciflow/ppc64le)

As PyTorch CI infrastructure evolves (e.g., GHARTS when available), the Power CI integration may be aligned with those mechanisms without changing ownership or execution semantics.

### Runner Roles, Triggers, and Jobs

#### Phase 1: Tester – Fork Validation ✅ (Completed)

**Role:** Tester

**Triggers:** Manual execution on forks only

**Jobs:**
* Build PyTorch (core)
* Execute tests as defined by the Power CI workflow

**Purpose:** Validate CI workflows, runner configuration, and correctness without impacting upstream PyTorch CI.

#### Phase 2: Tester – On-Demand PR Testing (Upstream, Non-Blocking)

**Role:** Tester

**Triggers:**
* Manual, opt-in execution on pull requests via the ciflow/ppc64le label
* Not enabled by default on all PRs

**Jobs:**
* Power CI workflow defined in `.github/workflows/ppc64le.yml`
* Build PyTorch
* Execute tests as defined by the workflow

**Infrastructure:**
* Jobs run exclusively on IBM-managed, vendor-provided runner pools

**Characteristics:**
* Non-blocking
* Failures reported for visibility only
* Intended to validate upstream CI integration and detect regressions

**Current Implementation:**
* Initial upstream enablement implemented in: Add on-demand ppc64le wheel build support #173519

**Purpose:**
* Validate Power CI behavior on real upstream PRs
* Collect reliability and signal-quality data before introducing scheduled jobs

#### Phase 3: Publisher – Nightly Builds

**Role:** Publisher

**Triggers:** Scheduled nightly workflows

**Jobs:**
* Build and publish Power nightly artifacts
* Update nightly build workflows to include Power
* Update common reusable CI workflows in alignment with PyTorch CI design guidelines and patterns

**Infrastructure:**
* All jobs run on IBM-operated, vendor-managed runner pools

**Benefits:**
* Enables broader downstream and user testing on Power systems
* Provides early signal for architecture-specific issues
* Aligns Power CI with existing PyTorch CI workflow design

**Prerequisite:**
* Demonstrated stability and reliability from Phase 2 on-demand PR testing

### Non-Goals (Initial Phases)

* Making Power CI a required merge gate for pull requests
* Including Power in official release blocking criteria
* Achieving full test-matrix parity with tier-1 architectures

These may be revisited after sustained stability is demonstrated.


## **Metrics**

Key metrics to measure the value and health of Power CI:

* **CI Reliability:** Success rate of Power CI jobs over time
* **Signal Quality:** Rate of true positives vs. false positives in failure detection
* **Coverage:** Number of PRs tested with Power CI (via ciflow/ppc64le label)
* **Adoption:** Usage of nightly Power builds by downstream users
* **Response Time:** Time to triage and resolve Power-specific CI failures


## **Drawbacks**

Are there any reasons why we should not do this? Here we aim to evaluate risk and check ourselves.

Please consider:
* is it a breaking change?
* Impact on UX
* implementation cost, both in terms of code size and complexity
* integration of this feature with other existing and planned features

**Mitigations:**
* Start with non-blocking, opt-in execution
* Label-based, on-demand triggering
* Explicit IBM-only ownership and maintenance
* Clear escalation path for Power-specific issues


## **Alternatives**

What other designs have been considered? What is the impact of not doing this?


## **Prior Art**

* **s390x CI Integration:** PyTorch already supports s390x architecture with vendor-managed CI, providing a proven model for Power CI
* **ARM64 CI:** Similar phased approach used for ARM64 architecture support

## **How we teach this**

* **Documentation:** Update PyTorch CI documentation to include Power CI workflows and label usage
* **Developer Guide:** Provide clear instructions on when and how to use the `ciflow/ppc64le` label
* **CI Dashboard:** Ensure Power CI results are visible in existing CI dashboards and HUD
* **Communication:** Announce Power CI availability through PyTorch dev mailing list and community channels


## **Unresolved questions**

* Long-term criteria for expanding Power CI coverage beyond Phase 3
* Conditions for any future release-tier consideration
* Integration timeline with GHARTS or other evolving CI infrastructure

These topics are intentionally deferred until upstream signal is available from Phase 2 and Phase 3 implementation.


## **Ownership and Maintenance**

The Power CI and build infrastructure is fully owned and maintained by IBM.

**IBM responsibilities include:**
* Runner provisioning and capacity management
* CI job reliability, debugging, and failure triage
* Workflow and configuration maintenance
* Build and test infrastructure upkeep

There is no expectation for the PyTorch community to operate, debug, or maintain the Power CI infrastructure.


## Resolution

*This section will be completed after RFC review and approval.*

### Level of Support

Choose one of the following:
* 1: Overwhelming positive feedback.
* 2: Positive feedback.
* 3: Majority Acceptance, with conflicting Feedback.
* 4: Acceptance, with Little Feedback.
* 5: Unclear Resolution.
* 6: RFC Rejected.
* 7: RFC Rejected, with Conflicting Feedback.


#### Additional Context

*To be filled after community discussion.*


### Next Steps

1. Gather feedback on this RFC from CI Infra and stakeholders
2. Iterate on Phase 2 based on upstream PR signal
3. Introduce nightly build workflows for Power when Phase 2 stabilizes
4. Reassess scope and coverage based on observed results


#### Tracking issue

*To be created after RFC acceptance.*


#### Exceptions

*None at this time.*