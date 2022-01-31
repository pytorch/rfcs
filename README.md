# PyTorch RFCs

The RFC (request for comments) is a structured document that allows community members 
to propose an idea to everyone before it is implemented. RFCs enable stakeholders 
to be aware and confident about the direction PyTorch is evolving in.

Many changes, including bug fixes and documentation improvements can be
implemented and reviewed via the normal GitHub pull request workflow on the main PyTorch repo.

RFCs are more suitable for design proposals that are too large to discuss on a feature-request issue, 
like adding a new abstraction, or if a discussion about the tradeoffs involved in a new addition are non-trivial.

If you are unsure whether something should be an RFC or a feature request issue, 
please open an issue in the main PyTorch repository first to discuss.


## What the process is
In short, to get a major feature added to PyTorch, one must first get the RFC
merged into the RFC repository as a markdown file. At that point the RFC is
"active" and may be implemented with the goal of eventual inclusion into Rust.
 
- Fork the rfcs repo: https://github.com/pytorch/rfcs
- Copy `RFC-0000-template.md` to `RFC-0000-my-feature.md`, or write your own open-ended proposal. Put care into the details.
- Submit a pull request titled `RFC-OOxx-my-feature`. Assign the `commenting` label on the PR to open it for discussions. 
- Build consensus for your proposal, integrate feedback and revise it as needed, and summarize the outcome of the discussion via a [resolution template](https://github.com/pytorch/rfcs/blob/rfc-process/RFC-0000-template.md#resolution).
    - If the RFC is idle here (no activity for 2 weeks), assign the label `stalled` to the PR.
- Once the discussion has settled, assign a new label based on the level of support:
    - `approved` if a decision has been made in the RFC
    - `draft` if the author needs to rework the RFC’s proposal
    - `postponed `if there are no plans to move ahead with the current RFC’s proposal
- A state of `approved` means that the core team has agreed in principle to the proposal, and it is ready for implementation. 
- The author (or any interested developer) should next open a tracking issue on Github corresponding to the RFC.
    - This tracking issue should contain the implementation next steps ([example](https://github.com/pytorch/pytorch/issues/50344)).
    - Link to this tracking issue on the RFC (in the Resolution > Next Steps section)
- Once all relevant PRs are merged, the RFC’s status label can be finally updated to `closed`.



## Reviewing RFCs
[Reviewing RFCs]: #reviewing-rfcs

While the RFC pull request is up, the sub-team may schedule meetings with the
author and/or relevant stakeholders to discuss the issues in greater detail,
and in some cases the topic may be discussed at a sub-team meeting. In either
case a summary from the meeting will be posted back to the RFC pull request.

A sub-team makes final decisions about RFCs after the benefits and drawbacks
are well understood. These decisions can be made at any time, but the sub-team
will regularly issue decisions. When a decision is made, the RFC pull request
will either be merged or closed. In either case, if the reasoning is not clear
from the discussion in thread, the sub-team will add a comment describing the
rationale for the decision.


## Implementing an RFC
[Implementing an RFC]: #implementing-an-rfc

Some accepted RFCs represent vital features that need to be implemented right
away. Other accepted RFCs can represent features that can wait until some
arbitrary developer feels like doing the work. Every accepted RFC has an
associated issue tracking its implementation in the Rust repository; thus that
associated issue can be assigned a priority via the triage process that the
team uses for all issues in the Rust repository.

The author of an RFC is not obligated to implement it. Of course, the RFC
author (like any other developer) is welcome to post an implementation for
review after the RFC has been accepted.

If you are interested in working on the implementation for an "active" RFC, but
cannot determine if someone else is already working on it, feel free to ask
(e.g. by leaving a comment on the associated issue).


## RFC Postponement
[RFC Postponement]: #rfc-postponement

Some RFC pull requests are tagged with the "postponed" label when they are
closed (as part of the rejection process). An RFC closed with "postponed" is
marked as such because we want neither to think about evaluating the proposal
nor about implementing the described feature until some time in the future, and
we believe that we can afford to wait until then to do so. Historically,
"postponed" was used to postpone features until after 1.0. Postponed pull
requests may be re-opened when the time is right. We don't have any formal
process for that, you should ask members of the relevant sub-team.

Usually an RFC pull request marked as "postponed" has already passed an
informal first round of evaluation, namely the round of "do we think we would
ever possibly consider making this change, as outlined in the RFC pull request,
or some semi-obvious variation of it." (When the answer to the latter question
is "no", then the appropriate response is to close the RFC, not postpone it.)



## License
By contributing to rfcs, you agree that your contributions will be licensed under the LICENSE file in the root directory of this source tree.
