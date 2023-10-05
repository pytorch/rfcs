This page contains instructions on how to propose and implement feature changes to PyTorch.

## Table of Contents

- [Introduction](#introduction)
- [Proposing a Feature to PyTorch](#proposing-a-feature-to-pytorch)
  - [Step 1: Create an RFC](#step-1-create-an-rfc)
  - [Step 2: Get Feedback on the RFC](#step-2-get-feedback-on-the-rfc)
  - [Step 3: Implement your Feature](#step-3-implement-your-feature)
- [Implementing an RFC](#implementing-an-rfc)
- [RFC Rejection](#rfc-rejection)
- [Inspiration](#inspiration)
- [License](#license)

## Introduction
Welcome to the PyTorch Feature Proposal and Implementation Guide!

This guide provides comprehensive instructions on how to propose and implement feature changes within the PyTorch ecosystem. Whether you're a seasoned developer or a newcomer, understanding the process of introducing new features and improvements to PyTorch is essential for contributing to this dynamic and rapidly evolving framework.

In this document, you will find step-by-step guidance on how to submit a Request For Comments (RFC), which serves as a design proposal for your proposed changes. RFCs offer a structured way to share detailed descriptions of the features you wish to introduce, explain their necessity, and outline your proposed implementation approach.

We emphasize the importance of initiating discussions and refining your ideas during the ideation phase rather than waiting until the pull request (PR) phase. Engaging with PyTorch's core maintainers early on allows for valuable feedback, alignment with existing efforts, and the possibility of making your feature more widely applicable.

While smaller changes, including bug fixes and documentation enhancements, can follow the standard GitHub pull request workflow on the main PyTorch repository, RFCs are specifically designed for more extensive design proposals that require in-depth discussion and deliberation.

If you're uncertain whether your proposed change should follow the RFC process or be submitted as a feature-request issue, we encourage you to open an issue in the main PyTorch/PyTorch repository for clarification.

Throughout this guide, we'll walk you through the three primary steps of proposing and implementing a PyTorch feature: creating an RFC, gathering feedback, and implementing your feature. We'll also explain how to navigate the implementation phase, track progress, and seek assistance when needed.

Remember that your contributions, whether they lead to an RFC approval or not, play a vital role in the evolution of PyTorch. We appreciate your dedication to improving this open-source project and look forward to your valuable contributions.

Let's dive into the process of shaping the future of PyTorch together!

# Proposing a Feature to Pytorch

To propose a new feature, you’ll submit a Request For Comments (RFC).  This RFC is basically a design proposal where you can share a detailed description of what change you want to make, why it’s needed, and how you propose to implement it.

It’s easier to make changes while your feature is in the ideation phase vs the PR phase, and this doc gives core maintainers an opportunity to suggest refinements before you start code.  For example, they may know of other planned efforts that your work would otherwise collide with, or they may suggest implementation changes that make your feature more broadly usable.

Smaller changes, including bug fixes and documentation improvements can be
implemented and reviewed via the normal GitHub pull request workflow on the main PyTorch repo.

RFCs are more suitable for design proposals that are too large to discuss on a feature-request issue, 
like adding a new abstraction, or if a discussion about the tradeoffs involved in a new addition are non-trivial.

If you are unsure whether something should be an RFC or a feature-request issue, you can ask by 
opening an issue in the main PyTorch/PyTorch repository.

# The Request for Comments

## Step 1: Create an RFC
RFCs are located in their own repository.  

To create one:

1. For the https://github.com/pytorch/rfcs repository
2. Copy the template file `RFC-0000-template.md` to `RFC-00xx-your-feature.md` and fill it out with your proposal. The template is a guideline, feel free to add sections as appropriate
3. You may also have the template simply link to another editor, like a Google Docs file, but please ensure that the document is publicly visible.  This can make the template easier to add edit, but commenting doesn’t scale very well, so please use this option with caution.

## Step 2: Get Feedback on the RFC
1. Submit a pull request titled `RFC-00xx-your-feature.md`
2. Before your PR is ready for review, give it the draft label.
3. Once it’s ready for review, remove the draft label and give it the `commenting` label
4. File an issue against the https://github.com/pytorch/pytorch repository to review your proposal.
5. In the description, include a short summary of your feature and a link to your RFC PR
6. Pytorch Triage review will route your issue to core contributors with the appropriate expertise.
7. Build consensus. Those core contributors will review your PR and offer feedback. Revise your proposal as needed until everyone agrees on a path forward. Additional forums you can share the proposal on include the [developer forum](https://dev-discuss.pytorch.org/c/rfc-chatter), and the [Slack channel](https://bit.ly/ptslack). Tagging interested stakeholders (identifiable via [CODEOWNERS](https://github.com/pytorch/pytorch/blob/master/CODEOWNERS)) can help with consensus building.

_(Note: A proposal may get rejected if it comes with unresolvable drawbacks or if it’s against the long term plans of the pytorch maintiners)_

## Step 3: Implement your Feature
1. If your RFC PR is accepted, you can merge it into the [pytorch/rfcs](https://github.com/pytorch/rfcs) repository and begin working on the implementation.
2. When you submit PRs to implement your proposal, remember to link to your RFC to help reviewers catch up on the context.



## Implementing an RFC
Every accepted RFC has an associated issue tracking its implementation in the PyTorch repository; thus that
associated issue can be assigned a priority via the triage process that the team uses for all issues.

The author of an RFC is not obligated to implement it. Of course, the RFC
author (like any other developer) is welcome to post an implementation for
review after the RFC has been accepted.

If you are interested in working on the implementation for an accepted RFC, but
cannot determine if someone else is already working on it, feel free to ask
(e.g. by leaving a comment on the associated issue).


## RFC Rejection
Some RFC pull requests are tagged with the "shelved" label when they are
closed (as part of the rejection process). An RFC closed with "shelved" is
marked as such because we want neither to think about evaluating the proposal
nor about implementing the described feature until some time in the future, and
we believe that we can afford to wait until then to do so. 

## Inspiration
PyTorch's RFC process owes inspiration to the [Rust RFC Process](https://github.com/rust-lang/rfcs) and [React RFC Process](https://github.com/reactjs/rfcs/), and the [Artsy RFC process](https://github.com/artsy/README/blob/main/playbooks/rfcs.md#resolution) for the resolution template.

## License
By contributing to rfcs, you agree that your contributions will be licensed under the LICENSE file in the root directory of this source tree.
