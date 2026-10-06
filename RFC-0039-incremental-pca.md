# [Implementation of Incremental PCA]

**Authors:**
* @sirluk


## **Summary**
Implement a class for Incremental PCA in vanilla PyTorch with native GPU support.


## **Motivation**
- Incremental PCA is important where data arrives in streams or is too large to fit in memory.
- PyTorch currently lacks a built-in implementation of incremental PCA, and the best available implementations, like `sklearn`, do not support have GPU support.
- Requested in issue tracker: [issue40770](https://github.com/pytorch/pytorch/issues/40770)


## **Proposed Implementation**
The proposed implementation would be similar to the one in `sklearn` ([sklearn.decomposition.IncrementalPCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html)) but with the following differences:
- torch.tensor objects instead of numpy arrays
- GPU support
- Option to use truncated SVD via torch.svd_lowrank

Draft implementation can be found [here](https://github.com/sirluk/pytorch_incremental_pca). Potentially it could make sense to inherit from torch.nn.Module and define internal state as buffers.


## **Metrics **
- Ability to handle datasets larger than available RAM
- GPU vs. CPU performance comparison
- Accuracy comparison with sklearn's IncrementalPCA

## **Drawbacks**
- Implementation could be relatively self-contained, no additional dependencies required. Can reuse existing linear algebra functions in pytorch. Draft implementation already created.
- Tests required to ensure numerical stability accross various data types and sizes.


## **Alternatives**
What other designs have been considered?
- Use scikit-learn's Incremental PCA by moving data to CPU. Very slow.
What is the impact of not doing this?
- Users might resort to less efficient workarounds or third-party libraries
- Potential migration of users to other tools for specific PCA-heavy workflows

## **Prior Art**
- Implemented in sklearn: [sklearn.decomposition.IncrementalPCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html)
- Large parts of the sklearn implementation can be reused


## **How we teach this**
- "IncrementalPCA" or "IPCA" for the main class name. Clearly indicates the incremental nature of the algorithm
- Use terms like "partial_fit" for incremental updates.
- Emphasize use cases: streaming data, large datasets, online learning
- Demonstrate GPU speedup with benchmarks
- Provide code examples demonstrating integration with PyTorch datasets and dataloaders

## **Unresolved questions**
- Where in the pytorch codebase should this live?
- What should the API look like?
- Should this be implemented as a torch.nn.Module or should it be a standalone class?
- Should the device be implicit (inferred from the input tensor) or explicit?
- Should this work with multiple GPUs out of the box?
Out of scope for this RFC:
- Multi-GPU support

## Resolution
We decided to do it. X% of the engineering team actively approved of this change.

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
Some people were in favor of it, but some people didn’t want it for project X.


### Next Steps
Will implement it. 


#### Tracking issue
<github issue URL>


#### Exceptions
Not implementing on project X now. Will revisit the decision in 1 year.
