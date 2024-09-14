

<details>
<summary>Instructions - click to expand</summary>

- Fork the rfcs repo: https://github.com/pytorch/rfcs
- Copy `RFC-0000-template.md` to `RFC-00xx-my-feature.md`, or write your own open-ended proposal. Put care into the details.
- Submit a pull request titled `RFC-00xx-my-feature`. 
    - Assign the `draft` label while composing the RFC. You may find it easier to use a WYSIWYG editor (like Google Docs) when working with a few close collaborators; feel free to use whatever platform you like. Ideally this document is publicly visible and is linked to from the PR.
    - When opening the RFC for general discussion, copy your document into the `RFC-00xx-my-feature.md` file on the PR and assign the `commenting` label.
- Build consensus for your proposal, integrate feedback and revise it as needed, and summarize the outcome of the discussion via a [resolution template](https://github.com/pytorch/rfcs/blob/master/RFC-0000-template.md#resolution).
    - If the RFC is idle here (no activity for 2 weeks), assign the label `stalled` to the PR.
- Once the discussion has settled, assign a new label based on the level of support:
    - `accepted` if a decision has been made in the RFC
    - `draft` if the author needs to rework the RFC’s proposal
    - `shelved` if there are no plans to move ahead with the current RFC’s proposal. We want neither to think about evaluating the proposal
nor about implementing the described feature until some time in the future.
- A state of `accepted` means that the core team has agreed in principle to the proposal, and it is ready for implementation. 
- The author (or any interested developer) should next open a tracking issue on Github corresponding to the RFC.
    - This tracking issue should contain the implementation next steps. Link to this tracking issue on the RFC (in the Resolution > Next Steps section)
- Once all relevant PRs are merged, the RFC’s status label can be finally updated to `closed`.

</details>





# Pytorch-DataLoader-Economic

**Authors:**
* @yoadbs

## **Summary**
A new pytorch dataloader multiprocessing pipline is suggested. This pipline is designated to significantly reduce random-access-memory (RAM) usage, without any significant reduction in throughput (TPT).

## **Motivation**
Model input batch may require significant amounts of RAM. For example, in video processing or in 3D graphics applications.

By current dataloader multiprocessing pipline design, workers simultaneously prepere batches and send them into shared memory, using a queue.
In practice, about [num_workers] batches are simultenously stored in shared memory, nearly after epoch start. 
At most, [num_workers * prefetch_factor] may be stored in shared memory at the same time.
The main process operates in parallel, to extract one batch after another, and inject it into the model for training/validation/test. 

Storing about [num_workers] batches in shared memory, at the same time, imposes a limit over [num_workers]:\
[num_workers < SERVER_RAM_AVAILABLE_BYTES / BATCH_SIZE_BYTES]\
This limitation can produce a bottleneck over training TPT, not allowing to increase num_workers, due to server's RAM limitations.
Alternatively, severs with more RAM can be used, to increase num_workers, increaseing severs cost.

A new dataloader multiprocessing pipeline is suggested.
In this pipline the amount of batches sent into shared memory is not dependant in [num_workers].
This decoupling, allowes to increase [num_workers] without any significant increase in RAM consumption. 
As in current implemnation, workers are not expected to enter idle state during the epoch, hence no TPT reduction is expected for the same num_workers.
The new flow is designated to reduce RAM related bottelnecks and/or requirements, and improve training costeffectiveness.

## **Proposed Implementation**
### **Definitions**

| symbol               | description                                                                                                                                                  |
|----------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| iw                   | items_worker (there are num_workers workers)                                                                                                                 |
| bw                   | batch_worker                                                                                                                                                 |
| index_queue[iw]      | A queue for each items_worker - used to send items index (and metadata) to item_workers. Main process is putting data, and items_worker[iw] is gettting data |
| item_queue[ib]       | One queue for each batch_worker - used to retrive items from item_workers. All items workers are putting data, batch_worker[ib] is getting data              |
| worker_result_queue  | One queue - used to send prepared batches back to main process. All batches workers are putting data, main process is getting data                           |
| item_idx             | Item serial index from epoch start (0 for first item, 1 for next item, etc)                                                                                  |
| batch_idx            | Batch serial index from epoch start (0 for first batch, 1 for next batch, etc)                                                                               |
| item_index           | Item's dataset index, as in dataset.__getitem__(index)                                                                                                       |
| iw_idx               | Item_worker index (which item_worker is designated to process the item)                                                                                      |  
| bw_idx               | Batch_worker index (which batch_worker is designated to process the item)                                                                                    |


By current design, the class _MultiProcessingDataLoaderIter has one level of [num_workers] workers. 
The main process sends [prefetch_factor] batches to each worker, by index_queues.
Each worker prepares the batch, and send it back to the main process through worker_result_queue.
After a batch is retrived by the main process, another batch is sent to the appropriate worker.

A new design for MultiProcessingDataLoaderIter class is suggested. In the suggested design, there are 2 levels of workers: 
* item_workers - Designated to generate one item at a time (by running dataset \_\_getitem__ function), and send it to shared memory 
  * This worker is similar to the workers of the current design, but it recieves and sends one item at a time (and not one batch at a time) 
* batch_workers - Designated to get items from shared memory, collect [batch_size] items, run collate function, and send the prepared batch back to shared memory

By the new design, data will flow by the following order: main_process -> item_workers -> batch_workers -> main_proces

### **main process high-level flow**
* Send one item at a time to item_workers (by index_queues)
  * Each item should include the following data: (item_idx, batch_idx, item_index, iw_idx, bw_idx):
  * Track number of items at work ("work-load") for each worker.  
    * A different iw_idx should be assigned to each item
      * Select iw_idx of the items_worker with the minimal work-load
    * An identical bw_idx should be assigned to all items in the same batch
      * Select bw_idx of the batches_worker with the minimal work-load
    * Make sure that the sum of item_workers work-load is always <= [prefetch_factor] * [batch_size]. Stop sending items when reaching this limit.
* Retrive and store prepared batches from batch_workers (by worker_result_queue)
  * Make sure to reduce work-load for the relevant batch_worker and for each relevant batch_worker when retriving the batch
* Once the next required batch is retrived (by , return batch to caller function 

### **items_worker main-loop flow**
* get item from index_queue
* run dataset.\_\_getitem__(item_index)
* send item to batch_worker by item_queue[bw_idx]

### **batches_worker main-loop flow**
* get items from item_queue
* Once all items of a given batch are recived, run collate_fn and send the prepared batch to worker_result_queue

### **Notes**
* A new dataloader parameter: num_batch_workers should be introduced
  * By default, this parameter should be set to prefetch_factor. 
  * There is no reason to use a larger value than prefetch_factor. However, smaller value may be considered by the user, if collate_fn is very fast

## **Metrics **
For similar configuration, the new flow should require significantly less shared memory, while preserving TPT.
To monitor shared memory usage, type in linux server terminal: \
$ monitor -n0.1 df -h \
and review /dev/shm "used" column

## **Drawbacks**
Are there any reasons why we should not do this? Here we aim to evaluate risk and check ourselves.

Please consider:
* is it a breaking change?
* Impact on UX
* implementation cost, both in terms of code size and complexity
* integration of this feature with other existing and planned features


## **Alternatives**
What other designs have been considered? What is the impact of not doing this?


## **Prior Art**
Discuss prior art (both good and bad) in relation to this proposal:
* Does this feature exist in other libraries? What experience has their community had?
* What lessons can be learned from other implementations of this feature?
* Published papers or great posts that discuss this


## **How we teach this**
* What names and terminology work best for these concepts and why? How is this idea best presented?
* Would the acceptance of this proposal mean the PyTorch documentation must be re-organized or altered?
* How should this feature be taught to existing PyTorch users?


## **Unresolved questions**
* What parts of the design do you expect to resolve through the RFC process before this gets merged?
* What parts of the design do you expect to resolve through the implementation of this feature before stabilization?
* What related issues do you consider out of scope for this RFC that could be addressed in the future independently of the solution that comes out of this RFC?


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
