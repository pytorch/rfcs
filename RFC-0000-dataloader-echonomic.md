

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

By current dataloader multiprocessing pipline design, workers simultaneously prepere batches and send them into shared memory, by a queue.
In practice, about [num_workers] batches are simultenously stored in shared memory, nearly after epoch start. 
At most, [num_workers * prefetch_factor] may be simultenously stored in shared memory.
The main process operates in parallel to the workers, to extract one batch after another, from shared memory, and inject it into the model for training/validation/test. 

Simultenously storing about [num_workers] batches in shared memory, imposes a limit over [num_workers]:\
[num_workers < servers_total_available_ram_in_bytes / batch_size_in_bytes]\
This limitation can produce a bottleneck over training TPT, not allowing to increase num_workers, due to server's RAM limitations.
Alternatively, in order to increase num_workers, a severs with more RAM can be used, increaseing sever cost.

A new dataloader multiprocessing pipeline is suggested.
In this pipline, only up to [prefetch_factor] batches are simultenously processed by all the workers together.
This decoupling from [num_workers], allowes to increase [num_workers], without any significant increase in shared memory consumption. 
As in current implemnation, workers are not expected to enter idle state, hence no TPT reduction is expected.

The suggested flow is designated to reduce RAM related bottelnecks and/or requirements, and improve training costeffectiveness.

## **Proposed Implementation**
### **Definitions**

| symbol                | description                                                                                                                     |
|-----------------------|:--------------------------------------------------------------------------------------------------------------------------------|
| index_queue           | A queue used to send item's index and metadata from main process to item_worker. There is a seperate queue to each item_worker. |
| item_queue            | A queue used to send item from item_workers to batch_worker. There is a seperate queue to each batch_worker.                    |
| worker_result_queue   | A queue used to send prepared batches from batch_workers to main process.                                                       |
| item_idx              | Item serial index from epoch start (0 for first item, 1 for next item, etc)                                                     |
| batch_idx             | Batch serial index from epoch start (0 for first batch, 1 for next batch, etc)                                                  |
| item_index            | Item's dataset index, as in dataset.__getitem__(index)                                                                          |
| iw_idx                | Item_worker index                                                                                                               
| bw_idx                | Batch_worker index                                                                                                              


By the current multiprocessing pipeline, a single level of workers is used. 
The main process sends [prefetch_factor] batches to each worker, by the worker's index_queue.
Each worker prepares one batch at a time, and send it back to the main process by worker_result_queue.
After a batch is retrived by the main process, another batch is sent to the appropriate worker.

A new multiprocessing pipline is suggested. In the suggested pipeine, there are 2 levels of workers: 
* item_workers - Designated to generate one item at a time (by running dataset \_\_getitem__ function), and send it to shared memory by item_queue 
  * This worker is similar to the workers of the current design, but it recieves and sends one item at a time (and not one batch at a time) 
* batch_workers - Designated to get items from shared memory, collect [batch_size] items, run collate function, and send the prepared batch back to shared memory, by worker_result_queue

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
The new flow should require significantly less shared memory, while preserving TPT, using similar configurations. \
To monitor shared memory usage, type in linux server terminal: \
$ monitor -n0.1 df -h \
and review /dev/shm "used" column.

## **Drawbacks**
* In the suggested implementation, the prefetch_factor becomes more prominent.
It determines the total number of items sent simultenously to all workers, and (by default) also determines num_workers_batches.
Hence, this parameter should be set with more attention. Additionally, a larger default value may be considered (possibly 3 instead of 2).
* Number of workers required for the same TPT increases by num_batch_workers.

## **How we teach this**
* dataloader documentation should be updated to include:
  * Add a new parameter: num_batch_workers
  * Revise parameter description: prefetch_factor
  
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
