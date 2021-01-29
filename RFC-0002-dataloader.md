# DataLoader architecture updates and TarDataset implementation
# Problem statement
This proposal aims to construct a modular, user-friendly, and performant toolset to address the ambiguous activity referred to as “dataloading” within PyTorch, a simplification attributable to the indivisibility of the DataLoader abstraction prescribed today. In reality, “dataloading” is a diverse set of operations that should be supported by extensible building blocks, out of which the present abstractions, and far more could be easily built. Some typical needs which are scarcely supported in the present implementation include:

- Lazy loading - Users want to point PyTorch to a remote data source (e.g. HTTP, S3, GCP, Azure, Manifold, Hive) and iterate over the contents without downloading the entire dataset, ideally only downloading samples as-soon-needed. Further, If a user writes such code for a remote storage platform, there is no natural place to contribute it for public or private reuse.

- Structured data, heterogeneous storage

  - There are hundreds of ways to store a single structured dataset, each requiring a custom or highly configured DataLoader. Users want to take advantage of modularity and not reimplement complete DataSets over and over again. Suppose we have a simple dataset of sample:label pairs of images and ints. There are a number of dimensions whose outer product enumerates the possible storage formats for this data, each requiring a distinct (or specifically configured) DataLoader:

    - Primitive formats - Are images stored in an image storage format (e.g. one of these), as a tensor (.pt), as a serialized (e.g. pickle, json) Python data structure, etc.?
    - Grouping - Are pairs grouped together by directory, by an archive format (tar, HDF5), by a serialization format (e.g. json, protobuf), by common file string (e.g. image00012321.jpg, label00012321.txt), by meaningful filenames (e.g. image_00023423_dog.jpg), by contents of file headers, by pickled Python data structures, etc? Are filenames otherwise meaningful? Are file headers otherwise meaningful?
    - Sharding - Is the dataset partitioned for performance reasons into arbitrary groups, each containing grouped pairs? In which grouping format (e.g. directories, tar, arrow, parquet, HDF5)?
    - Compression - Are files or groups compressed or binarized, e.g. gz, zip, protobuf?
    - Locale - Are the files local, remote via custom request format (e.g. proprietary data, public REST API, kaggle dataset), on an http server, in cloud object storage?
  - The above example is only for an extremely simple data structure case. The reality of data is often dramatically more heterogeneous and complex (e.g. variable-length lists of bounding box points and strings in object detection, highly nested structures of user or product features in ranking).
  - Further, users want to find or contribute ops to decode specific file types (e.g. HDF5) and accelerated kernels (e.g. GPU mp3 decoding).
  - Given PyTorch maintains decoders for many storage formats, users want more powerful top-level abstractions, such as simply pointing PyTorch to a local or remote directory and receiving an iterator over best-efforts deserializations of the files within.
- Shuffling - Users want control over when shuffling occurs within “Dataloading.” It often makes a big performance and accuracy difference whether samples within a shard are shuffled, samples are globally shuffled, shards are shuffled, etc.
- Pipelining and parallelism - Users want to be able to pipeline their loading and preprocessing (rather than make multiple CPU passes, for example), specify a number of workers to read and preprocess data, and not worry about whether reading, preprocessing, or model execution are starved. This can include asynchronous processes which prefetch data to feed to others.

Tensorflow addresses many of the above needs with their TFRecord, dramatically simplifying the problem by taking a strong opinion of the data format with which Tensorflow works best. This has been extremely successful from a performance perspective. However, by prescribing a single storage format, all others are demoted, and the diversity of data needs and entrenched formats made ubiquitous adoption of TFRecord for storage practically impossible. We’ve heard directly from users that they do not want to be forced into a single first-class format, and the public datasets (which Google rehosts in TFRecord), tend to agree (by completely disagreeing on format). For this reason, we prefer extensibility over prescription, wherein we provide performant support for a basic set of formats in-tree (e.g. Hive and Manifold internally, tar shard and Arrow externally) but users can plug in modular extensions for new formats easily.

## Underlying DataLoader Issues
Beyond the needs described above, the existing DataLoader is also a frequent source of user requests and github issues. Such feedback includes, but is not limited to:
- Fork and general multi-processing memory usage patterns - There are multiple reports in GitHub that users are confused about how Fork’s copy-on-write and Python’s object counting work together, and that leading to OOMs. And Pytorch users shop for custom solutions as separate list management processes, or sharing binary segments etc.
- Threading vs Multiprocessing - Different use cases require one or the other. For example, threading generally performs better while multiprocessing works better with third-party libraries with non-threadlocal state.
- Overcomplication of solutions - TarDataset requires custom shuffling and sampling implemented as Datasets, while our built-in solution requires altering the DataLoader. It would be best to separate data processing (reordering included) from process management.
- Multiprocessing support - Today, proper pre-fetching is not possible due to the synchronous nature of Datasets. In order to bypass this, users must implement custom multiprocessing-enabled Datasets and DataLoaders themselves.
- Manual sharding - Sharding is increasingly becoming a mandatory feature, allowing better multiprocessing and distributed execution. Currently users must implement it manually.

Finally, the ubiquity of the DataLoader necessitates strong backward compatibility. For this reason we do not plan to deprecate any existing functionality, but in some cases may offer a more modern way of doing things.

# Solution

Break down Dataset into smaller components reducing logic to a queue of data-in and a queue of data-out.  

DataLoader observes the acyclic graph of DataSets and provides the necessary level of parallelism using multiprocessing and multithreading.

Bear in mind that even if we use IteratableDataset in examples below, all this also applicable to MapDataset

### Separating by smaller datasets and connecting them together

```python
class ListFilesIteratableDataset(IteratableDataset):
   #...
   def __iter__(self):
       # yield file_names
 
class LoadFilesIteratableDataset(IteratableDataset):
   def __init__(self, listfiles_ds):
       self._listfiles_ds = listfiles_ds
       # ...
  
   def __iter__(self):
       for file_name in listfiles_ds:
           yield (file_name, load_file(file_name))
```

Will allow us to simplify datasets code and make them reusable across various implementations (for example ImageFolder and TarDataset). Also necessary in case of moving memory consuming datasets into separate processes.

### Turning IteratableDataset and MapDataset into AsyncIteratableDataset and AsyncMapDataset

Multiprocessing/threading support makes us prefer async_next over __next__ function. Key difference is that async_next might throw NotAvailable exception, meaning that data is not yet available and should be requested again with async_next.

Datasets which implements only async_next can be easily used as standard datasets because parent class provides necessary API:

```python
class AsyncIteratableDataset(IteratableDataset):
   def __iter__(self):
       return self
 
   def __next__(self):
       while True:
           try:
               return self.async_next()
           except StopIteration:
               raise StopIteration
           except NotAvailable:
               time.sleep(DELAY)
               EventLoop.iteration()
              
   def async_next(self):
       raise NotImplemented
```

Existing synchronous datasets can be turned into async datasets using helper function: 

```python
def EnsureAsyncNextDataset(validated_dataset):
   if not isinstance(validated_dataset, IteratableDataset):
       raise Exception('Not IteratableDataset')
   if isinstance(validated_dataset, AsyncIteratableDataset):
       return validated_dataset
   if not hasattr(validated_dataset, '_as_iterator'):
       setattr(validated_dataset, '_as_iterator', None)
   if not hasattr(validated_dataset, 'async_next'):
       def async_next(self):
           if self._as_iterator is None:
               self._as_iterator = iter(self)
           return next(self._as_iterator)
       setattr(validated_dataset, 'async_next', async_next)
       validated_dataset.async_next = types.MethodType(async_next, validated_dataset)
   return validated_dataset
```

Combination of two approaches will allow a mix of old-style datasets and new async datasets.

As next_async does not guarantee results to be returned, it can be used to schedule requests ahead:

```python
class PrefetcherAsyncIteratableDataset(AsyncIteratableDataset):
   def __init__(self, source_ds, buffer_size = 10):
       self._souce_ds = source_ds
       self._buffer_size = buffer_size
       self._buffer = []
       self._source_depleted = False
 
   def async_next(self):
       if not self._source_depleted:
           while len(self._buffer) < self._buffer_size:
               try:
                   data = self._souce_ds.async_next()
               except NotAvailable:
                   # break or put more requests, depends from implementation
                   break
               except StopIteration:
                   self._source_depleted = True
                   break
               self._buffer.append(data)
       if len(self._buffer):
           data = self._buffer.pop(0)
           return data
       else:
           if self._source_depleted:
               raise StopIteration
           else:
               raise NotAvailable
```

Similar approach will be applied to MapDataset with async_get(id).

### Connecting blocks with queues

Having all datasets as asynchronous, allows to connect them with a couple of queues.

For example in multiprocessing version, sub process main loop can look like this:

```python
def IteratableDatasetToQueuesLoop(source_dataset, req_queue, res_queue):
   steps = 0
   EventLoop.enabled = False
   for _ in IteratableDatasetBehindQueues(source_dataset, req_queue, res_queue, raise_stop=True):
       steps += 1
       time.sleep(DELAY)
       pass
 
def IteratableDatasetBehindQueues(source_dataset, req_queue, res_queue, raise_stop = False):
   source_dataset = EnsureAsyncNextDataset(source_dataset)
   while True:
       try:
           req_queue.get(block = False)
       except:
           yield True
           continue
       while True:
           try:
               value = source_dataset.async_next()
           except NotAvailable:
               yield True
               continue
           except StopIteration:
               res_queue.put(StopIteration())
               if raise_stop:
                   raise StopIteration
               else:
                   yield True
               continue
           res_queue.put(value)
           yield True # Returns control
           Break
```

When main process can transparently access this dataset with simple wrapper:

```python
class QIteratableDataset(AsyncIteratableDataset):
   def __init__(self, request_queue, response_queue, response_wait_time = 0.00001):
       self._req_q = request_queue
       self._res_q = response_queue
       self._req_sent = False
       self.counter = 0
       self._stop_iteration = False
       self._response_wait_time = response_wait_time
       
   def async_next(self):
       if self._stop_iteration:
           raise Exception('next called after receiving StopIteration')
       if not self._req_sent:
           self._req_q.put(self.counter)
           self.counter += 1
           self._req_sent = True
       try:
           value = self._res_q.get(block = True, timeout = self._response_wait_time)
       except:
           raise NotAvailable
       self._req_sent = False
       if isinstance(value, StopIteration):
           self._stop_iteration = True
           raise StopIteration
       return value
```

Allow to send Dataset into separate process by few lines of code:
 
```python
req_queue = multiprocessing.Queue()
res_queue = multiprocessing.Queue()
p2 = multiprocessing.Process(target=IteratableDatasetToQueuesLoop, args=(source_dataset, req_queue, res_queue))
p2.start()
separated_source_dataset = QIteratableDataset(req_queue, res_queue)
```

Please note, that only one request in the queue, is an implementation restriction and not enforced by design.

### DataLoaderQueue

The above examples using standard multiprocessing Queue, but it is not the best choice (performance-wise) in some cases and not working in others. Instead we suggest to replace it with higher abstraction DataLoaderQueue.

DataLoaderQueue  - used to pass data between elements of a pipeline inside a single thread, between threads, between processes, in distributed env. DataLoader will replace queue with best for the moment implementation, but they all should follow next requirements:

- Non-blocking
- Guaranteed delivery
- Guaranteed no duplicates
- Guaranteed order
- Customizable length
- Queue is always between TWO processes/threads

API: 
- `def get(blocking=True)` - returns any python structure, or raises NotAvailableException, or raises QueueError
- `def put(data, blocking=True)` - data is any Python structure, may raise QueueError

DataLoaderQueue implementation also defines ‘serialization’ technique, from simple pass object reference inside the same thread to IPC calls and full object serialization to be passed via network. 

### Users API

Datasets should work as standard iterators (or implement __get__item__) outside of DataLoader.

```python
numbers_ds = ds.NumbersIteratableDataset() # Returns range of integers
ds1, ds2, ds3 = MultiplyIteratableDatasetList(numbers_ds, 3) # Creates 3 copies of input data
def mult100(x):
   return x * 100
ds2_modified = CallableIteratableDataset(ds2, mult100)
def mult111(x):
   return x * 111
ds3_modified = CallableIteratableDataset(ds2, mult111)
joined_ds = GreedyJoinIteratableDataset(ds1, ds2_modified, ds3_modified)
 
for i in iter(joined_ds):
   print(i) # 0 0 0 1 100 111 222 200 2 ......
```

DataLoader output should be exactly the same, but different pieces of graph might be executed as separate threads/processes.

```python
for i in DataLoader(joined_ds):
   print(i) # 0 0 0 1 100 111 222 200 2 ......
```

### Sharding

Sharding should be implemented on the framework level and hidden from DataSet users. DataSet developers will get control over sharding settings and running configurations. DataLoader will decide how to split DataSet into shards and run configuration.

DataSet blocks will provide information to the DataLoader if they support sharding via dataset.is_shardable(). If a function is not defined Dataset will be considered as non-shardable. 
DataLoader will callback DataSet objects with sharding settings using dataset.sharding_settings(total_num_of_shards, id_of_shard).

Example:

```python
list_files_ds = ListFilesMapDataset(root = '.') * marked as non shardable
load_bins_ds = LoadFilesMapDataset(list_files_ds) * marked as shardable
decode_images_ds = DecodeImagesMapDataset(load_bins_ds) * marked as shardable
transform_ds = TransformImagesDataset(decode_images_ds) * marked as shardable shuffle_ds = ShuffleMapDataset(transform_ds) * marked as non shardable
sampler_ds = SamplerIteratableDataset(shuffle_ds) * marked as non shardable
```

### Individual Process (Thread)

Situations like prefetching and large non-forkable arrays require to spawn separate processes for a Dataset. DataSet blocks will provide information to the DataLoader if they are recommended to be executed as separate processing via dataset.is_separate_process().

### Lazy Initialization

In some cases it is inefficient to initialize DataSet data before usage. For example, we need to postpone loading a full list of files before forking out a file scanner. For this purpose lazy_init function will be called prior to any __len__, __get_item__, __iter__ operators. 

### Functional Dataset
DataLoader should not care about any data logic (including sampling, shuffle, and collate). 

### Moving Sampler from DataLoader into separate Datasets 
We are planning to create Samplers datasets for each existing logic as well as a wrapper around existing Sampler classes.
PR: https://github.com/pytorch/pytorch/pull/49363
```python
sequential_sampled_ds = SampleIterableDataset(iter_ds) # use default sequential sampler (basically do nothing)
random_sampled_ds = SampleIterableDataset(iter_ds, sampler=RandomSampler, replacement=True) # use random sampler with replacement to generate random item from input dataset
```
#### Note:
All of SamplerDatasets can be replaced by another IterableDataset, and Sampler is not required in the Data pipeline.
- RandomSampler without replacement or SubsetSampler -> ShuffleIterableDataset with different buffer size
- WeightedSampler -> WeightedShuffleDataset (If needed)
- BatchSamper -> BatchDataset
- Other customized samplers -> CallableDataset to run customized sample function
In general, sampler dataset is not suggested to be used in the new pipeline, and we keep it in favor of non BC-breaking.
Example for the replacement of SubsetSampler:
```python
def subset_sampler(ds):
    buffer = []
    for x in ds:
        if len(buffer) == buffer_size:
            idx = random.randint(0, buffer_size - 1)
            yield buffer[idx]
            buffer[idx] = x
        else:
            buffer.append(x)
    random.shuffle(buffer)
    while buffer:
        yield buffer.pop()
out = CallableIterableDataset(ds, fn=subset_sampler)
```

### Moving Collate functions from DataLoader into separate Datasets

We are going to move collate logic out of DataLoader and implement it as IterableDataset, it will accept old collate functions as argument or can be rewritten entirely.
PR: https://github.com/pytorch/pytorch/pull/48933
```python
batch_ds = ds.BatchNumbersIteratableDataset() # Returns batch of integers [1,2,3],[4,5,6],..
default_collated_ds = CollateDataset(batch_ds) # use original default collate function
for i in DataLoader(default_collated_ds):
   print(i) # tensor([1, 2, 3]), tensor([4, 5, 6]), ...
 
def collate_fn(batch):
    sum = batch[0] + batch[1] + batch[2]
    return torch.tensor(sum, dtype=torch.float)
collated_ds = CollateDataset(batch_ds, collate_fn=collate_fn)
for i in DataLoader(default_collated_ds):
   print(i) # tensor([6.]), tensor([15.]), ...
```

### Moving Shuffle from DataLoader into separate Datasets 
https://github.com/pytorch/pytorch/blob/7729581414962ac0a23ebd269f165f6a877490ae/torch/utils/data/dataset.py#L257-L312
```python
Iter_ds = ds # Returns 0, 1, 2, 3, 4, 5, 6, 7, 8,...
shuffled_ds = ShuffleIterableDataset(iter_ds) # Returns 5, 2, 9, 0,...
```

### Moving Collate functions from DataLoader into separate Datasets
We are going to move collate logic out of DataLoader and implement it as IterableDataset, it will accept old collate functions as argument or can be rewritten entirely.
PR: https://github.com/pytorch/pytorch/pull/48933
```python
batch_ds = ds.BatchNumbersIteratableDataset() # Returns batch of integers [1,2,3],[4,5,6],..
default_collated_ds = CollateDataset(batch_ds) # use original default collate function
for i in DataLoader(default_collated_ds):
   print(i) # tensor([1, 2, 3]), tensor([4, 5, 6]), ...
 
def collate_fn(batch):
    sum = batch[0] + batch[1] + batch[2]
    return torch.tensor(sum, dtype=torch.float)
collated_ds = CollateDataset(batch_ds, collate_fn=collate_fn)
for i in DataLoader(default_collated_ds):
   print(i) # tensor([6.]), tensor([15.]), ...
```

### Other functional Dataset
In order to provide more versatile API, we plan to add more functional Dataset for users.
- Batch
- PaddedBatch
- unbatch ...
- Repeat
- Cache
- Filter
- zip
- ...

### Reproducibility and randomness

Should be part of DataLoader implementation, to be able to define random seed in case of various parallelization techniques.

Async operations also introduce non-determinism of order, so we would need to implement a DataLoader attribute to order of async calls fulfillments and to guarantee order determinism.

### To Do

This document doesn’t touch the problem of varying batch size for different phases of processing. It is archivable by passing a list of objects into the queue and will be considered at the phase of queue implementation. However it is better to put code example here.

This document doesn't cover distributed training in detail. We are going to extend on this topic using additional sharding parameters and queue implementations.

### Considerations

User defined sharding was considered unnecessary at the early stages, however, nothing in the proposed architecture prevents from implementing it later.

CPP implementation was considered as non-flexible. However, nothing prevents users from creating Datasets with CPP internals.

Torchscript can be used inside of Datasets, but we are not limited to it.

Arrow/Proto/… can be used to pass data between Datasets. 

Error Tracing?

C++


cc @SsnL @VitalyFedyunin @ejguan
