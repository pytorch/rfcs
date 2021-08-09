# Should Pytorch Provide Performance Hints?

Pytorch prides itself on being a user centric library where historically most of its users have been researchers. However, a growing number of developers are interested in deploying Pytorch with the best possible out of box throughput while meeting some hard latency constraint in inference.

Unfortunately, getting these speedups while possible requires lots of domain knowledge and without lots of Googling and talking to people it's difficult to figure out how to proceed next. I went through this experience trying to get the best possible performance on a BERT model and summarized some of my observations below.

The goal of this RFC is to write down optimizations people have in their heads in code.

All the [results from my experiments are here](https://drive.google.com/file/d/1m_I-ikRh_8i7s_ClLosyL-3whx6qCF5k/view?usp=sharing)

## Where do we give hints?
In the wider Pytorch ecosystem we already give out performance hints for example in the profiler suggesting larger batch sizes or that a data loader would be beneficial. Other partners like HuggingFace and Pytorch Lightning will give out hints from their trainers. Notably HuggingFace gives out a warning when you're downloading a pretrained model that it needs to be finetuned. Pytorch proper doesn't give out hints as far as I can tell.

## Magic Configs

The best evidence of some of this "you're holding it wrong complexity" is what I call "magic configs". Which is a line that doesn't make too much a priori sense increases performance substantially. No-one is going to google whether model dimensions need to be powers of 8 unless they know the answer.

1. On CPU changing a single line of code dramatically improves throughput `torch.set_num_threads(1)`
2. On newer GPUs, model params need to be multiples of 8 to take advantage of Tensor Cores https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#tensorop - outside of a few people at large internet companies I haven't seen people generally be aware of this optimization.
3. Batch sizes should be models of 2.

But these and others could be automatically added based on some profiling and device identification. They don't have to be magic.


## torch.device()
In some ways `torch.device` feels like it's missing an extra step.

```
torch.cuda.get_device_name(0)
>>> 'GeForce GTX 950M'
```

So you can get the device name but it's not entirely clear which classes of optimizations would work on this hardware? Does model.half() work? Would mixed precision training work? What about Tensor Cores? Int8?

So perhaps one solution is to maintain some dictionary `torch.device.optimizations[torch.cuda.get_device_name(0)]` to help people figure this out more directly.

Another solution is something we've seen Intel do with IPEX which is set all the magic configs in a Docker image https://github.com/intel/intel-extension-for-pytorch/blob/master/docker/Dockerfile. This may be easier for the Pytorch team to manage but means that out of the box performance with Pytorch will rarely be ideal.

## Device specific quantization
A good place to add these kinds of optimizations is gate them behind a CLI config in torchserve and add them to the handler. We talk a lot about quantization in our mobile story but in our server story it's much more implicit. Are there any other places where suggesting quantization would be appropriate?

### CPU 
On CPU with torchsript `model.half()` will throw errors like

```
RuntimeError: "LayerNormKernelImpl" not implemented for 'Half'
```

Whereas dynamic quantization just works and gives a free performance boost with 1 line

```
quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

Unfortunately the Pytorch Lightning Trainer's callback seems to have some usability kinks that aren't fully ironed out https://github.com/PyTorchLightning/pytorch-lightning/issues/8570



### GPU
CPU based dynamic quantization is 1 line almost free perf booster. 

GPU based `model.half()` works by again just changing a single line of code

```
model.to(device).eval().half()
```

Inference mode mixed precision is also a single line code change

```
with torch.cuda.amp.autocast():
  predictions = self.model(input_ids_batch, attention_mask_batch)
```

A glaring gap here is lack of support for TensorRT.

## Data parallelism
A warning saying you're not using available GPUs or your worker to model assignment isn't great on CPU. This could come up by default if you're using Pytorch Distributed but not actually distributing your work. On GPU this is a bit more obvious but on CPU this is a confusing point. At least from experiments I've ran with torchserve, I found setting number of workers to 1 even on the fanciest CPUs to give the best throughput.

## Domain specific optimizations

Some optimizations are instead domain specific so as an example if you're trying to optimize the throughput of a BERT like model. Some things you'd think about are.

1. Making sure you're not over-padding (which you can detect via profiling)
2. Use DistilBERT instead of BERT
3. Increase batch size and see throughput increases until latency budget is exceeded
4. Caching - in a real setting not all requests are necessarily unique so making this easier is of real value

I haven't worked much with CNNs but I would suspect the same heuristics work fine.

The first 2 may be specific to HuggingFace but 3 is generally useful anywhere since you can just check for the device utilization during a run and suggest increasing the batch size if it's small.

## Benchmarks

All the [results from my experiments are here](https://drive.google.com/file/d/1m_I-ikRh_8i7s_ClLosyL-3whx6qCF5k/view?usp=sharing)

Most of these benchmarks were ran using Torchserve's Apache Bench suite https://github.com/pytorch/serve/tree/master/benchmarks#run-benchmark-using-a-config-file where the steps involved are
1. Creating a mar file of a HuggingFace model which archives the necessary model dependencies and handler https://github.com/pytorch/serve/tree/master/examples/Huggingface_Transformers
2. Upload the mar file to S3
3. Run the benchmarks and collect the results