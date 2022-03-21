<details>
<summary>Instructions - click to expand</summary>

- Fork the rfcs repo: https://github.com/pytorch/rfcs
- Copy `RFC-0000-template.md` to `RFC-00xx-my-feature.md`, or write your own open-ended proposal. Put care into the details.
- Submit a pull request titled `RFC-00xx-my-feature`.
    - Assign the `draft` label while composing the RFC. You may find it easier to use a WYSIWYG editor (like Google Docs) when working with a few close collaborators; feel free to use whatever platform you like. Ideally this document is publicly visible and is linked to from the PR.
    - When opening the RFC for general discussion, copy your document into the `RFC-00xx-my-feature.md` file on the PR and assign the `commenting` label.
- Build consensus for your proposal, integrate feedback and revise it as needed, and summarize the outcome of the discussion via a [resolution template](https://github.com/pytorch/rfcs/blob/rfc-process/RFC-0000-template.md#resolution).
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





# A suggestion of channels last memory format implementation for 3D tensor

**Authors:**
* @Kevin Yu


## **Summary**
ChannelsLast1d implementation suggestion which is aligned to ChannelsLast(2d) and ChannelsLast3d usage habits.

## **Motivation**
Pytorch has already supported ChannelsLast(2d) for 4D tensor(N, C, H, W) and ChannelsLast3d for 5D tensor(N, C, H, W, D), but doesn't support ChannelsLast1d for 3D tensor(N, C, L). See below:

**ChannelsLast for 4D tensor works fine:**
```
>>> import torch
>>> N, C, H, W = 8, 3, 32, 32
>>> _4d_tensor = torch.empty(N, C, H, W)
>>> _4d_tensor_cl = _4d_tensor.to(memory_format=torch.channels_last)
>>> _4d_tensor_cl.is_contiguous(memory_format=torch.channels_last)
True
>>> tensor_4d.stride()
(3072, 1024, 32, 1)
```
**ChannelsLast for 5D tensor works fine:**
```
>>> import torch
>>> N, C, H, W, D = 8, 3, 32, 32, 32
>>> _5d_tensor = torch.empty(N, C, H, W, D)
>>> _5d_tensor_cl = _5d_tensor.to(memory_format=torch.channels_last_3d)
>>> _5d_tensor_cl.is_contiguous(memory_format=torch.channels_last_3d)
True
>>> _5d_tensor_cl.stride()
(98304, 1, 3072, 96, 3)
```
**ChannelsLast for 3D tensor doens't work:**
```
>>> import torch
>>> N, C, L = 8, 3, 32
>>> _3d_tensor = torch.empty(N, C, L)
>>> _3d_tensor_cl = _3d_tensor.to(memory_format=torch.channels_last_1d)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: module 'torch' has no attribute 'channels_last_1d'
>>> _3d_tensor_cl = _3d_tensor.to(memory_format=torch.channels_last)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: required rank 4 tensor to use channels_last format
>>> _3d_tensor_cl = _3d_tensor.to(memory_format=torch.channels_last_3d)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
RuntimeError: required rank 5 tensor to use channels_last_3d format
```

However, operators such as conv1d, pool1d etc. demand ChannelsLast1d to get better performance boost due to the natural advantages of Channels Last memory format.

## **Usage**
After the feature is supported, it works as below:
```
>>> import torch
>>> N, C, L = 8, 3, 32
>>> _3d_tensor = torch.empty(N, C, L)
>>> _3d_tensor_cl = _3d_tensor.to(memory_format=torch.channels_last_1d)
>>> _3d_tensor_cl.is_contiguous(memory_format=torch.channels_last_1d)
True
>>> _3d_tensor_cl.stride()
(96, 1, 3)
```

## **Value**
ChannelsLast1d feature would benifit such as time series analysis models, deep learning model based on Lidar data, voice model wav2vec, etc.

## **Proposed Implementation**
The general implementation principle is as below:
1. ChannelsLast1d will align to the usage habits of ChannelsLast(2d) and ChannelsLast3d to provide consistent use experience;
2. No extra bits are added in TensorImpl structure;
3. Does not introduce any overhead for important function refresh_continguous(). It does not affect the computation of any original ChannelsLast(2d) and ChannelsLast3d associated flags;
4. The feature is transparent to the end users if they don't use it, both in functionality and performance.


The details are as follows:

Regarding to 1: Users can use it as below:
```
_3d_tensor_cl = _3d_tensor.to(memory_format=torch.channels_last_1d)
_4d_tensor_cl = _4d_tensor.to(memory_format=torch.channels_last)
_5d_tensor_cl = _5d_tensor.to(memory_format=torch.channels_last_3d)
```
Regarding to 2 and 3: As is known, for ChannelsLast(2d) and ChannelsLast3d, there are associated flags in TensorImpl structure as below:
```
  bool is_channels_last_ : 1;
  bool is_channels_last_contiguous_ : 1;
  bool is_channels_last_3d_ : 1;
  bool is_channels_last_3d_contiguous_ : 1;
```
Then refresh_contiguous() would update these flags to track the tensor memory format information. APIs such as is_contiguous(), is_strides_like_channels_last(), is_strides_like_channels_last_3d(), etc. could work based on these flags.
To avoide to introudce extra bits into TensorImpl structure, don't define such as ```bool is_channels_last_1d_ : 1; bool is_channels_last_1d_contiguous_ : 1;``` in TensorImpl structure for ChannelsLast1d, which would not introudce any overhead for key function refresh_contiguous(). If the associated APIs(e.g.: is_contiguous()) demand the memory format information for ChannelsLast1d, we do it as below:
```
  TENSORIMPL_MAYBE_VIRTUAL bool is_contiguous(
      at::MemoryFormat memory_format = at::MemoryFormat::Contiguous) const {
    ......
    if (memory_format == at::MemoryFormat::ChannelsLast1d) {
      return compute_channels_last_contiguous_1d();  //<----------------- caculate it once we need it for ChannelsLast1d
    } else if (memory_format == at::MemoryFormat::ChannelsLast) {
      return is_channels_last_contiguous_; //<--------------------------- just read it once we need it for ChannelsLast(2d), becasue the flag has been updated by refresh_contiguous()
    } else if (memory_format == at::MemoryFormat::ChannelsLast3d) {
      return is_channels_last_3d_contiguous_; //<------------------------ the same as ChannelsLast(2d) above
    }
    ......
  }
```
Regarding 4: If users don't use ChannelsLast1d, they don't need do anything. If user want to use ChannelsLast1d, they can get the same user experience as ChannelsLast(2d) and ChannelsLast3d.


## **Metrics **
As is known, channels last format has better performance than channels first format for most of operators such as conv.
* Conv1d channels last format on Intel CPU achieves about 1.99x maximum performance boost compared with conv1d channels first format for differenct shapes from wav2vec model.


## **Alternatives**
We have considered to add such as ```bool is_channels_last_1d_ : 1; bool is_channels_last_1d_contiguous_ : 1;``` into TensorImpl, and update their values by refresh_contiguous() function, however, refresh_contiguous() is used to create all new tensors, even when people aren't interested in channels-last 1d support. This alternative implementation may introduce overhead for Pytorch.


## **How we teach this**
ChannelsLast1d will align to the usage habits of ChannelsLast(2d) and ChannelsLast3d to provide consistent use experience as below, so we believe that there is no learning cost for users.
```
_3d_tensor_cl = _3d_tensor.to(memory_format=torch.channels_last_1d)
_4d_tensor_cl = _4d_tensor.to(memory_format=torch.channels_last)
_5d_tensor_cl = _5d_tensor.to(memory_format=torch.channels_last_3d)
```
