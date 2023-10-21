# Proposal of implementation of Lion Optimizer

**Authors:**
* @RahulVadhyar

## **Summary**
Lion Optimizer is quickly becoming a good alternative to the popular AdamW Optimizer. It is an optimizer that uses the sign operator to control the magnitude of the update rather than using
second order moments. By not calculating second order moments, we save on memory and increase training speed. It generally performs similarly to Adam optimizer. Only change is that the learning rate has to be 8-10x smaller than what is typically used for Adam.
The Paper from Google Brain - https://arxiv.org/pdf/2302.06675.pdf


## **Motivation**
There are many third party libraries and implementations of the Lion Optimizer. It would be beneficial to the community if there was a default native implementation of the Lion optimzer. Not only would there be less friction in using this optimizer, but also more people would get exposed to this optimizer in deep learning and improve their models. In addition Tensorflow has implemented this optimizer. 
Links to Tensorflow implementation:
Docs - https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Lion
GitHub - https://github.com/keras-team/keras/blob/v2.14.0/keras/optimizers/lion.py

Some implementations of this for pytorch are:

- https://github.com/DHDev0/Lion-optimizer-pytorch
- https://github.com/lucidrains/lion-pytorch
- https://fastxtend.benjaminwarner.dev/optimizer.lion.html

## **Proposed Implementation**
The implementation would be inspired by the many third party libaries for PyTorch. It would be adapted and modified heavily to abide by the PyTorch guidelines. The optimizer would have a similar implementation to what other optimizers look like in PyTorch with the function header looking like this:

```python
torch.optim.Lion(params, lr=<required parameter>, momentum = 0, beta_1 = 0.9, beta_2 = 0.99, weight_decay = 0, nesterov=False, *, maximize = False, foreach = None, differentiable: bool = False)
```

The algorithm would be implemented as per the implmentation in the above paper.

## **Drawbacks**
One possible drawback is that it could bloat the api of PyTorch and may add unneccessary features that are either niche or not used. However since this is a feature which is used in other libaries in tensorflow, and considering that it is better than SGD and in some cases AdamW, it should be a well used optimizer once implemented.


