import torch


class MyCos(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sparse, fill_value):
        ctx.save_for_backward(sparse, fill_value)
        out_sparse = torch.sparse_coo_tensor(
            sparse.indices(),
            torch.cos(sparse.values()),
            sparse.shape).coalesce()
        out_fill_value = torch.cos(fill_value)
        return out_sparse, out_fill_value

    @staticmethod
    def backward(ctx, grad_out, grad_out_fill_value):
        sparse, fill_value = ctx.saved_tensors
        assert (sparse.indices() == grad_out.indices()).all()
        grad_sparse = torch.sparse_coo_tensor(
            sparse.indices(),
            -torch.sin(sparse.values()) * grad_out.values(),
            sparse.shape)
        grad_fill_value = -torch.sin(fill_value) * grad_out_fill_value
        return grad_sparse, grad_fill_value


def test_MyCos():
    # dense tensor example
    x_dense = torch.tensor([0., 2., 1., 2., 2.], requires_grad=True)
    torch.cos(x_dense).sum().backward()

    # sparse tensor and non-zero fill value:
    x = torch.sparse_coo_tensor([[0, 2]], [0., 1.], (5,)).coalesce()
    x.requires_grad = True
    f = torch.tensor(2., requires_grad=True)

    # evaluate MyCos
    cosx, cosf = MyCos.apply(x, f)

    # apply backward to sum
    (cosx.values().sum() + cosf * 3).backward()

    # relate the gradients from dense and sparse examples:
    assert x.grad[0] == x_dense.grad[0]
    assert x.grad[2] == x_dense.grad[2]
    assert f.grad == x_dense.grad[1] + x_dense.grad[3] + x_dense.grad[4]

    print('ok')


if __name__ == '__main__':
    test_MyCos()
