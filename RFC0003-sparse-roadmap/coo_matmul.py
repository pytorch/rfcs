"""
Prototype for matrix multiplication of sparse matrices in COO format.
"""
# Author: Pearu Peterson
# Created: July 2020
# Testing: pytest -sv coo_matmul.py -x

from collections import defaultdict
import numpy as np


class COO:
    """Sparse matrix in COO format.
    """
    def __init__(self, indices, values, shape):
        indices = np.asarray(indices)
        values = np.asarray(values)
        self._indices = indices
        self._values = values
        self.shape = shape

    def coalesce(self):
        d = defaultdict(self._values.dtype.type)
        for _i, _j, _v in sorted(
                zip(self._indices[0], self._indices[1], self._values)):
            d[_i, _j] += _v
        i = []
        j = []
        v = []
        for _i, _j in sorted(d):
            i.append(_i)
            j.append(_j)
            v.append(d[_i, _j])
        return type(self)([i, j], v, self.shape)

    @property
    def is_coalesced(self):
        i = self._indices[0]
        j = self._indices[1]
        c = 0
        d = set()
        for _i, _j in zip(i, j):
            _c = _i * self.shape[1] + _j
            if _c < c:
                return False
            c = _c
            if (_i, _j) in d:
                return False
            d.add((_i, _j))
        return True

    @staticmethod
    def _indices2csr(indices, dim):
        nnz = len(indices)
        r = [0] * (dim + 1)
        last_i = 0
        for i in indices:
            if i == last_i:
                r[last_i + 1] += 1
            else:
                for _i in range(last_i, i + 1):
                    r[_i + 1] = r[last_i + 1]
                last_i = i
                r[last_i + 1] += 1
        for _i in range(last_i, dim):
            r[_i + 1] = r[last_i + 1]
        assert r[-1] == nnz
        return r

    def __matmul__(self, other):
        if not isinstance(other, COO):
            return NotImplemented

        assert self.shape[1] == other.shape[0]

        i1 = self._indices[0]
        j1 = self._indices[1]
        v1 = self._values
        nnz1 = len(i1)

        i2 = other._indices[0]
        j2 = other._indices[1]
        v2 = other._values
        nnz2 = len(i2)

        if self.is_coalesced and other.is_coalesced:
            # i1 and i2 are sorted
            # j1 and j2 are sorted for the same i1 and i2 values

            # find CSR representation of other
            r2 = self._indices2csr(i2, other.shape[0])
            d = defaultdict(v1.dtype.type)
            for n1 in range(nnz1):
                for n2 in range(r2[j1[n1]], r2[j1[n1]+1]):
                    assert i2[n2] == j1[n1]
                    d[i1[n1], j2[n2]] += v1[n1] * v2[n2]
        else:
            d = defaultdict(v1.dtype.type)
            for n1 in range(nnz1):
                for n2 in range(nnz2):
                    if i2[n2] == j1[n1]:
                        d[i1[n1], j2[n2]] += v1[n1] * v2[n2]

        i3 = []
        j3 = []
        v3 = []
        for i, j in sorted(d):
            i3.append(i)
            j3.append(j)
            v3.append(d[i, j])

        return type(self)([i3, j3], v3, (self.shape[0], other.shape[1]))

    def numpy(self):
        i = self._indices[0]
        j = self._indices[1]
        v = self._values
        nnz = len(i)

        a = np.zeros(self.shape, dtype=self._values.dtype)
        for n in range(nnz):
            a[i[n], j[n]] = v[n]
        return a

    @classmethod
    def from_numpy(cls, a):
        assert len(a.shape) == 2
        i, j, v = [], [], []
        for _i in range(a.shape[0]):
            for _j in range(a.shape[1]):
                _v = a[_i, _j]
                if _v != 0:
                    i.append(_i)
                    j.append(_j)
                    v.append(_v)
        return cls([i, j], v, a.shape)

    @classmethod
    def random(cls, shape, dtype=int, sparsity=0.75, coalesce=True):
        assert len(shape) == 2
        assert 0 <= sparsity <= 1
        nnz = int(shape[0] * shape[1] * (1 - sparsity))
        nnz = max(0, min(nnz, shape[0] * shape[1]))

        i, j, d = [], [], set()
        for n in range(nnz):
            while 1:
                _i = np.random.randint(shape[0])
                _j = np.random.randint(shape[1])
                if (_i, _j) in d:
                    continue
                d.add((_i, _j))
                break
        if coalesce:
            for _i, _j in sorted(d):
                i.append(_i)
                j.append(_j)
        else:
            for _i, _j in d:
                i.append(_i)
                j.append(_j)
        v = np.array(range(1, nnz+1), dtype=dtype)

        return cls([i, j], v, shape)

    def __repr__(self):
        return (f'{type(self).__name__}('
                f'{self._indices}, {self._values}, {self.shape})')

    def __str__(self):
        return str(self.numpy())


def test_coalesce():
    while 1:
        a = COO.random((3, 5), sparsity=.5, coalesce=False)
        if not a.is_coalesced:
            break
    assert not a.is_coalesced
    assert a.coalesce().is_coalesced
    assert (a.coalesce().numpy() == a.numpy()).all()
    assert COO.from_numpy(a.numpy()).is_coalesced
    assert (COO.from_numpy(a.numpy()).numpy() == a.numpy()).all()
    a = COO.random((3, 5), sparsity=.5, coalesce=True)
    assert a.is_coalesced


def test_matmul_uncoalesce():
    while 1:
        a = COO.random((3, 5), sparsity=.5, coalesce=False)
        if not a.is_coalesced:
            break
    while 1:
        b = COO.random((5, 4), sparsity=.5, coalesce=False)
        if not b.is_coalesced:
            break

    c = a @ b
    assert c.is_coalesced

    cn = a.numpy() @ b.numpy()
    assert (cn == c.numpy()).all()


def test_matmul_coalesce():
    N, K, M = 3, 5, 4

    a = COO.random((N, K), sparsity=.65, coalesce=True)
    b = COO.random((K, M), sparsity=.65, coalesce=True)
    assert a.is_coalesced
    assert b.is_coalesced

    c = a @ b
    assert c.is_coalesced

    cn = a.numpy() @ b.numpy()
    assert (cn == c.numpy()).all()
