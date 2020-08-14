"""Reference implementations to N-dimensional arrays.

This module provides N-D array implementations for the following formats:

  Strided
  COO
  CRS

To run unit-tests:
  pytest --cov=ndarrays --cov-report term-missing ndarrays.py
"""
# Author: Pearu Peterson
# Created: June 2020

import itertools
from collections import defaultdict

class NDArray(object):
    """Base class for N-D array implementations.

    0-D arrays are interpreted as scalars.
    """

    @classmethod
    def from_data(cls, data):
        """
        Construct array instance from a nested sequence of values.
        """
        if isinstance(data, cls):
            return data
        raise NotImplementedError(f'{cls.__name__}.from_data({type(data)})')  # pragma: no cover

    def to_layout(self, new_cls):
        """Convert array from one storage format to another.

        When possible, avoid coping data.
        """
        if isinstance(self, new_cls):
            return self
        return new_cls.from_data(self)

    @property
    def shape(self) -> tuple:
        """
        Return the shape of N-dimensional array as a N-tuple of dimension lengths.
        """
        return self._shape

    @property
    def ndims(self) -> int:
        """
        Return the number of array dimensions N.
        """
        return len(self.shape)

    @property
    def numel(self) -> int:
        """
        Return the total number of array elements.
        """
        if self.ndims == 0:  # scalar
            return 1
        return product(self.shape)

    def __str__(self):
        if self.ndims == 0:
            return str(self[()])
        lst = []
        for i in range(self.shape[0]):
            row = self[i]
            lst.append(str(row))
        return '[' + ', '.join(lst) + ']'

    def items(self):
        """Iterator of over index and element values.
        """
        raise NotImplementedError(f'{type(self).__name__}.items()')  # pragma: no cover

    def __getitem__(self, index):
        """Return element or sub-array at index.
        """
        raise NotImplementedError(f'{type(self).__name__}.__getitem__()')  # pragma: no cover

    def __eq__(self, other):
        """Return True if two arrays are equal element-wise.
        """
        if not isinstance(other, NDArray):
            other = Strided.from_data(other)
        if self.shape != other.shape:
            return False
        if self.ndims == 0:
            return self[()] == other[()]
        indices = [tuple(range(dims)) for dims in self.shape]
        for index in itertools.product(*indices):
            if self[index] != other[index]:
                return False
        return True

    def __ne__(self, other):
        return not (self == other)


class Strided(NDArray):
    """Dense array with strides.
    """
    
    def __init__(self, strides, values, shape):
        """
        Parameters
        ----------
        strides : N-tuple
        values : 1-D sequence
        shape : N-tuple
        """
        self._strides = strides
        self._values = values
        self._shape = shape

    @classmethod
    def from_data(cls, data):
        if isinstance(data, NDArray):
            if data.ndims == 0:
                return cls((), [data[()]], ())
            if isinstance(data, (COO, GCS)):
                shape = data.shape
                ndims = data.ndims
                strides = make_strides(shape)
                numel = product(data.shape)
                values = [data._fill_value] * numel
                for index, value in data.items():
                    p = sum(strides[i] * index[i] for i in range(ndims))
                    values[p] = value
                return cls(strides, values, shape)
        elif is_sequence(data):
            return cls(*get_strided_data(data))
        if is_scalar(data):
            return cls((), [data], ())
        return super(Strided, cls).from_data(data)

    def __repr__(self):
        return f'{type(self).__name__}({self._strides!r}, {self._values!r}, {self.shape})'

    @property
    def is_contiguous(self):
        """Return True if array values are stored contiguously in row-major
        order.
        """
        return self._strides == make_strides(self.shape)

    def items(self):
        if self.ndims == 0:
            yield (), self[()]
        else:
            indices = [tuple(range(dims)) for dims in self.shape]
            for index in itertools.product(*indices):
                yield index, self[index]

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        n = len(index)
        N = self.ndims
        assert n <= N
        p = sum([self._strides[i] * index[i] for i in range(n)])
        if n == N:
            return self._values[p]
        strides = self._strides[n:]
        shape = self._shape[n:]
        values = self._values[p:p + product(shape)]
        return type(self)(strides, values, shape)

    
class COO(NDArray):
    """Sparse array in COO format.
    """

    def __init__(self, indices, values, shape, fill_value=None):
        """Parameters
        ----------
        indices : (N, NNZ)-sequence
        values : NNZ-sequence
        shape : N-tuple
        fill_value : {None, scalar, type(values[0])}

        Notes
        -----

          If fill_value is None then it means that unspecified
          elements are undefined and do not participate in array
          operations.

        """
        nnz = len(values)
        N = len(shape)
        assert len(indices) == N, (indices, values, shape)
        for i in range(N):
            assert len(indices[i]) == nnz
        self._indices = indices
        self._values = values
        self._shape = shape
        self._fill_value = fill_value

    @classmethod
    def from_data(cls, data, fill_value=None):
        if isinstance(data, NDArray):
            if data.ndims == 0:
                return cls([], [data[()]], ())
            if isinstance(data, Strided):
                shape = data.shape
                indices = [[] for j in range(len(shape))]
                values = []
                for index, value in data.items():
                    if value == fill_value:
                        continue
                    for j in range(len(shape)):
                        indices[j].append(index[j])
                    values.append(value)
                return cls(indices, values, shape, fill_value=fill_value)
            if isinstance(data, COO):
                assert data._fill_value == fill_value
                return cls(data._indices, data._values, data.shape, fill_value=fill_value)
            if isinstance(data, GCS):
                assert data._fill_value == fill_value
        elif is_sequence(data):
            return cls(*get_coo_data(data), **dict(fill_value=fill_value))
        elif is_scalar(data):
            return cls([], [data], (), fill_value=fill_value)
        return super(COO, cls).from_data(data)

    def __repr__(self):
        return f'{type(self).__name__}({self._indices!r}, {self._values!r}, {self.shape}, fill_value={self._fill_value})'

    def items(self):
        if self.ndims == 0:
            yield (), self._values[0]
        else:
            N = self.ndims
            nnz = len(self._values)
            for i in range(nnz):
                index = tuple(self._indices[j][i] for j in range(N))
                yield index, self._values[i]

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)

        n = len(index)
        N = self.ndims
        assert n <= N

        if n == N == 0:
            return self._values[0]

        nnz = len(self._values)
        
        indices = [[] for j in range(n, N)]
        values = []
        for i in range(nnz):
            _index = tuple(self._indices[j][i] for j in range(n))
            if index == _index:
                for j in range(n, N):
                    indices[j - n].append(self._indices[j][i])
                values.append(self._values[i])
        if n == N:
            if values:
                return sum(values)
            else:
                return self._fill_value
        return type(self)(indices, values, self._shape[n:], fill_value = self._fill_value)


class GCS(NDArray):
    """
    Generalized compressed storage.
    """
    def __init__(self, pointers, indices, values, reduction, shape, fill_value=None):
        """
        Parameters
        ----------
        pointers : M-sequence
          M = shape[0] + 1
          `pointers[i + 1] - pointers[i]` is the number of nz elements in i-th row
          assert pointers[0] == 0
          assert pointers[-1] == NNZ
        indices : NNZ-sequence
        values : NNZ-sequence
        reduction : (N+1)-sequence
          l = reduction[-1]
          pi = reduction[:l]
          rho = reduction[l:]
          assert set(pi + rho) == set(range(N))
        shape : N-tuple
        fill_value : {None, scalar, type(values[0])}

        Notes
        -----
        1. For N==2 and `A` in CRS format, we have
          n = 0
          for i in range(shape[0]):
              for j in indices[pointers[i]:pointers[i+1]]:
                  assert A[i, j] == values[n]
                  n += 1

        2. For general case, we have

          N = len(shape)
          l = reduction[-1]
          pi = reduction[:l]
          rho = reduction[l:N]
          n = 0
          for p in range(product(pi)):
              for q in indices[pointers[p]:pointers[p+1]]
                  index = unpack(p, q, reduction)
                  assert A[*index] == values[n]
                  assert pack(index, reduction) == (p, q)
                  n += 1
        """
        N = len(shape)
        if reduction is None:
            # For N == 2 this choice corresponds to CRS format.
            l = N//2
            reduction = shape + (l,)
        assert 0 <= reduction[-1] < N, repr((reduction, shape))
        assert set(reduction[:-1]) == set(range(N)), repr((reduction, shape))

        self._pointers = pointers
        self._indices = indices
        self._values = values
        self._reduction = reduction
        self._shape = shape
        self._fill_value = fill_value

    def __repr__(self):
        return f'{type(self).__name__}({self._pointers!r}, {self._indices!r}, {self._values!r}, {self._reduction}, {self.shape}, fill_value={self._fill_value})'

    @classmethod
    def from_data(cls, data, reduction=None, fill_value=None):
        if isinstance(data, NDArray):
            if data.ndims == 0:
                return cls([], [], [data[()]], (), ())
            if isinstance(data, Strided):
                raise NotImplementedError('gcs from Strided')
                shape = data.shape
                indices = [[] for j in range(len(shape))]
                values = []
                for index, value in data.items():
                    if value == fill_value:
                        continue
                    for j in range(len(shape)):
                        indices[j].append(index[j])
                    values.append(value)
                return cls(indices, values, shape, fill_value=fill_value)
            if isinstance(data, COO):
                raise NotImplementedError('gcs from COO')
                assert data._fill_value == fill_value
                return cls(data._indices, data._values, data.shape, fill_value=fill_value)
            if isinstance(data, GCS):
                assert data._fill_value == fill_value
        elif is_sequence(data):
            return cls(*get_gcs_data(data, reduction=reduction, fill_value=fill_value), **dict(fill_value=fill_value))
        elif is_scalar(data):
            return cls([], [], [data], (), (), fill_value=fill_value)
        return super(GCS, cls).from_data(data)
    
    @classmethod
    def apply_reduction(cls, index, strides, dims):
        print(index, strides, dims)
        return sum(strides[k] * index[dims[k]] for k in range(len(dims)))
        
    def _reduce(self, index):
        l = self._reduction[-1]
        dims1 = self._reduction[:l]
        dims2 = self._reduction[l:-1]
        s1 = make_strides(self.shape, dims=dims1)
        s2 = make_strides(self.shape, dims=dims2)
        p1 = self.apply_reduction(index, s1, dims1)
        p2 = self.apply_reduction(index, s2, dims2)
        return p1, p2

    def _invreduce(self, p1, p2):
        N = self.ndims
        l = self._reduction[-1]
        pi1 = self._reduction[:l]
        pi2 = self._reduction[l:-1]
        s1 = make_strides(pi1)
        s2 = make_strides(pi2)
        invpi = [None] * N
        for i, k in enumerate(self._reduction[:-1]):
            invpi[k] = i
        assert None not in invpi
        index = [None] * len(self._shape)
        for r, s, pi in [
                (p1, s1, pi1),
                (p2, s2, pi2)]:
            d = 1
            for i, next_d in reversed(enumerate(pi)):
                ii = (r // d) % next_d
                r -= s[i] * ii
                index[invpi[i]] = ii
                d = next_d
        assert None not in index
        return tuple(index)

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        p1, p2 = self._reduce(index)
        nc1 = self._pointers[p1]
        nc2 = self._pointers[p1 + 1]
        for nc in range(nc1, nc2):
            if self._indices[nc] == p2:
                return self._values[nc]
        return self._fill_value
            
############################################
##### Array format conversion funtions #####
############################################

def make_strides(shape, dims=None):
    """Return strides of a contiguous array in row-major storage order.
    
    Parameters
    ----------
    shape : tuple

    Returns
    -------
    strides : tuple
    """
    if dims is None:
        dims = tuple(range(len(shape)))
    ndims = len(dims)
    if ndims == 0:
        return ()
    strides = [1]
    for i in range(ndims - 1):
        strides.insert(0, strides[0] * shape[dims[ndims - i - 1]])
    return tuple(strides)    


def get_shape(data):
    """Return the shape of a N-D sequence structure representing a N-D array.
    """
    if is_sequence(data):
        dims = len(data)
        if dims == 0:
            return (0,)
        return (dims, ) + get_shape(data[0])
    elif isinstance(data, np.ndarray):
        return data.shape
    return ()


def reduction_forward(index, strides_lst):
    k = 0
    rindex = ()
    for strides in strides_lst:
        n = len(strides)
        p = sum(strides[i] * index[k + i] for i in range(n))
        rindex += (p, )
        k += n
    return rindex


def reduction_backward(rindex, strides_lst):
    index = ()


# def get_items(data, fill_value=None):
    

def get_strided_data(data):
    """
    Return strided array format data from a N-D sequence structure.

    Parameters
    ----------
    data : {scalar, sequence}

    Returns
    -------
    strides, values, shape
    """
    if is_sequence(data):
        strides = (1, )
        values = []
        shape = (len(data), )
        for i, item in enumerate(data):
            item_strides, item_values, item_shape = get_strided_data(item)
            if i == 0:
                if item_shape:
                    strides = (item_strides[0] * item_shape[0], ) + item_strides
                shape += item_shape
            else:
                assert shape[1:] == item_shape
                assert strides[1:] == item_strides
            values.extend(item_values)
        return strides, values, shape
    else:
        # scalar
        return (), [data], ()


def get_coo_data(data, fill_value=None):
    """
    Return COO sparse array format data from a N-D sequence structure.

    Parameters
    ----------
    data : {scalar, sequence}

    Returns
    -------
    indices, values, shape
    """
    if is_sequence(data):
        indices = []
        values = []
        shape = (len(data),)
        for i, item in enumerate(data):
            item_indices, item_values, item_shape = get_coo_data(item, fill_value=fill_value)
            if i == 0:
                shape += item_shape
                indices = [[] for d in shape]
            else:
                assert shape[1:] == item_shape
            indices[0].extend([i] * len(item_values))
            for d in range(len(item_shape)):
                indices[d + 1].extend(item_indices[d])            
            values.extend(item_values)
        return indices, values, shape
    else:
        if data == fill_value:
            return [], [], ()
        return [], [data], ()


def get_gcs_data(data, reduction=None, fill_value=None):    
    shape = get_shape(data)
    N = len(shape)
    # TODO: N=0, N=1
    if reduction is None:
        dims1 = tuple(range(N//2))
        dims2 = tuple(range(N//2, N))
        reduction = dims1 + dims2 + (N//2,)
        l = N // 2
    else:
        l = reduction[-1]
        dims1 = reduction[:l]
        dims2 = reduction[l:-1]

    strides1 = make_strides(shape[:l])
    strides2 = make_strides(shape[l-1:-1])
    print(f'{shape=} {strides1=} {strides2=} {dims1=} {dims2=}')
    # <row>: <list of (colindex, value)>
    col_value = defaultdict(list)
    for index in itertools.product(*map(range, shape)):
        print(index)
        v = data
        for i in index:
            v = v[i]
        if v == fill_value:
            continue
        p1 = GCS.apply_reduction(index, strides1, dims1)
        p2 = GCS.apply_reduction(index, strides2, dims2)
        col_value[p1].append((p2, v))
    ro = [0]
    co = []
    values = []
    for i in range(max(col_value)+1):
        cv = col_value.get(i, [])
        ro.append(ro[-1] + len(cv))
        cv.sort()
        c, v = zip(*cv)
        co.extend(c)
        values.extend(v)

    return ro, co, values, reduction, shape


#############################
##### Utility functions #####
#############################


def product(sequence):
    return sequence[0] * product(sequence[1:]) if sequence else 1


def is_scalar(data):
    if isinstance(data, (int, float, bool)) or data is None:
        return True
    if isinstance(data, NDArray):
        return data.ndims == 0
    if isinstance(data, (list, tuple)):
        return False
    if isinstance(data, np.ndarray):
        return False
    raise NotImplementedError(f'{type(data)=}')  # pragma: no cover


def is_sequence(data):
    if isinstance(data, (list, tuple)):
        return True
    if isinstance(data, NDArray):
        return data.ndims > 0
    if is_scalar(data):
        return False
    if isinstance(data, np.ndarray):
        return False
    raise NotImplementedError(f'{type(data)=}')  # pragma: no cover


######################
##### Unit-tests #####
######################


def test_scalar_api():
    arrays = []
    a = Strided.from_data(123)
    assert a._strides == ()
    assert a._values == [123]
    arrays.append(a)

    a = COO.from_data(123)
    assert a._indices == []
    assert a._values == [123]
    arrays.append(a)

    for a in arrays:
        assert a.shape == ()
        assert a.ndims == 0
        assert a.numel == 1
        assert a[()] == 123
        assert tuple(zip(*a.items())) == (((), ), (123,))
        assert str(a) == '123'

        assert a == 123
        assert a != 125
        for b in arrays:
            assert a == b

        for acls in [Strided, COO]:
            b = acls.from_data(a)
            assert isinstance(b, acls)
            assert a == b

def test_dense_array_api():
    arrays = []

    a = Strided.from_data([[1, 2], [3, 4]])
    assert a._strides == (2, 1)
    assert a._values == [1, 2, 3, 4]
    assert repr(a) == 'Strided((2, 1), [1, 2, 3, 4], (2, 2))'
    arrays.append(a)
    
    a = COO.from_data([[1, 2], [3, 4]])
    assert a._indices == [[0, 0, 1, 1], [0, 1, 0, 1]]
    assert a._values == [1, 2, 3, 4]
    assert a._fill_value == None
    assert repr(a) == 'COO([[0, 0, 1, 1], [0, 1, 0, 1]], [1, 2, 3, 4], (2, 2), fill_value=None)'
    arrays.append(a)

    a = GCS.from_data([[1, 2], [3, 4]])

    print(a)
    
    for a in arrays:
        assert a.shape == (2, 2)
        assert a.ndims == 2
        assert a.numel == 4
        assert a[0, 0] == 1
        assert a[0, 1] == 2
        assert a[1, 0] == 3
        assert a[1, 1] == 4
        assert tuple(zip(*a.items())) == (((0, 0), (0, 1), (1, 0), (1, 1)), (1, 2, 3, 4))
        assert str(a) == '[[1, 2], [3, 4]]'

        assert a == [[1, 2], [3, 4]]
        assert a != [[1, 2], [3, 5]]
        for b in arrays:
            assert a == b

        for acls in [Strided, COO]:
            b = acls.from_data(a)
            assert isinstance(b, acls)
            assert a == b

        assert a[0] == [1, 2]
        assert a[1] == [3, 4]


def test_sparse_array_api():
    arrays = []

    a = COO([[0, 0, 1], [0, 1, 1]], [1, 2, 4], (2, 2))
    assert a._fill_value == None
    assert tuple(zip(*a.items())) == (((0, 0), (0, 1), (1, 1)), (1, 2, 4))
    arrays.append(a)

    a = Strided.from_data(arrays[0])
    assert a._strides == (2, 1)
    assert a._values == [1, 2, None, 4]
    assert tuple(zip(*a.items())) == (((0, 0), (0, 1), (1, 0), (1, 1)), (1, 2, None, 4))
    arrays.append(a)

    a = COO.from_data(arrays[-1])
    assert a._fill_value == None
    assert a._indices == [[0, 0, 1], [0, 1, 1]]
    assert a._values == [1, 2, 4]
    arrays.append(a)

    for a in arrays:
        assert a.shape == (2, 2)
        assert a.ndims == 2
        assert a.numel == 4
        assert a[0, 0] == 1
        assert a[0, 1] == 2
        assert a[1, 0] == None
        assert a[1, 1] == 4
        assert str(a) == '[[1, 2], [None, 4]]'

        assert a == [[1, 2], [None, 4]]
        assert a != [[1, 2], [None, 5]]
        for b in arrays:
            assert a == b

        for acls in [Strided, COO]:
            b = acls.from_data(a)
            assert isinstance(b, acls)
            assert a == b

        assert a[0] == [1, 2]
        assert a[1] == [None, 4]
