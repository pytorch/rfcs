"""
Reduction of slice objects
"""
# Author: Pearu Peterson
# Created: June 2020


def reduced(s, size):
    """Return reduced slice with respect to given size.

    Notes
    -----

    If `x = list(range(size))` then `x[s] == x[reduced(s, size)]` for
    any slice instance `s`.

    """
    start, stop, step = s.start, s.stop, s.step
    if step is None:
        step = 1
    if step > 0:
        if start is None or start < -size:
            start = 0
        elif start < 0:
            start += size
        elif start > size:
            start = size
        if stop is None or stop > size:
            stop = size
        elif stop < -size:
            stop = 0
        elif stop < 0:
            stop += size
        if step > size:
            step = size
    else:
        if start is None or start >= size:
            start = -1
        elif start < -size:
            start = -size - 1
        elif start >= 0:
            start -= size
        if stop is None or stop < -size:
            stop = -size - 1
        elif stop >= size:
            stop = -1
        elif stop >= 0:
            stop -= size
        if step < -size:
            step = -size
    if (stop - start) * step <= 0:
        start = stop = 0
        step = 1
    return slice(start, stop, step)


def nitems(s, size):
    """Return the number of items for given size.

    Notes
    -----

    1. If `x = list(range(size))` and `s = reduced(s, size)` then
      `len(x[s]) == nitems(s, size)` for any slice instance `s`.

    2. If `s = reduced(s, size)` then `x[s] = [(s.start + i * s.step)
      % size for i in range(nitems(s, size))]`.

    """
    s = reduced(s, size)
    if s.stop == s.start:
        return 0
    sgn = s.step // abs(s.step)
    return max(0, ((s.stop - s.start) + s.step - sgn) // s.step)


def strides_from_shape(shape):
    strides = [1] * len(shape)
    for i in reversed(range(len(shape) - 1)):
        strides[i] = strides[i+1] * shape[i+1]
    return tuple(strides)


def dimensionreduction1(indices, strides):
    return sum([strides[i] * indices[i] for i in range(len(indices))])


def dimensionpromotion1(index, strides, shape):
    indices = [None] * len(shape)
    for i in reversed(range(len(shape))):
        # indices[i] = ((index - sum([indices[k] * strides[k]
        #                            for k in range(i+1, len(shape))]))
        #               // strides[i] % shape[i])
        indices[i] = index // strides[i] % shape[i]
    return tuple(indices)


def test_reduced():
    for d in range(1, 10):
        for start in list(range(-d-10, d+10)) + [None]:
            for stop in list(range(-d-10, d+10)) + [None]:
                for step in list(range(-d-10, d+10)) + [None]:
                    if step == 0:
                        continue
                    s = slice(start, stop, step)
                    r = list(range(d))[s]

                    s2 = reduced(s, d)
                    r2 = list(range(d))[s2]

                    assert r == r2, (s, r, s2, r2, d)
                    assert nitems(s, d) == len(r), (s, s2, r, d)

                    r3 = [(s2.start % d + i * s2.step)
                          for i in range(nitems(s2, d))]
                    assert r == r3, (s, r, r3, d)

                    if s2.step > 0:
                        r4 = [i for i in range(d)
                              if (i >= s2.start
                                  and i < s2.stop
                                  and (i - s2.start) % s2.step == 0)]
                    else:
                        r4 = [d - i - 1
                              for i in range(d)
                              if (i >= -s2.start - 1
                                  and i < -s2.stop - 1
                                  and (i + s2.start + 1) % s2.step == 0)]
                    assert r == r4, (r, r4, s2, d)


def test_dimensionpromotion1():
    import itertools
    max_dimension = 5

    for ndims in range(1, 5+1):
        for shape in itertools.product(
                *([tuple(range(1, max_dimension+1))]*ndims)):
            strides = strides_from_shape(shape)
            for indices in itertools.product(
                    *([tuple(range(dim)) for dim in shape])):
                index = dimensionreduction1(indices, strides)
                indices2 = dimensionpromotion1(index, strides, shape)
                assert indices == indices2
