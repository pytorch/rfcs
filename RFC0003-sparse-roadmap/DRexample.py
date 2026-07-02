
import numpy as np
import itertools

def generate_selectors(N):
    for dimensions in itertools.permutations(range(N)):
        yield [d for i, d in enumerate(dimensions)]
    
def generate_partitions(N, offset=0):
    for i in range(1, N):
        for parts in generate_partitions(N-i, offset=i):
            yield [tuple(j + offset for j in p) for p in [range(i)] + parts]
    yield [tuple(j+offset for j in range(N))]

def product(seq):
    if len(seq) == 0:
        return 1
    return seq[0] * product(seq[1:])


verbose = False
shape = (3, 2, 4)
N = len(shape)

dimension_selectors = list(generate_selectors(N))
partitions = list(generate_partitions(N))

if verbose:
    print(f'{dimension_selectors=}')
    print(f'{partitions=}')

def array_to_latex(A, name=None):
    if len(A.shape) == 2:
        lines = []
        last_i = -1
        line = []
        for i, j in itertools.product(*[tuple(range(d)) for d in A.shape]):
            if i != last_i:
                if line:
                    values = line
                    if len(values) > 10:
                        values = values[:7] + ['\\ldots'] + values[-2:]
                    lines.append(' & '.join(values))
                    line = []
                last_i = i
            line.append(str(A[i, j]))
        else:
            values = line
            if len(values) > 10:
                values = values[:7] + ['\\ldots'] + values[-2:]
            lines.append(' & '.join(values))

    elif len(A.shape) == 1:
        values = [str(v) for v in A]
        if len(values) > 10:
            values = values[:7] + ['\\ldots'] + values[-2:]
        lines = [' & '.join(values)]
    else:
        raise NotImplementedError(A.shape)
    s = '\\begin{bmatrix}\n' + '\\\\\n'.join(lines) + '\n\\end{bmatrix}' 
    if name is not None:
        return name + ' = ' + s
    return s

def kappa_to_latex(kappa):
    return '\\{' + ', '.join('(%s, %s)' % (i,v) for i, v in enumerate(kappa)) + '\\}'

def partition_to_latex(partition):
    line = []
    for p in partition:
        word = [str(k) for k in p]
        line.append('\\{' + ', '.join(word) + '\\}')
    return ' \\cup '.join(line)

data = list(range(1, product(shape) + 1))


for ik, kappa in enumerate(dimension_selectors):
    for ip, partition in enumerate(partitions):
        M = len(partition)
        #print(f'{partition=}')
        strides = [None] * N
        rshape = [None] * M
        for j in range(M):
            p = partition[j]
            strides[p[-1]] = 1
            for k in reversed(range(p[0], p[-1])):
                strides[k] = strides[k + 1] * shape[kappa[k + 1]]
            rshape[j] = product([shape[kappa[k]] for k in p])
        #print(f'{strides=} {rshape=}')

        A = np.array(data).reshape(shape)
        rA = np.zeros(rshape, dtype=A.dtype)

        for indices in itertools.product(*[tuple(range(d)) for d in shape]):
            rindices = tuple([sum([strides[k] * indices[kappa[k]]
                                   for k in range(partition[j][0], partition[j][-1]+1)])
                              for j in range(M)])
            rA[rindices] = A[indices]

        if M > 2:
            continue

        print(f'''
<img data-latex="
\\begin{{equation}}
\\label{{eq:case-{ik}-{ip}}}
\\left\\{{
\\begin{{aligned}}
\\kappa&={kappa_to_latex(kappa)}\\\\
[0, {N})&={partition_to_latex(partition)}\\\\
{array_to_latex(rA, name="A'&")}
\\end{{aligned}}
\\right.
\\end{{equation}}" src="tobecompleted.svg" alt="latex">''')

    #break
