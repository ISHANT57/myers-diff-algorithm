"""Myers diff. Usage: main.py lines A B | main.py highlight A B

Linear-space Myers (middle snake, divide and conquer): O(ND) time, O(N) memory.
"""
import sys


def snake(a, b, alo, ahi, blo, bhi, vf, vb):
    """Middle snake of a[alo:ahi] vs b[blo:bhi], in absolute coordinates.
    Returns (x, y, u, v): a diagonal run (x, y) -> (u, v) on an optimal path.
    Diagonal k = x - y (negative k wraps around the list, which is fine because
    the lists are long enough). vf[k] = furthest x reached going forward from
    the top-left; vb[k] = smallest x reached going backward from the bottom-right."""
    k0, k1 = alo - blo, ahi - bhi          # start diagonals of forward / backward search
    odd = (k1 - k0) & 1
    vf[k0 + 1] = alo
    vb[k1 - 1] = ahi
    for d in range((ahi - alo + bhi - blo + 1) // 2 + 1):
        lo, hi = k0 - d, k0 + d
        left = vf[lo - 1]
        for k in range(lo, hi + 1, 2):                     # forward pass
            right = vf[k + 1]
            if k == lo or (k != hi and left < right):
                x = right                                  # step down: insert
            else:
                x = left + 1                               # step right: delete
            left = right                                   # next k's vf[k - 1]
            y = x - k
            sx, sy = x, y
            while x < ahi and y < bhi and a[x] == b[y]:    # follow snake
                x += 1
                y += 1
            vf[k] = x
            # odd delta: forward path can meet backward path here
            if odd and k1 - d < k < k1 + d and vb[k] <= x:
                return sx, sy, x, y
        lo, hi = k1 - d, k1 + d
        left = vb[lo - 1]
        for k in range(lo, hi + 1, 2):                     # backward pass
            right = vb[k + 1]
            if k == hi or (k != lo and left < right - 1):
                x = left                                   # step up: insert
            else:
                x = right - 1                              # step left: delete
            left = right
            y = x - k
            sx, sy = x, y
            while x > alo and y > blo and a[x - 1] == b[y - 1]:
                x -= 1
                y -= 1
            vb[k] = x
            # even delta: backward path can meet forward path here
            if not odd and k0 - d <= k <= k0 + d and vf[k] >= x:
                return x, y, sx, sy


def solve(a, b):
    """For each index i of a: matching index in b (LCS pairing), or -1."""
    n, m = len(a), len(b)
    res = [-1] * n
    size = 2 * (n + m + 4) + 1                             # diagonals run -(n+m)..(n+m)
    vf = [0] * size
    vb = [0] * size
    stack = [(0, n, 0, m)]
    while stack:
        alo, ahi, blo, bhi = stack.pop()
        while alo < ahi and blo < bhi and a[alo] == b[blo]:        # trim common head
            res[alo] = blo
            alo += 1
            blo += 1
        while alo < ahi and blo < bhi and a[ahi - 1] == b[bhi - 1]:  # trim common tail
            ahi -= 1
            bhi -= 1
            res[ahi] = bhi
        if alo == ahi or blo == bhi:                               # one side empty: no matches
            continue
        x, y, u, v = snake(a, b, alo, ahi, blo, bhi, vf, vb)
        for t in range(u - x):                                     # snake = matches
            res[x + t] = y + t
        stack.append((alo, x, blo, y))                             # left half
        stack.append((u, ahi, v, bhi))                             # right half
    return res


def lcs_match(a, b):
    """Like solve, but first drops items that appear in only one sequence
    (they can never match), which shrinks the problem a lot on real files."""
    sa, sb = set(a), set(b)
    ia = [i for i in range(len(a)) if a[i] in sb]
    ib = [j for j in range(len(b)) if b[j] in sa]
    small = solve([a[i] for i in ia], [b[j] for j in ib])
    match = [-1] * len(a)
    for x, y in enumerate(small):
        if y >= 0:
            match[ia[x]] = ib[y]
    return match


def read_lines(path):
    """Raw bytes, split on b'\\n', last empty piece dropped, \\r kept."""
    with open(path, 'rb') as f:
        parts = f.read().split(b'\n')
    if parts[-1] == b'':
        parts.pop()
    return parts


def ranges(unmatched):
    """Sorted indices -> 'a-b,c-d' with touching ranges merged, '.' if empty."""
    out, start, prev = [], None, None
    for i in unmatched:
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            out.append(f'{start}-{prev + 1}')
            start = prev = i
    if start is not None:
        out.append(f'{start}-{prev + 1}')
    return ','.join(out) or '.'


def char_ranges(old, new):
    """Changed-character ranges (code points) of one line pair."""
    o = old.decode('utf-8', 'surrogateescape')
    n = new.decode('utf-8', 'surrogateescape')
    match = lcs_match(o, n)
    used = set(match)
    return (ranges([i for i, j in enumerate(match) if j < 0]),
            ranges([j for j in range(len(n)) if j not in used]))


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ('lines', 'highlight'):
        sys.stderr.write('usage: main.py lines|highlight A B\n')
        return 2
    try:
        A, B = read_lines(sys.argv[2]), read_lines(sys.argv[3])
    except OSError as e:
        sys.stderr.write(f'error: {e}\n')
        return 2
    hl = sys.argv[1] == 'highlight'

    ids = {}                                       # intern lines -> ints (fast compares)
    a = [ids.setdefault(l, len(ids)) for l in A]
    b = [ids.setdefault(l, len(ids)) for l in B]
    match = lcs_match(a, b)

    out = []
    i = j = 0
    # walk matched pairs in order; between them is one change block
    for mi in [x for x in range(len(A)) if match[x] >= 0] + [len(A)]:
        mj = match[mi] if mi < len(A) else len(B)
        dels, adds = A[i:mi], B[j:mj]
        out += [b'-' + l for l in dels]            # delete-first rule
        for t, l in enumerate(adds):
            out.append(b'+' + l)
            if hl and t < len(dels):               # pair t-th '-' with t-th '+'
                co, cn = char_ranges(dels[t], l)
                out.append(f'? {co} | {cn}'.encode())
        if mi < len(A):
            out.append(b' ' + A[mi])
        i, j = mi + 1, mj + 1
    if out:
        sys.stdout.buffer.write(b'\n'.join(out) + b'\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
