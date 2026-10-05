"""Myers diff: minimal line diff plus changed-character highlights.

    python3 main.py lines     A B    # Part A: line diff of file A -> file B
    python3 main.py highlight A B    # Part B: same diff + changed-character ranges

How it works
  1. Every line becomes an integer id, so comparing lines is fast.
  2. We find the longest common subsequence (LCS) of the two line lists: the
     lines that stay. Everything else is a deletion (only in A) or insertion
     (only in B). Fewest edits = longest LCS.
  3. The LCS uses Myers' O(ND) algorithm in its linear-space form: search from
     both ends at once, meet in the middle (the "middle snake"), then solve the
     left and right halves separately. Memory stays O(N).
  4. Part B runs the same LCS on the characters of each changed line pair.

Terms
  x, y   position in A and in B.
  k      diagonal, k = x - y. Moving right (delete) raises k, moving down
         (insert) lowers it. A "snake" is a free run of matches along a diagonal.
  d      number of edits used so far.
"""
import sys


# ---------------------------------------------------------------- reading input

def read_lines(path):
    """Read a file as raw bytes and split it into lines (without the newline).

    Split on b'\\n'; drop the last piece if empty (so a final newline adds no
    line and an empty file has none); keep any b'\\r' as part of its line.
    """
    with open(path, 'rb') as f:
        pieces = f.read().split(b'\n')
    if pieces[-1] == b'':
        pieces.pop()
    return pieces


# ------------------------------------------------------- Myers core (the LCS)

def middle_snake(a, b, alo, ahi, blo, bhi, vf, vb):
    """Find the middle of an optimal edit path through a[alo:ahi] vs b[blo:bhi].

    Returns (x, y, u, v): a snake (run of matches) from (x, y) to (u, v), in
    absolute coordinates, that lies on a shortest edit path.

    vf[k] = furthest x reached on diagonal k searching forward from the top-left.
    vb[k] = smallest x reached on diagonal k searching backward from the bottom-right.
    (k may be negative: Python wraps negative list indexes, and the lists are
    long enough that the negative and positive ends never collide.)

    Both searches grow one edit per round d until they overlap.
    """
    fwd_start = alo - blo              # diagonal where the forward search starts
    bwd_start = ahi - bhi              # diagonal where the backward search starts
    delta_is_odd = (bwd_start - fwd_start) & 1
    vf[fwd_start + 1] = alo            # seed so round d=0 begins at (alo, blo)
    vb[bwd_start - 1] = ahi            # seed so round d=0 begins at (ahi, bhi)

    for d in range((ahi - alo + bhi - blo + 1) // 2 + 1):

        # ---- forward round: diagonals fwd_start-d .. fwd_start+d, step 2 ----
        first, last = fwd_start - d, fwd_start + d
        left = vf[first - 1]                       # vf[k - 1], carried along
        for k in range(first, last + 1, 2):
            right = vf[k + 1]                      # vf[k + 1]
            if k == first or (k != last and left < right):
                x = right                          # come from k+1: insert (move down)
            else:
                x = left + 1                       # come from k-1: delete (move right)
            left = right                           # becomes vf[k - 1] for k + 2
            y = x - k
            start_x, start_y = x, y
            while x < ahi and y < bhi and a[x] == b[y]:    # follow the snake
                x += 1
                y += 1
            vf[k] = x
            # Odd delta: the forward path can meet the backward path here.
            if delta_is_odd and bwd_start - d < k < bwd_start + d and vb[k] <= x:
                return start_x, start_y, x, y

        # ---- backward round: mirror image, walking up-left from the end ----
        first, last = bwd_start - d, bwd_start + d
        left = vb[first - 1]
        for k in range(first, last + 1, 2):
            right = vb[k + 1]
            if k == last or (k != first and left < right - 1):
                x = left                           # insert (move up)
            else:
                x = right - 1                      # delete (move left)
            left = right
            y = x - k
            start_x, start_y = x, y
            while x > alo and y > blo and a[x - 1] == b[y - 1]:   # follow snake backward
                x -= 1
                y -= 1
            vb[k] = x
            # Even delta: the backward path can meet the forward path here.
            if not delta_is_odd and fwd_start - d <= k <= fwd_start + d and vf[k] >= x:
                return x, y, start_x, start_y


def solve(a, b):
    """LCS of two lists. Returns, for each index i of a, its partner index in b
    (or -1 if a[i] is not part of the LCS).

    Works through a stack of windows (alo, ahi, blo, bhi). Each window: trim the
    matching ends, find the middle snake, record it, then queue the two halves.
    """
    n, m = len(a), len(b)
    partner = [-1] * n
    size = 2 * (n + m + 4) + 1                     # diagonals run -(n+m) .. (n+m)
    vf = [0] * size
    vb = [0] * size
    windows = [(0, n, 0, m)]
    while windows:
        alo, ahi, blo, bhi = windows.pop()

        while alo < ahi and blo < bhi and a[alo] == b[blo]:          # common head
            partner[alo] = blo
            alo += 1
            blo += 1
        while alo < ahi and blo < bhi and a[ahi - 1] == b[bhi - 1]:  # common tail
            ahi -= 1
            bhi -= 1
            partner[ahi] = bhi
        if alo == ahi or blo == bhi:               # one side empty: nothing matches
            continue

        x, y, u, v = middle_snake(a, b, alo, ahi, blo, bhi, vf, vb)
        for step in range(u - x):                  # every step of the snake is a match
            partner[x + step] = y + step
        windows.append((alo, x, blo, y))           # left half
        windows.append((u, ahi, v, bhi))           # right half
    return partner


def lcs_match(a, b):
    """Same result as solve(), but faster: first drop items that occur in only
    one list (they can never match), solve the smaller problem, then map the
    answer back to the original positions."""
    in_a, in_b = set(a), set(b)
    keep_a = [i for i in range(len(a)) if a[i] in in_b]    # original indexes kept
    keep_b = [j for j in range(len(b)) if b[j] in in_a]
    small = solve([a[i] for i in keep_a], [b[j] for j in keep_b])

    partner = [-1] * len(a)
    for small_i, small_j in enumerate(small):
        if small_j >= 0:
            partner[keep_a[small_i]] = keep_b[small_j]
    return partner


# -------------------------------------------------- Part B: character ranges

def to_ranges(indexes):
    """Sorted indexes -> 'start-end,start-end' (end excluded), merging indexes
    that touch. Returns '.' when there are none."""
    ranges = []
    start = prev = None
    for i in indexes:
        if start is None:
            start = prev = i
        elif i == prev + 1:                        # continues the current range
            prev = i
        else:                                      # gap: close it, start a new one
            ranges.append(f'{start}-{prev + 1}')
            start = prev = i
    if start is not None:
        ranges.append(f'{start}-{prev + 1}')
    return ','.join(ranges) or '.'


def char_ranges(old_line, new_line):
    """Changed characters of one paired line, as (old_ranges, new_ranges).

    Characters are Unicode code points. A character is "changed" when it is not
    in the LCS of the two lines.
    """
    old = old_line.decode('utf-8', 'surrogateescape')
    new = new_line.decode('utf-8', 'surrogateescape')
    partner = lcs_match(old, new)
    matched_in_new = set(partner)
    old_changed = [i for i in range(len(old)) if partner[i] < 0]
    new_changed = [j for j in range(len(new)) if j not in matched_in_new]
    return to_ranges(old_changed), to_ranges(new_changed)


# ------------------------------------------------------------ building output

def build_diff(lines_a, lines_b, with_highlight):
    """Return the output as a list of byte strings, one per output line."""
    line_ids = {}                                  # distinct line -> small int
    ids_a = [line_ids.setdefault(line, len(line_ids)) for line in lines_a]
    ids_b = [line_ids.setdefault(line, len(line_ids)) for line in lines_b]
    partner = lcs_match(ids_a, ids_b)

    out = []
    next_a = next_b = 0                            # first line not yet printed
    # Each kept line ends a change block. The end of the file closes the last one.
    kept_a = [i for i in range(len(lines_a)) if partner[i] >= 0] + [len(lines_a)]
    for keep_a in kept_a:
        keep_b = partner[keep_a] if keep_a < len(lines_a) else len(lines_b)
        deleted = lines_a[next_a:keep_a]           # change block between two kept lines
        inserted = lines_b[next_b:keep_b]

        out += [b'-' + line for line in deleted]   # all deletions come first
        for t, line in enumerate(inserted):
            out.append(b'+' + line)
            if with_highlight and t < len(deleted):    # t-th '+' pairs with t-th '-'
                old_r, new_r = char_ranges(deleted[t], line)
                out.append(f'? {old_r} | {new_r}'.encode())

        if keep_a < len(lines_a):
            out.append(b' ' + lines_a[keep_a])
        next_a, next_b = keep_a + 1, keep_b + 1
    return out


# ---------------------------------------------------------------------- main

def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ('lines', 'highlight'):
        sys.stderr.write('usage: main.py lines|highlight A B\n')
        return 2
    try:
        lines_a = read_lines(sys.argv[2])
        lines_b = read_lines(sys.argv[3])
    except OSError as e:
        sys.stderr.write(f'error: {e}\n')          # nothing on stdout
        return 2

    out = build_diff(lines_a, lines_b, with_highlight=(sys.argv[1] == 'highlight'))
    if out:
        sys.stdout.buffer.write(b'\n'.join(out) + b'\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
