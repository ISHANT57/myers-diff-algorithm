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

The edit graph (the mental picture for everything below)
  Draw A along the x axis and B along the y axis. A path from the top-left
  (0, 0) to the bottom-right (n, m) is an edit script:
      move right  (x+1)        = delete A[x]
      move down   (y+1)        = insert B[y]
      move diagonal (x+1,y+1)  = keep (only allowed where A[x] == B[y]; it is FREE)
  Fewest edits = path with the fewest right/down moves = most diagonal moves.
  The diagonal moves on that path are exactly the LCS, i.e. the kept lines.

File map (read in this order)
  read_lines      bytes -> list of lines
  middle_snake    the heart: find the middle of an optimal path (forward + backward)
  solve           divide and conquer driver around middle_snake
  lcs_match       solve() plus a speed-up (drop items that cannot match)
  to_ranges       indexes -> "3-5,9-12"            (Part B helper)
  char_ranges     two lines -> changed-char ranges (Part B)
  build_diff      LCS -> printable diff lines, with delete-first ordering
  main            arguments, file errors (exit 2), printing
  (a quiz cheat sheet is at the very bottom of this file)
"""
import sys


# ---------------------------------------------------------------- reading input

def read_lines(path):
    """Read a file as raw bytes and split it into lines (without the newline).

    Split on b'\\n'; drop the last piece if empty (so a final newline adds no
    line and an empty file has none); keep any b'\\r' as part of its line.

    Why bytes, not text: text mode would turn \\r\\n into \\n (so "a\\r\\n" and
    "a\\n" would look equal, but the task says they differ), and some test
    files are not valid UTF-8. Comparing raw bytes is always exact.

    Examples:  b""        -> []          b"\\n"      -> [b""]
               b"a\\n\\nb"  -> [b"a", b"", b"b"]     b"a\\r\\nb" -> [b"a\\r", b"b"]
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

    ---- How the search works ----
    Round d means "paths using exactly d edits". After d edits a path can only
    be on diagonals start-d, start-d+2, ..., start+d (each edit moves k by 1,
    so after d edits k has the same parity as d). That is why every loop below
    steps k by 2.
    On each diagonal we only remember the FURTHEST x reached (vf[k]); a path
    further along the same diagonal is always at least as good. Then we
    "follow the snake": slide down the diagonal for free while lines match.

    Forward step onto diagonal k, two ways to arrive:
        from k+1 by moving down  (insert): x stays the same  -> x = vf[k+1]
        from k-1 by moving right (delete): x grows by one    -> x = vf[k-1] + 1
    Take the one with the larger x. At the edge of the round (k == first or
    k == last) only one neighbour exists, so there is no choice.

    ---- Why the two searches meet ----
    Forward needs ~D/2 rounds and backward needs ~D/2 rounds, so each one only
    explores a small triangle instead of one big one. When a forward point and
    a backward point lie on the same diagonal and have crossed (backward x <=
    forward x), the two paths join into one full path of D edits. The snake
    where they join is returned; the total is 2d-1 edits when they meet in the
    forward pass (odd delta) and 2d when they meet in the backward pass (even
    delta).

    ---- Traced example (A = a b c a b b a, B = c b a b a c) ----
    Window is the whole grid, so fwd_start = 0, bwd_start = 7 - 6 = 1, delta = 1
    (odd => meeting is detected in the FORWARD pass). Values are x per diagonal:
        d=0  fwd {0: 0}                  bwd {1: 7}
        d=1  fwd {-1: 0, 1: 1}           bwd {0: 6, 2: 5}
        d=2  fwd {-2: 2, 0: 2, 2: 3}     bwd {-1: 5, 1: 3, 3: 4}
        d=3  fwd {-3: 3, -1: 4, 1: 5}    <- diagonal 1: vb[1] = 3 <= x = 5, MEET
    Snake returned: (3, 2) -> (5, 4), the lines "a b" (A[3:5] == B[2:4]).
    Total edits = 2*3 - 1 = 5, the known minimum for this example.
    """
    fwd_start = alo - blo              # diagonal where the forward search starts
    bwd_start = ahi - bhi              # diagonal where the backward search starts
    # delta = difference of the two start diagonals = (len of A part) - (len of B part).
    # Its parity decides in which pass the two searches can first touch (see below).
    delta_is_odd = (bwd_start - fwd_start) & 1
    # Seeds: round d=0 reads vf[k + 1] / vb[k - 1] for its single diagonal, so we
    # plant a value there that makes d=0 start exactly at (alo, blo) / (ahi, bhi).
    vf[fwd_start + 1] = alo            # seed so round d=0 begins at (alo, blo)
    vb[bwd_start - 1] = ahi            # seed so round d=0 begins at (ahi, bhi)

    # Upper bound on rounds: ceil((n + m) / 2). The searches always meet before
    # that, so this loop always returns (it never falls through).
    for d in range((ahi - alo + bhi - blo + 1) // 2 + 1):

        # ---- forward round: diagonals fwd_start-d .. fwd_start+d, step 2 ----
        first, last = fwd_start - d, fwd_start + d
        # `left` / `right` are vf[k - 1] / vf[k + 1]. They are carried from one k
        # to the next (left = old right) so each array cell is read only once.
        # Those cells belong to round d-1 (other parity), so writing vf[k] below
        # never overwrites a value we still need in this round.
        left = vf[first - 1]                       # vf[k - 1], carried along
        for k in range(first, last + 1, 2):
            right = vf[k + 1]                      # vf[k + 1]
            if k == first or (k != last and left < right):
                x = right                          # come from k+1: insert (move down)
            else:
                x = left + 1                       # come from k-1: delete (move right)
            left = right                           # becomes vf[k - 1] for k + 2
            y = x - k                              # k = x - y  =>  y = x - k
            start_x, start_y = x, y                # where the snake begins (returned on a meet)
            while x < ahi and y < bhi and a[x] == b[y]:    # follow the snake
                x += 1
                y += 1
            vf[k] = x                              # furthest x on diagonal k after d edits
            # Odd delta: the forward path can meet the backward path here.
            # The `bwd_start - d < k < bwd_start + d` test asks "did the backward
            # search (which has finished round d-1) reach this diagonal?"; if so and
            # vb[k] <= x, the backward path is already at or before us: they overlap.
            if delta_is_odd and bwd_start - d < k < bwd_start + d and vb[k] <= x:
                return start_x, start_y, x, y

        # ---- backward round: mirror image, walking up-left from the end ----
        first, last = bwd_start - d, bwd_start + d
        # Same idea mirrored: backward moves go UP (y-1, insert) or LEFT (x-1,
        # delete) and we keep the SMALLEST x (vb[k]), so comparisons flip.
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
            # Even delta: the backward path can meet the forward path here. Forward
            # has already finished round d, so its diagonals are start +- d.
            if not delta_is_odd and fwd_start - d <= k <= fwd_start + d and vf[k] >= x:
                return x, y, start_x, start_y


def solve(a, b):
    """LCS of two lists. Returns, for each index i of a, its partner index in b
    (or -1 if a[i] is not part of the LCS).

    Works through a stack of windows (alo, ahi, blo, bhi). Each window: trim the
    matching ends, find the middle snake, record it, then queue the two halves.

    Why divide and conquer: classic Myers keeps a copy of V for every round
    (O(D^2) memory) so it can walk back. Here we instead find one point of an
    optimal path (the middle snake) and recurse on the two halves, so memory is
    O(N) and time is still O(ND).

    Why a stack instead of recursion: no recursion-depth limit to worry about.
    Why trimming: it costs nothing, it shrinks D, and after trimming a
    non-empty window needs at least 2 edits, which guarantees each half is
    strictly smaller (so the loop always ends).
    The two V arrays (vf, vb) are allocated once and reused by every window.
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
            continue                               # (all of the other side is delete/insert)

        x, y, u, v = middle_snake(a, b, alo, ahi, blo, bhi, vf, vb)
        for step in range(u - x):                  # every step of the snake is a match
            partner[x + step] = y + step
        windows.append((alo, x, blo, y))           # left half
        windows.append((u, ahi, v, bhi))           # right half
    return partner


def lcs_match(a, b):
    """Same result as solve(), but faster: first drop items that occur in only
    one list (they can never match), solve the smaller problem, then map the
    answer back to the original positions.

    Why this is safe: a line that exists in only one file can never be in a
    common subsequence, so it is a pure delete/insert in every edit script.
    Removing it does not change the minimum, and on real files it removes most
    of the changed lines, which makes D (and the running time) much smaller.
    """
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
    that touch. Returns '.' when there are none.

    Examples: [3, 4, 5, 9]  -> "3-6,9-10"      (end is excluded: 3-6 = chars 3,4,5)
              [11]          -> "11-12"
              []            -> "."
    """
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

    Steps: decode bytes -> str so indexes count code points (an emoji is ONE
    character, not 4 bytes); run the same lcs_match on the two strings; every
    position with no partner is a changed character; to_ranges merges them.
    'surrogateescape' keeps undecodable bytes instead of crashing (highlight
    tests only use valid UTF-8, so this is just a safety net).
    Removing the changed characters leaves identical text, and since the LCS is
    the longest, the number of changed characters is the minimum possible.
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
    """Return the output as a list of byte strings, one per output line.

    Idea: the kept lines (LCS) split both files into "change blocks". Between
    two consecutive kept lines, A has some lines only in A (deleted) and B has
    some lines only in B (inserted). Example:
        A = [x, p, q, y]   B = [x, r, y]   kept = x, y
        block between them: deleted [p, q], inserted [r]
        output:  " x", "-p", "-q", "+r", " y"
    Printing all deletions of a block before its insertions is the
    "delete-first rule". In highlight mode the t-th '+' of a block is paired
    with the t-th '-' and followed by a "? old | new" line.
    """
    # Intern: each distinct line gets a small integer, so equal lines compare
    # as ints (fast) instead of comparing byte strings again and again.
    line_ids = {}                                  # distinct line -> small int
    ids_a = [line_ids.setdefault(line, len(line_ids)) for line in lines_a]
    ids_b = [line_ids.setdefault(line, len(line_ids)) for line in lines_b]
    partner = lcs_match(ids_a, ids_b)

    out = []
    next_a = next_b = 0                            # first line not yet printed
    # Each kept line ends a change block. The end of the file closes the last one.
    # (Sentinel: len(lines_a) acts as a fake kept line at the very end.)
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
    # Exit code 2 = bad usage or an unreadable file. In that case NOTHING may be
    # printed on stdout (the message goes to stderr), as the task requires.
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


# ============================================================================
# QUIZ CHEAT SHEET  (comments only; nothing here runs)
# ============================================================================
# One-sentence answers, then say the "why" in your own words.
#
#  What does vf[k] mean?      Furthest x reached on diagonal k (k = x - y) by a
#                             forward path using d edits.
#  What does vb[k] mean?      Smallest x reached on diagonal k by a backward path
#                             (searching from the bottom-right corner).
#  What is a snake?           A run of diagonal moves = consecutive matching lines.
#                             They are free (cost no edit).
#  Which line follows a snake? `vf[k] = x` (forward) / `vb[k] = x` (backward):
#                             store where the snake ended.
#  Why step k by 2?           After d edits, k has the same parity as d.
#  Why is the answer minimal? Myers explores d = 0, 1, 2, ... so the first time
#                             the searches meet is with the fewest edits.
#  Why two searches?          Each needs only ~D/2 rounds, and meeting in the
#                             middle lets us split the problem and keep O(N) memory.
#  Why is memory O(N)?        We never keep V for every round; vf and vb are one
#                             flat array each, reused by every window.
#  Time?                      O(ND), N = lines, D = number of edits.
#  Why 2d-1 vs 2d?           Odd delta: meet in the forward pass (2d-1 edits).
#                             Even delta: meet in the backward pass (2d edits).
#  Why trim head/tail?        Free matches; makes the problem smaller; guarantees
#                             progress (a non-empty window then needs >= 2 edits).
#  Why drop unique lines?     They can never match, so they are always deleted or
#                             inserted; removing them shrinks D safely.
#  Why ints for lines?        Fast comparison; same results as comparing bytes.
#  Why read bytes?            Exact comparison, keeps \r, handles invalid UTF-8.
#  Delete-first rule?         build_diff prints all '-' of a block, then all '+'.
#  How are '?' lines built?   char_ranges: LCS on characters; unmatched positions
#                             -> to_ranges merges touching ones (end excluded).
#  How would you change X?    Tie rule between "insert" and "delete": the
#                             `left < right` test in middle_snake (result stays
#                             minimal, only WHICH minimal diff changes).
#                             Pairing of '-' with '+': the `t < len(deleted)` loop
#                             in build_diff.
