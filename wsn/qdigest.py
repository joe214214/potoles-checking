"""
Q-Digest: Approximate quantile summary over an integer domain [0, U).

Reference: Shrivastava et al., "Medians and Beyond: New Aggregation
Techniques for Sensor Networks," SenSys 2004.

Key idea
--------
Q-Digest stores a binary tree over the integer range [0, U) where each
non-zero node (lo, hi) holds the count of values that fell in [lo, hi]
but could NOT be merged further up without violating the accuracy bound.

Invariant (for every non-root node v):
    count(v) + count(parent(v)) + count(sibling(v)) > n / k

Compression algorithm (bottom-up):
    For each non-root node v (leaf → root direction):
        if count(v) + count(parent(v)) + count(sibling(v)) <= n / k:
            count(parent(v)) += count(v)
            remove v from tree

After compression, the non-zero nodes form a partial partition of [0, U)
that covers every inserted value exactly once.

Merge of two digests D1, D2:
    D_merged = D1 ⊕ D2   where  count_merged(v) = count1(v) + count2(v)
    then compress D_merged with the combined n = n1 + n2.

Quantile query:
    Traverse nodes in left-to-right order (sorted by lo).
    Accumulate counts; return the midpoint of the node where the running
    sum first reaches q * n.

Space: O(k log U) nodes.
Error: every quantile estimate is within ±(n / k) rank error.

Public API
----------
QDigest(U, k)       integer digest over [0, U), compression param k
ScalarQDigest(...)  floating-point wrapper via linear discretisation
"""

import math


# ─────────────────────────────────────────────────────────────────────────────
# Core integer Q-Digest
# ─────────────────────────────────────────────────────────────────────────────

class QDigest:
    """
    Approximate quantile digest for integer values in [0, U).

    Parameters
    ----------
    U : int  — Universe size; values are clamped to [0, U-1].
               Internally rounded up to the next power of two.
    k : int  — Compression factor (higher → more accurate, more nodes).
               Typical range: 10–50.
    """

    def __init__(self, U: int, k: int = 20):
        # Round U up to the smallest power of two ≥ U (min 2)
        self._U = 1
        while self._U < max(int(U), 2):
            self._U <<= 1
        self._depth = int(math.log2(self._U))   # depth of the full binary tree
        self._k = int(k)
        self._n: int = 0                         # total inserted count
        self._tree: dict = {}                    # (lo, hi) -> int count

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def total(self) -> int:
        """Total number of values inserted (before any drops)."""
        return self._n

    # ── Insertion ─────────────────────────────────────────────────────────────

    def insert(self, value) -> None:
        """
        Insert a single integer value (clamped to [0, U-1]).
        No automatic compression — call compress() when needed.
        """
        vi = max(0, min(int(value), self._U - 1))
        node = (vi, vi)
        self._tree[node] = self._tree.get(node, 0) + 1
        self._n += 1

    # ── Compression ───────────────────────────────────────────────────────────

    def compress(self) -> None:
        """
        Apply Q-digest compression bottom-up.
        Should be called after a batch of inserts or after a merge.
        """
        if self._n == 0:
            return
        threshold = self._n / self._k

        # Traverse depth-by-depth from leaves (depth = self._depth) to
        # children of root (depth = 1).  Root itself is never moved.
        for d in range(self._depth, 0, -1):
            node_size = self._U >> d       # each node at depth d covers this range
            num_nodes = 1 << d             # 2^d nodes at depth d
            for i in range(num_nodes):
                lo = i * node_size
                hi = lo + node_size - 1
                node = (lo, hi)
                count = self._tree.get(node, 0)
                if count == 0:
                    continue
                parent  = self._parent(lo, hi)
                sibling = self._sibling(lo, hi)
                p_count = self._tree.get(parent, 0)
                s_count = self._tree.get(sibling, 0)
                if count + p_count + s_count <= threshold:
                    # Merge this node's count into its parent
                    self._tree[parent] = p_count + count
                    del self._tree[node]

        # Remove any zero-count entries that might linger
        zero_keys = [k for k, v in self._tree.items() if v == 0]
        for k in zero_keys:
            del self._tree[k]

    # ── Merge ─────────────────────────────────────────────────────────────────

    def merge(self, other: "QDigest") -> None:
        """
        Merge another QDigest into this one in-place (C12 = C1 + C2 rule),
        then compress.  Both digests must use the same U.
        """
        if other._n == 0:
            return
        if other._U != self._U:
            raise ValueError(
                f"Cannot merge QDigests with different U: {self._U} vs {other._U}"
            )
        for node, count in other._tree.items():
            self._tree[node] = self._tree.get(node, 0) + count
        self._n += other._n
        self.compress()

    # ── Queries ───────────────────────────────────────────────────────────────

    def quantile(self, q: float) -> float:
        """
        Return the approximate q-th quantile (0 < q ≤ 1).

        Algorithm:
          Sort non-zero nodes by their left endpoint (lo) — this gives a
          left-to-right traversal of the partial partition.  Accumulate
          counts; return the midpoint of the first node whose running sum
          reaches q * n.
        """
        if self._n == 0:
            return 0.0
        q = max(0.0, min(float(q), 1.0))
        target = q * self._n

        # Sort by lo (primary), hi (secondary, finer intervals first)
        nodes_sorted = sorted(self._tree.items(),
                               key=lambda x: (x[0][0], x[0][1]))
        cumsum = 0.0
        for (lo, hi), count in nodes_sorted:
            cumsum += count
            if cumsum >= target:
                return (lo + hi) / 2.0   # midpoint of the matching interval
        return float(self._U - 1)

    def count_le(self, v) -> int:
        """
        Approximate number of inserted values that are ≤ v.
        Sums counts of all nodes whose entire interval lies within [0, v].
        """
        vi = int(v)
        total = 0
        for (lo, hi), count in self._tree.items():
            if hi <= vi:
                total += count
        return total

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def size(self) -> int:
        """Number of non-zero nodes currently stored."""
        return len(self._tree)

    def nodes(self):
        """Return a sorted list of (lo, hi, count) triples (for inspection)."""
        return sorted(
            [(lo, hi, c) for (lo, hi), c in self._tree.items()],
            key=lambda x: (x[0], x[1])
        )

    def __repr__(self):
        return (f"QDigest(U={self._U}, k={self._k}, "
                f"n={self._n}, nodes={self.size()})")

    # ── Internal tree helpers ─────────────────────────────────────────────────

    def _parent(self, lo: int, hi: int):
        """Return the (lo, hi) of the parent node in the binary tree."""
        length = hi - lo + 1
        parent_len = length * 2
        parent_lo = (lo // parent_len) * parent_len
        return (parent_lo, parent_lo + parent_len - 1)

    def _sibling(self, lo: int, hi: int):
        """Return the (lo, hi) of the sibling node."""
        length = hi - lo + 1
        parent_lo = (lo // (length * 2)) * (length * 2)
        if parent_lo == lo:
            # This node is the left child; sibling is the right child
            return (lo + length, hi + length)
        else:
            # This node is the right child; sibling is the left child
            return (parent_lo, parent_lo + length - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Floating-point wrapper
# ─────────────────────────────────────────────────────────────────────────────

class ScalarQDigest:
    """
    Q-Digest for floating-point scalars via linear discretisation.

    Maps value ∈ [v_min, v_max] → integer bin ∈ [0, U-1]:
        bin = round(clip((value - v_min) / (v_max - v_min), 0, 1) * (U - 1))

    All quantile results are converted back to the original float scale.

    Parameters
    ----------
    v_min, v_max : float  — Expected value range.  Values outside are clamped.
    U            : int    — Discretisation resolution (bins).  Default 128.
    k            : int    — Compression factor.  Default 20.
    """

    def __init__(self, v_min: float, v_max: float,
                 U: int = 128, k: int = 20):
        self._v_min = float(v_min)
        self._v_max = float(v_max)
        self._range = max(float(v_max) - float(v_min), 1e-9)
        self._U_raw = int(U)
        self._digest = QDigest(U, k)

    # ── Conversion helpers ────────────────────────────────────────────────────

    def _to_bin(self, value: float) -> int:
        frac = (value - self._v_min) / self._range
        frac = max(0.0, min(frac, 1.0))
        return int(round(frac * (self._U_raw - 1)))

    def _from_bin(self, b: float) -> float:
        return self._v_min + (b / (self._U_raw - 1)) * self._range

    # ── Public API ────────────────────────────────────────────────────────────

    def insert(self, value: float) -> None:
        self._digest.insert(self._to_bin(value))

    def compress(self) -> None:
        self._digest.compress()

    def merge(self, other: "ScalarQDigest") -> None:
        """Merge another ScalarQDigest in-place (same v_min/v_max assumed)."""
        self._digest.merge(other._digest)

    def quantile(self, q: float) -> float:
        """Return approximate q-th quantile in original float units."""
        bin_val = self._digest.quantile(q)
        return self._from_bin(bin_val)

    def count_le(self, v: float) -> int:
        return self._digest.count_le(self._to_bin(v))

    @property
    def total(self) -> int:
        return self._digest.total

    def size(self) -> int:
        return self._digest.size()

    def __repr__(self):
        return (f"ScalarQDigest([{self._v_min:.2f}, {self._v_max:.2f}], "
                f"U={self._digest._U}, k={self._digest._k}, "
                f"n={self._digest.total}, nodes={self.size()})")
