"""
Unit tests for wsn/qdigest.py

Tests cover:
  1.  Basic insert + compress + quantile on known distributions
  2.  Merge of two digests (inter-tree merge)
  3.  Edge cases (empty digest, single value, all-same values)
  4.  ScalarQDigest float mapping
  5.  Compression invariant verification
  6.  Integration smoke-test: run wsn_main.main() on real data

Run:
    python test_qdigest.py
"""

import math
import sys
import os
import random
import traceback

# ── path setup ────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from wsn.qdigest import QDigest, ScalarQDigest

# ── helpers ───────────────────────────────────────────────────────────────────

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
_failures = []


def check(condition, name, detail=""):
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f" — {detail}" if detail else ""))
        _failures.append(name)


def approx_equal(a, b, tol):
    return abs(a - b) <= tol


# ─────────────────────────────────────────────────────────────────────────────
# 1. Basic QDigest on a uniform integer distribution
# ─────────────────────────────────────────────────────────────────────────────

def test_basic_uniform():
    print("\n[1] Basic QDigest — uniform distribution [0, 63]")
    U, k = 64, 20
    n = 1000
    qd = QDigest(U, k)
    rng = random.Random(42)
    values = [rng.randint(0, 63) for _ in range(n)]
    for v in values:
        qd.insert(v)
    qd.compress()

    check(qd.total == n, "total count after insert", f"{qd.total} != {n}")

    # Sum of all node counts must equal n
    node_sum = sum(qd._tree.values())
    check(node_sum == n, "sum of node counts == n", f"{node_sum} != {n}")

    # Median of uniform[0,63] ≈ 31.5
    q50 = qd.quantile(0.50)
    check(20 <= q50 <= 43, "Q50 in reasonable range [20,43]",
          f"got {q50:.2f}")

    # Q90 should be near 57
    q90 = qd.quantile(0.90)
    check(50 <= q90 <= 63, "Q90 in reasonable range [50,63]",
          f"got {q90:.2f}")

    # Q10 should be near 6
    q10 = qd.quantile(0.10)
    check(0 <= q10 <= 15, "Q10 in reasonable range [0,15]",
          f"got {q10:.2f}")

    # Space: should be << n nodes (compression was applied)
    check(qd.size() < n / 2, "compression reduces node count",
          f"size={qd.size()}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Merge of two digests
# ─────────────────────────────────────────────────────────────────────────────

def test_merge():
    print("\n[2] Q-Digest merge (inter-tree)")
    U, k = 64, 20
    rng = random.Random(0)

    # D1: values in [0, 31]
    d1 = QDigest(U, k)
    for _ in range(500):
        d1.insert(rng.randint(0, 31))
    d1.compress()

    # D2: values in [32, 63]
    d2 = QDigest(U, k)
    for _ in range(500):
        d2.insert(rng.randint(32, 63))
    d2.compress()

    n1, n2 = d1.total, d2.total

    # Merge d2 into d1
    d1.merge(d2)

    check(d1.total == n1 + n2, "total after merge", f"{d1.total} != {n1 + n2}")

    node_sum = sum(d1._tree.values())
    check(node_sum == d1.total, "node sum == total after merge",
          f"{node_sum} != {d1.total}")

    # Combined distribution is uniform[0,63], so Q50 ≈ 31.5
    q50 = d1.quantile(0.50)
    check(20 <= q50 <= 43, "Q50 of merged digest",
          f"got {q50:.2f}")

    # Q25 should be near 15 (lower quartile)
    q25 = d1.quantile(0.25)
    check(q25 < 32, "Q25 of merged is in lower half",
          f"got {q25:.2f}")

    # Q75 should be in upper half
    q75 = d1.quantile(0.75)
    check(q75 >= 32, "Q75 of merged is in upper half",
          f"got {q75:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Edge cases
# ─────────────────────────────────────────────────────────────────────────────

def test_edge_cases():
    print("\n[3] Edge cases")

    # Empty digest
    qd = QDigest(64, 10)
    check(qd.quantile(0.5) == 0.0, "empty digest: quantile returns 0")
    check(qd.count_le(10)  == 0,   "empty digest: count_le returns 0")
    check(qd.size()        == 0,   "empty digest: size is 0")

    # Single value
    qd2 = QDigest(64, 10)
    qd2.insert(7)
    qd2.compress()
    check(qd2.total == 1, "single value: total == 1")
    q50 = qd2.quantile(0.5)
    check(q50 == 7.0, "single value 7: Q50 == 7", f"got {q50}")

    # All same value (all 5s)
    qd3 = QDigest(64, 10)
    for _ in range(200):
        qd3.insert(5)
    qd3.compress()
    check(qd3.total == 200, "all-same: total == 200")
    q50 = qd3.quantile(0.5)
    check(q50 == 5.0, "all-same value 5: Q50 == 5", f"got {q50}")

    # count_le correctness on clean data
    qd4 = QDigest(16, 10)
    for v in range(8):          # insert 0..7, one each
        qd4.insert(v)
    qd4.compress()
    check(qd4.total == 8, "count_le test: total == 8")
    le3 = qd4.count_le(3)
    check(3 <= le3 <= 5, "count_le(3) in [3,5]", f"got {le3}")

    # U not a power of two — should round up
    qd5 = QDigest(50, 10)     # rounds up to 64
    check(qd5._U == 64, "U=50 rounds up to 64")


# ─────────────────────────────────────────────────────────────────────────────
# 4. ScalarQDigest
# ─────────────────────────────────────────────────────────────────────────────

def test_scalar_qdigest():
    print("\n[4] ScalarQDigest (float values)")
    rng = random.Random(7)

    # Values in [0.0, 5.0), uniform
    sqd = ScalarQDigest(0.0, 5.0, U=128, k=20)
    n = 1000
    vals = [rng.uniform(0.0, 5.0) for _ in range(n)]
    for v in vals:
        sqd.insert(v)
    sqd.compress()

    check(sqd.total == n, "ScalarQDigest total", f"{sqd.total} != {n}")

    # Q50 ≈ 2.5, allow ±0.5
    q50 = sqd.quantile(0.50)
    check(abs(q50 - 2.5) < 0.6, "ScalarQDigest Q50 ≈ 2.5",
          f"got {q50:.4f}")

    # Q90 ≈ 4.5, allow ±0.5
    q90 = sqd.quantile(0.90)
    check(4.0 <= q90 <= 5.0, "ScalarQDigest Q90 ≈ 4.5",
          f"got {q90:.4f}")

    # Values are in the declared range
    check(0.0 <= q50 <= 5.0, "Q50 within range [0, 5]")
    check(0.0 <= q90 <= 5.0, "Q90 within range [0, 5]")

    # Merge test
    sqd2 = ScalarQDigest(0.0, 5.0, U=128, k=20)
    for v in vals:
        sqd2.insert(v + 0.0)   # same distribution
    sqd2.compress()

    sqd.merge(sqd2)
    check(sqd.total == 2 * n, "ScalarQDigest after merge: total doubled",
          f"{sqd.total}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Compression invariant verification
# ─────────────────────────────────────────────────────────────────────────────

def test_compression_invariant():
    """
    Verify that after compress(), every non-root node v satisfies:
        count(v) + count(parent(v)) + count(sibling(v)) > n / k
    (or count(v) == 0, which means the node isn't in the tree)
    """
    print("\n[5] Compression invariant check")
    rng = random.Random(13)
    U, k = 64, 10
    qd = QDigest(U, k)
    for _ in range(500):
        qd.insert(rng.randint(0, 63))
    qd.compress()

    threshold = qd.total / k
    violated = 0
    for (lo, hi), count in qd._tree.items():
        # Skip root
        if lo == 0 and hi == qd._U - 1:
            continue
        parent  = qd._parent(lo, hi)
        sibling = qd._sibling(lo, hi)
        p_count = qd._tree.get(parent, 0)
        s_count = qd._tree.get(sibling, 0)
        if count + p_count + s_count <= threshold:
            violated += 1

    check(violated == 0,
          f"invariant: no node violates count+parent+sibling > n/k",
          f"{violated} violations found")

    # Also: node counts are all positive (no zero-count nodes)
    zero_nodes = sum(1 for c in qd._tree.values() if c <= 0)
    check(zero_nodes == 0, "no zero-count nodes after compress()",
          f"{zero_nodes} found")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Accuracy bound: rank error ≤ n/k for each quantile query
# ─────────────────────────────────────────────────────────────────────────────

def test_accuracy_bound():
    """
    The Q-Digest paper guarantees rank error ≤ n/k per query.
    Check this empirically for several quantile levels.
    """
    print("\n[6] Accuracy bound (rank error ≤ n/k)")
    rng = random.Random(99)
    U, k = 128, 20
    n = 2000
    qd = QDigest(U, k)
    vals = sorted([rng.randint(0, 127) for _ in range(n)])
    for v in vals:
        qd.insert(v)
    qd.compress()

    allowed_err = n / k   # = 100 rank positions
    failures = 0
    for q in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
        estimated = qd.quantile(q)
        # True q-th quantile (exact, from sorted list)
        true_val = vals[int(q * n) - 1]
        # Rank of estimated value in sorted list
        rank_est = sum(1 for v in vals if v <= estimated)
        rank_err = abs(rank_est - q * n)
        if rank_err > allowed_err + 1:   # +1 for off-by-one in discrete case
            failures += 1
            print(f"    q={q:.2f}: true={true_val}, est={estimated:.1f}, "
                  f"rank_err={rank_err:.1f} > {allowed_err:.1f}  <- VIOLATION")

    check(failures == 0,
          f"all quantile rank errors ≤ n/k = {allowed_err:.0f}",
          f"{failures} violations")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Merge preserves total count and is commutative (Q50-wise)
# ─────────────────────────────────────────────────────────────────────────────

def test_merge_commutativity():
    print("\n[7] Merge commutativity")
    rng = random.Random(55)
    U, k = 64, 20

    def make_digest(seed, lo, hi, n=300):
        d = QDigest(U, k)
        r = random.Random(seed)
        for _ in range(n):
            d.insert(r.randint(lo, hi))
        d.compress()
        return d

    d1 = make_digest(1, 0, 30)
    d2 = make_digest(2, 20, 50)

    # D1 merge D2
    da = QDigest(U, k)
    for node, c in d1._tree.items():
        da._tree[node] = c
    da._n = d1._n
    da.merge(d2)

    # D2 merge D1
    db = QDigest(U, k)
    for node, c in d2._tree.items():
        db._tree[node] = c
    db._n = d2._n
    db.merge(d1)

    check(da.total == db.total, "merge commutativity: totals equal",
          f"{da.total} != {db.total}")

    # Q50 should be within 3 bins of each other
    diff = abs(da.quantile(0.5) - db.quantile(0.5))
    check(diff <= 3.0, "Q50 within 3 bins regardless of merge order",
          f"diff={diff:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Integration smoke-test: run wsn_main.main() end-to-end
# ─────────────────────────────────────────────────────────────────────────────

def test_integration():
    print("\n[8] Integration smoke-test (wsn_main.main)")
    try:
        import wsn_main
        wsn_main.main()
        # Verify output files were written
        results_ok = os.path.exists(
            os.path.join(HERE, "wsn_results.json"))
        anim_ok = os.path.exists(
            os.path.join(HERE, "wsn_animation_data.json"))
        check(results_ok, "wsn_results.json written")
        check(anim_ok,    "wsn_animation_data.json written")

        # Verify qdigest_filter_stats present in output
        import json
        with open(os.path.join(HERE, "wsn_results.json")) as f:
            data = json.load(f)
        has_qd = "qdigest_filter_stats" in data
        check(has_qd, "wsn_results.json contains qdigest_filter_stats")

        if has_qd:
            qd_stats = data["qdigest_filter_stats"]
            check(len(qd_stats) == 5, "qdigest_filter_stats has 5 CHs",
                  f"got {len(qd_stats)}")
            # At least one CH should have adaptive=True (>= QD_MIN_SAMPLES=30)
            adaptive_count = sum(1 for s in qd_stats if s.get("adaptive", False))
            check(adaptive_count > 0,
                  f"at least 1 CH in adaptive mode",
                  f"{adaptive_count} adaptive CHs")
            # Check quantile keys are present for adaptive CHs
            for s in qd_stats:
                if s.get("adaptive"):
                    has_keys = all(
                        k in s for k in
                        ["mag_max_q50", "mag_max_q90", "az_std_q50", "az_std_q90"])
                    check(has_keys, f"CH{s['n_samples']}: Q-stats keys present",
                          str(list(s.keys())))

        # Verify qdigest metadata in predictions (batch level)
        preds = data.get("predictions", [])
        check(len(preds) > 0, f"predictions list is non-empty ({len(preds)} entries)")

    except Exception:
        print("  Exception during integration test:")
        traceback.print_exc()
        _failures.append("integration smoke-test raised exception")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("Q-Digest Unit Tests")
    print("=" * 64)

    test_basic_uniform()
    test_merge()
    test_edge_cases()
    test_scalar_qdigest()
    test_compression_invariant()
    test_accuracy_bound()
    test_merge_commutativity()
    test_integration()

    print("\n" + "=" * 64)
    if _failures:
        print(f"\033[91mFAILED: {len(_failures)} test(s) failed:\033[0m")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"\033[92mAll tests passed.\033[0m")


if __name__ == "__main__":
    main()
