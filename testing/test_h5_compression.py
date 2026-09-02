#!/usr/bin/env python3
"""Gates for the gzip write path (src/h5util.py).

Three things must hold, and only the first is obvious:
  1. round trip -- a compressed dataset reads back bit-identical;
  2. backward compatibility -- files written BEFORE compression still open, and
     the readers cannot tell the difference;
  3. the arm-solution writer is unchanged in CONTENT, so a solution written now
     yields the same labels, the same argmax and the same gates as one written
     by the old path.
"""
import os, sys, tempfile, numpy as np, h5py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.h5util import create_dataset as h5_create, COMPRESS_MIN_BYTES

FAILED = []
def check(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond: FAILED.append(msg)

def main():
    tmp = tempfile.mkdtemp()
    V, N = 40000, 40                      # ~12.8 MB one-hot -> above the threshold
    rng = np.random.default_rng(0)
    labels = rng.integers(0, N, V)
    dens = np.zeros((V, N)); dens[np.arange(V), labels] = 1.0

    print("\n1/5  round trip: compressed data reads back bit-identical")
    f1 = os.path.join(tmp, "c.h5")
    with h5py.File(f1, "w") as f:
        h5_create(f, "x_opt", dens.ravel())
        h5_create(f, "labels", labels.astype(np.int32))
    with h5py.File(f1) as f:
        check(np.array_equal(f["x_opt"][:], dens.ravel()), "x_opt identical after round trip")
        check(np.array_equal(f["labels"][:], labels.astype(np.int32)), "labels identical after round trip")
        check(f["x_opt"].compression == "gzip", "large dataset IS gzip'd")

    print("\n2/5  small datasets are left alone (chunking overhead not worth it)")
    f2 = os.path.join(tmp, "s.h5")
    with h5py.File(f2, "w") as f:
        h5_create(f, "small", np.arange(10.0))
        h5_create(f, "scalar", np.float64(3.0))
    with h5py.File(f2) as f:
        check(f["small"].compression is None, "small array not compressed")
        check(f["scalar"].compression is None, "scalar not compressed (cannot be chunked)")
        check(float(f["scalar"][()]) == 3.0, "scalar value survives")

    print("\n3/5  compressed vs uncompressed are INDISTINGUISHABLE to a reader")
    f3 = os.path.join(tmp, "u.h5")
    with h5py.File(f3, "w") as f:                 # the OLD write path
        f.create_dataset("x_opt", data=dens.ravel())
    with h5py.File(f1) as a, h5py.File(f3) as b:
        check(np.array_equal(a["x_opt"][:], b["x_opt"][:]), "old and new files yield identical arrays")
        check(a["x_opt"].shape == b["x_opt"].shape and a["x_opt"].dtype == b["x_opt"].dtype,
              "shape and dtype preserved")
        ra = a["x_opt"][:].reshape(V, N).argmax(1)
        rb = b["x_opt"][:].reshape(V, N).argmax(1)
        check(np.array_equal(ra, rb) and np.array_equal(ra, labels), "argmax readout identical and correct")

    print("\n4/5  an explicit caller choice still wins")
    f4 = os.path.join(tmp, "e.h5")
    with h5py.File(f4, "w") as f:
        h5_create(f, "x", dens.ravel(), compression=None)
    with h5py.File(f4) as f:
        check(f["x"].compression is None, "explicit compression=None respected")

    print("\n5/5  it actually saves space, and the threshold is honoured")
    sz_c, sz_u = os.path.getsize(f1), os.path.getsize(f3)
    check(sz_c * 10 < sz_u, f"compressed is >10x smaller ({sz_u/1e6:.1f} MB -> {sz_c/1e6:.2f} MB, "
                            f"{sz_u/sz_c:.0f}x)")
    check(dens.nbytes >= COMPRESS_MIN_BYTES, "test payload is above the compression threshold")

    print("\n" + "="*66)
    print("RESULT: " + ("PASS" if not FAILED else f"FAIL ({len(FAILED)})"))
    return 1 if FAILED else 0

if __name__ == "__main__":
    sys.exit(main())
