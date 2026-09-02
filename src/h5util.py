"""Shared HDF5 write helper: transparent gzip for the large one-hot payloads.

The bulk of this project's on-disk footprint is a handful of datasets that store
a *hard* 0/1 field as uncompressed float64 -- the Phase 1 densities (``x_opt`` /
``x0``), the arm labellings, and Phase 2's ``indicator_functions``. Measured on a
real N=500 solution, gzip level 4 shrinks such a block **459x** (457 MB -> 1.0 MB)
and is *faster* end to end, because writing 457 MB of mostly-zeros costs more I/O
than compressing it costs CPU.

gzip is a standard HDF5 filter, so ``h5py`` decompresses transparently: no reader
anywhere in the project needs to change, and files written before this helper
existed keep opening exactly as before.

Small datasets are left uncompressed -- chunking overhead is not worth it below
roughly a megabyte, and scalars cannot be chunked at all.
"""

import numpy as np

#: Datasets at least this large (bytes, uncompressed) are gzip'd.
COMPRESS_MIN_BYTES = 1 << 20  # 1 MiB
#: gzip level. 4 captures essentially all of the available ratio on one-hot data
#: (459x vs 9's 470x on the calibration file) at a fraction of the CPU.
COMPRESS_LEVEL = 4


def create_dataset(group, name, data, **kwargs):
    """``group.create_dataset`` with gzip when the payload is big enough.

    Compression is only added when the caller has not asked for something
    specific, so an explicit ``compression=`` or ``chunks=`` always wins.
    """
    arr = np.asarray(data) if not isinstance(data, np.ndarray) else data
    big = arr.ndim > 0 and arr.nbytes >= COMPRESS_MIN_BYTES
    if big and "compression" not in kwargs:
        kwargs["compression"] = "gzip"
        kwargs.setdefault("compression_opts", COMPRESS_LEVEL)
        kwargs.setdefault("chunks", True)
    return group.create_dataset(name, data=arr, **kwargs)
