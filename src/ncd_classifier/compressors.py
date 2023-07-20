from __future__ import annotations
import zlib
import bz2
import lzma
import gzip
import numpy as np
from typing import Callable, Sequence, Union


def compress_data(
    data: Union[str, Sequence[int]], compression_function: Callable[[bytes], bytes]
) -> int:
    """
    Uses the provided compression function to compress a string or a list of integers
    and returns the length of the compressed data.

    Args:
        data (Union[str, list[int]]): Data to compress.
        compression_function (Callable[[bytes], bytes]): Function to use for compression.

    Returns:
        int: Length of the compressed data.
    """
    if isinstance(data, str):
        compressed_data = compression_function(data.encode("utf-8"))
    elif isinstance(data, np.ndarray):
        compressed_data = compression_function(data.tobytes())
    elif isinstance(data, list):
        compressed_data = compression_function(np.array(data).tobytes())

    return len(compressed_data)  # type: ignore


def gzip_compression(data):
    return compress_data(data, gzip.compress)


def zlib_compression(data):
    return compress_data(data, zlib.compress)


def bz2_compression(data):
    return compress_data(data, bz2.compress)


def lzma_compression(data):
    return compress_data(data, lzma.compress)


COMPRESSORS = {
    "gzip": gzip_compression,
    "zlib": zlib_compression,
    "bz2": bz2_compression,
    "lzma": lzma_compression,
}
