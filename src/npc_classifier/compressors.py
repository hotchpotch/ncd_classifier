from __future__ import annotations
import zlib
import bz2
import lzma
import gzip
import numpy as np
from typing import Callable, Sequence, Union


def compression_length(
    compression_function: Callable[[bytes], bytes]
) -> Callable[[Union[str, Sequence[int]]], int]:
    """
    Returns a function that uses the provided compression function to compress a string or a list of integers
    and returns the length of the compressed data.

    Args:
        compression_function (Callable[[bytes], bytes]): Function to use for compression.

    Returns:
        Callable[[Union[str, list[int]]], int]: Function that takes a string or a list of integers and returns the length of the compressed data.
    """

    def inner(data: Union[str, Sequence[int]]) -> int:
        if isinstance(data, str):
            compressed_data = compression_function(data.encode("utf-8"))
        elif isinstance(data, np.ndarray):
            # np.ndarray だったら
            compressed_data = compression_function(data.tobytes())
        elif isinstance(data, list):
            compressed_data = compression_function(np.array(data).tobytes())

        return len(compressed_data)  # type: ignore

    return inner


gzip_compression = compression_length(gzip.compress)
zlib_compression = compression_length(zlib.compress)
bz2_compression = compression_length(bz2.compress)
lzma_compression = compression_length(lzma.compress)

COMPRESSORS = {
    "gzip": gzip_compression,
    "zlib": zlib_compression,
    "bz2": bz2_compression,
    "lzma": lzma_compression,
}
