from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


class ImageFormatError(ValueError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_grayscale(path: Path) -> np.ndarray:
    """Load a grayscale image without silently converting RGB or rescaling."""
    try:
        with Image.open(path) as image:
            image.load()
            if image.mode in {"L", "P"}:
                if image.mode == "P":
                    raise ImageFormatError("调色板图像不视为原生单通道灰度图")
                array = np.asarray(image, dtype=np.uint8)
            elif image.mode in {"I;16", "I;16L", "I;16B"}:
                array = np.asarray(image)
                if array.dtype.byteorder == ">":
                    array = array.byteswap().view(array.dtype.newbyteorder("="))
                array = array.astype(np.uint16, copy=False)
            elif image.mode == "I":
                raw = np.asarray(image)
                if raw.min(initial=0) < 0 or raw.max(initial=0) > 65535:
                    raise ImageFormatError("32 位整数图像的像素超出 uint16 范围")
                array = raw.astype(np.uint16)
            else:
                raise ImageFormatError(f"图像模式 {image.mode!r} 不是受支持的单通道灰度格式")
    except ImageFormatError:
        raise
    except Exception as exc:
        raise ImageFormatError(f"无法读取图像：{exc}") from exc

    if array.ndim != 2:
        raise ImageFormatError("图像不是二维单通道数据")
    if array.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise ImageFormatError(f"不支持的数据类型：{array.dtype}")
    return np.ascontiguousarray(array)


def image_metadata(path: Path) -> dict[str, object]:
    array = load_grayscale(path)
    height, width = array.shape
    return {
        "sha256": file_sha256(path),
        "width": int(width),
        "height": int(height),
        "dtype": str(array.dtype),
        "channels": 1,
    }


def dtype_range(dtype: str) -> tuple[int, int]:
    if dtype == "uint8":
        return 0, 255
    if dtype == "uint16":
        return 0, 65535
    raise ImageFormatError(f"不支持的数据类型：{dtype}")

