"""Loading PDFs / images into BGR numpy arrays, page by page."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageSequence

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_EXTS = {".pdf"}


@dataclass
class PageImage:
    source: Path
    page_index: int            # 0-based
    image: np.ndarray          # BGR uint8
    dpi: Optional[int]         # set for PDF renders
    is_pdf: bool


def _pil_to_bgr(im: Image.Image) -> np.ndarray:
    im = im.convert("RGB")
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def iter_input_files(paths: list[str | Path]) -> list[Path]:
    """Expand files and directories (non-recursive) into a sorted list of inputs."""
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS | PDF_EXTS)
        elif p.exists():
            out.append(p)
        else:
            raise FileNotFoundError(p)
    return out


def load_pages(path: Path, dpi: int = 200, draw_annotations: bool = False) -> Iterator[PageImage]:
    ext = path.suffix.lower()
    if ext in PDF_EXTS:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(path))
        try:
            for i in range(len(pdf)):
                page = pdf[i]
                bmp = page.render(scale=dpi / 72.0, draw_annots=draw_annotations)
                yield PageImage(path, i, _pil_to_bgr(bmp.to_pil()), dpi, True)
                page.close()
        finally:
            pdf.close()
    elif ext in IMAGE_EXTS:
        with Image.open(path) as im:
            # Only TIFFs are true multi-page images. Phone JPEGs (MPO/HEIC-derived) can carry
            # hidden extra frames (depth/gain maps) that must not be audited as pages.
            frames = ImageSequence.Iterator(im) if ext in (".tif", ".tiff") else [im]
            for i, frame in enumerate(frames):
                frame = ImageOps.exif_transpose(frame.copy())        # phone photos
                yield PageImage(path, i, _pil_to_bgr(frame), None, False)
    else:
        raise ValueError(f"Unsupported file type: {path}")


def load_image_file(path: Path, dpi: int = 200) -> np.ndarray:
    """First page of a PDF or an image, as BGR (used for blank templates)."""
    return next(load_pages(Path(path), dpi=dpi, draw_annotations=False)).image
