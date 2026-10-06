"""Optional external PDF/layout engine; input is authorized PDF bytes on stdin."""
import io
import json
import sys

import pdfplumber
from pypdf import PdfReader


def rectangle(box, width, height, rotation):
    def inverse(x, y):
        if rotation == 90:
            return y, height - x
        if rotation == 180:
            return width - x, height - y
        if rotation == 270:
            return width - y, x
        return x, y
    points = [inverse(x, y) for x in (box[0], box[2]) for y in (box[1], box[3])]
    return dict(left=min(p[0] for p in points), top=min(p[1] for p in points),
                right=max(p[0] for p in points), bottom=max(p[1] for p in points))


def cells_for(page, width, height, rotation, limit):
    result = []
    for index, table in enumerate(page.find_tables()):
        xs = sorted({x for cell in table.cells for x in (cell[0], cell[2])})
        ys = sorted({y for cell in table.cells for y in (cell[1], cell[3])})
        for number, cell in enumerate(table.cells):
            if len(result) >= limit:
                raise OverflowError("cell limit")
            # Rotation changes orientation of native table grids. A rotated table
            # requires an explicit grid normalization profile, not inferred rows.
            if rotation:
                raise NotImplementedError("rotated table grid")
            text = page.crop(cell).extract_text() or ""
            result.append(dict(table=f"t{index}", element=f"c{number}",
                               row=ys.index(cell[1]), column=xs.index(cell[0]),
                               row_span=ys.index(cell[3])-ys.index(cell[1]),
                               column_span=xs.index(cell[2])-xs.index(cell[0]),
                               text=text, region=rectangle(cell, width, height, rotation)))
    return result


def parse(data, page_limit, word_limit, cell_limit, image_limit):
    reader = PdfReader(io.BytesIO(data))
    labels = reader.page_labels
    result = dict(page_count=len(reader.pages), diagnostics=[], pages=[], error="")
    if len(reader.pages) > page_limit:
        result["diagnostics"].append(dict(code="page_limit", element=""))
    with pdfplumber.open(io.BytesIO(data)) as document:
        for index, page in enumerate(document.pages[:page_limit]):
            media = [float(value) for value in reader.pages[index].mediabox]
            crop = [float(value) for value in reader.pages[index].cropbox]
            rotation = int(page.rotation or 0)
            if media[:2] != [0., 0.] or crop != media or rotation not in (0, 90, 180, 270):
                raise NotImplementedError("native page geometry")
            width, height = media[2], media[3]
            words = page.extract_words()
            if len(words) > word_limit or len(page.images) > image_limit:
                raise OverflowError("element limit")
            text_parts, normalized_words, offset = [], [], 0
            for number, word in enumerate(words):
                text = word["text"]
                if not text:
                    continue
                if text_parts:
                    offset += 1
                start = offset
                offset += len(text.encode("utf-8"))
                text_parts.append(text)
                region = rectangle((word["x0"], word["top"], word["x1"], word["bottom"]), width, height, rotation)
                normalized_words.append(dict(id=f"w{number}", span=dict(start=start, end=offset), region=region))
            images, diagnostics = [], []
            for number, image in enumerate(page.images):
                images.append(dict(id=f"img{number}", region=rectangle(
                    (image["x0"], image["top"], image["x1"], image["bottom"]), width, height, rotation)))
                diagnostics.append(dict(code="ocr_unprocessed", element=f"img{number}"))
            result["pages"].append(dict(
                geometry=dict(physical_index=index, printed_label=labels[index],
                              width_pt=width, height_pt=height, rotation_deg=rotation),
                text=" ".join(text_parts), words=normalized_words,
                cells=cells_for(page, width, height, rotation, cell_limit), images=images,
                coverage="partial" if diagnostics else "complete", diagnostics=diagnostics))
    return result


try:
    output = parse(sys.stdin.buffer.read(), *(int(value) for value in sys.argv[1:]))
except NotImplementedError:
    output = dict(page_count=0, diagnostics=[], pages=[], error="unsupported_geometry")
except OverflowError:
    output = dict(page_count=0, diagnostics=[], pages=[], error="limit_exceeded")
except Exception:
    # Never echo parser exception text or source data into the result/error stream.
    output = dict(page_count=0, diagnostics=[], pages=[], error="invalid_pdf")
json.dump(output, sys.stdout, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
