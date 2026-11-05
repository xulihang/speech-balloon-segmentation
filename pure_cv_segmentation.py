"""Pure image-processing balloon segmentation: dilate to close, then region growing.

No model.  The caller supplies the text boxes ([x0, y0, x1, y1], page pixels,
one per text line or block):

  1. Dilate the ink: a balloon outline with hairline gaps becomes a closed
     ring, so its interior is enclosed.  The gaps are a few px, so the dilation
     radius only has to be a few px too.
  2. Region growing: the balloon interior is the white connected component that
     the text box sits in after the dilation (two thicknesses are tried; the
     thinnest one that keeps the text enclosed wins).
  3. The grown region contains the dilated outline; erode it back by the
     dilation radius so the drawn line stays outside the mask.
  4. Boxes with no enclosing outline fall back to the white region around the
     box (flood fill at the same dilation, without eroding).

Usage:
    python pure_cv_segmentation.py <image_or_folder> <output_folder> --texts texts.json --paint
    texts.json: {"<image name>": [[x0, y0, x1, y1], ...], ...}
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

SHRINK = 2              # extra erosion after the outline is removed


def imread_u(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(path, img, quality=92):
    ext = Path(path).suffix
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext.lower() in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    buf.tofile(str(path))


def _ink(gray):
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return ink


def _kernel(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _box_majority(labels, box, lines=8, box_option=False):
    """Label most of the text box lies in, and its share of the box.

    Sampled on horizontal scan lines of the box only, so the shape of the
    balloon the text is in (not the box) draws the boundary between labels.
    """
    x0, y0, x1, y1 = box
    vals = []
    for k in range(lines):
        y = y0 + (k + 0.5) * (y1 - y0) / lines
        vals.append(labels[int(y), x0:x1 + 1])
    vals = np.concatenate(vals)
    vals = vals[vals > 0]
    if len(vals) == 0:
        return 0, 0.0
    counts = np.bincount(vals)
    lab = int(np.argmax(counts))
    fraction = float(counts[lab] / len(vals))
    if box_option:
        # option 2: the label that wins the share of the whole box footprint
        inside = labels[box[1]:box[3] + 1, box[0]:box[2] + 1]
        inside = inside[inside > 0]
        if len(inside):
            counts_box = np.bincount(inside)
            lab_box = int(np.argmax(counts_box))
            if lab_box != lab:
                return lab, fraction, lab_box
    return lab, fraction


def _encloses(outer, inner):
    """Every pixel of `inner` lies inside `outer`."""
    return cv2.countNonZero(cv2.bitwise_and(inner, cv2.bitwise_not(outer))) == 0


def _all_glyph_mask(ink, boxes, pad=12):
    """Ink inside every text box (a little wider): glyphs, not outlines."""
    h, w = ink.shape
    m = np.zeros((h, w), np.uint8)
    for x0, y0, x1, y1 in boxes:
        m[max(0, y0 - pad):min(h, y1 + pad + 1), max(0, x0 - pad):min(w, x1 + pad + 1)] = 255
    return cv2.bitwise_and(ink, m)


def _touches_wide(mask, erode_size, min_tiles):
    """Does the region reach the window border through a wide connection?"""
    h, w = mask.shape
    eroded = cv2.erode(mask, np.ones((erode_size, erode_size), np.uint8))
    tiles = min_tiles
    edges = [eroded[:tiles, :], eroded[-tiles:, :], eroded[:, :tiles], eroded[:, -tiles:]]
    return any(cv2.countNonZero(e) > 0 for e in edges)


def _find_outline(gray, ink, glyph_ink, box):
    """Outline contour around one text box, searched in a local window.

    The ink is closed at a few sizes so hairline gaps in the outline join up
    (the glyphs of every text box are removed first, so a ")" next to a line
    cannot pose as a tiny outline).  The contour must wrap the box, be
    balloon-sized, mostly filled, and not reach past the text by more than its
    own size.  Returns (contour_mask, area) or None.
    """
    h, w = gray.shape
    x0, y0, x1, y1 = box
    text_w, text_h = x1 - x0 + 1, y1 - y0 + 1
    win = int(1.6 * max(text_w, text_h)) + 30
    wx0, wy0 = max(0, x0 - win), max(0, y0 - win)
    wx1, wy1 = min(w, x1 + win + 1), min(h, y1 + win + 1)
    sub = ink[wy0:wy1, wx0:wx1]
    sub_ink = cv2.bitwise_and(sub, cv2.bitwise_not(glyph_ink[wy0:wy1, wx0:wx1]))
    bx0, by0, bx1, by1 = x0 - wx0, y0 - wy0, x1 - wx0, y1 - wy0
    max_area = 0.12 * h * w
    best = None
    for close in (5, 9, 13, 17):
        ik = cv2.morphologyEx(sub_ink, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
        contours, hier = cv2.findContours(ik, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            continue
        hier = hier[0]
        for ci, c in enumerate(contours):
            if hier[ci][3] != -1 and hier[ci][2] != -1:
                continue                       # outermost contours only
            hx, hy, hw, hh = cv2.boundingRect(c)
            if not (hx <= bx0 and hy <= by0 and hx + hw >= bx1 and hy + hh >= by1):
                continue
            box_area = hw * hh
            if box_area < 1500 or box_area > max_area:
                continue
            if hw > 3 * text_w + 120 or hh > 3 * text_h + 120:
                continue                       # page ground around the balloon
            m = np.zeros((wy1 - wy0, wx1 - wx0), np.uint8)
            cv2.drawContours(m, [c], -1, 255, cv2.FILLED)
            # a balloon interior is enclosed by its outline: it must not have
            # a wide white channel to the window edge (the page background)
            if _touches_wide(m, 2 * close + 1, 3):
                continue
            fill = cv2.countNonZero(m) / box_area
            hull_area = cv2.contourArea(cv2.convexHull(c))
            solidity = cv2.countNonZero(m) / hull_area if hull_area > 0 else 0
            if fill < 0.6 or solidity < 0.86:
                continue                       # not a closed balloon-shaped area
            if best is None or box_area < best[1]:
                full = np.zeros((h, w), np.uint8)
                full[wy0:wy1, wx0:wx1] = m
                best = (full, box_area)
    return best


def _page_ground(gray):
    """White components connected to the page border, through any brightness."""
    h, w = gray.shape
    free = (gray > 150).astype(np.uint8) * 255
    n, labels = cv2.connectedComponents(free, connectivity=4)
    border = np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
    ground = np.isin(labels, border[border > 0]).astype(np.uint8) * 255
    return ground


def _compact_white(gray, ink, box, ground=None):
    """Fallback without an outline: the white blob the text sits in, limited
    to the text surroundings (keeps it from flowing into the page ground).
    Returns a full-page mask or None."""
    h, w = gray.shape
    x0, y0, x1, y1 = box
    text_w, text_h = x1 - x0 + 1, y1 - y0 + 1
    win = int(1.4 * max(text_w, text_h)) + 20
    wx0, wy0 = max(0, x0 - win), max(0, y0 - win)
    wx1, wy1 = min(w, x1 + win + 1), min(h, y1 + win + 1)
    sub_ink = ink[wy0:wy1, wx0:wx1]
    sub_g = gray[wy0:wy1, wx0:wx1]
    bx0, by0, bx1, by1 = x0 - wx0, y0 - wy0, x1 - wx0, y1 - wy0
    text_ink = np.zeros_like(sub_ink)
    text_ink[by0:by1 + 1, bx0:bx1 + 1] = 255
    for g in (3, 1):
        wall = cv2.bitwise_and(sub_ink, cv2.bitwise_not(cv2.bitwise_and(sub_ink, text_ink)))
        free = cv2.bitwise_and(cv2.bitwise_not(cv2.dilate(wall, _kernel(g))),
                               (sub_g > 128).astype(np.uint8) * 255)
        n, labels = cv2.connectedComponents(free, connectivity=4)
        if n <= 1:
            continue
        mid = (by0 + by1) // 2
        vals = labels[mid, bx0:bx1 + 1]
        vals = vals[vals > 0]
        if len(vals) == 0:
            continue
        lab = int(np.argmax(np.bincount(vals)))
        bubble = (labels == lab).astype(np.uint8) * 255
        # limit to a balloon-sized band around the text
        band = int(0.5 * max(text_w, text_h)) + 15
        band_mask = np.zeros_like(bubble)
        band_mask[max(0, by0 - band):by1 + band + 1, max(0, bx0 - band):bx1 + band + 1] = 255
        bubble = cv2.bitwise_and(bubble, band_mask)
        # grow the text area step by step and stop where the fill starts to
        # widen without limit (a corridor into the page ground): accept the
        # last step whose region is still compact
        seed = np.zeros_like(bubble)
        seed[by0:by1 + 1, bx0:bx1 + 1] = 255
        current = cv2.bitwise_and(bubble, seed)
        if cv2.countNonZero(current) == 0:
            continue
        grown = current.copy()
        limit = int(1.2 * max(text_w, text_h))
        for _ in range(limit):
            grown = cv2.bitwise_or(grown, cv2.bitwise_and(cv2.dilate(grown, _kernel(2)), bubble))
            if cv2.countNonZero(grown) >= cv2.countNonZero(bubble):
                break
            bx, by, bw, bh = cv2.boundingRect(grown)
            if max(bw, bh) > 1.8 * max(text_w, text_h) + 10:
                break
        region = grown
        # limit to a balloon-sized band around the text
        band = int(0.5 * max(text_w, text_h)) + 15
        band_mask = np.zeros_like(region)
        band_mask[max(0, by0 - band):by1 + band + 1, max(0, bx0 - band):bx1 + band + 1] = 255
        region = cv2.bitwise_and(region, band_mask)
        full = np.zeros((h, w), np.uint8)
        full[wy0:wy1, wx0:wx1] = region
        return full
    return None


def segment_with_texts(bgr, text_boxes, shrink=SHRINK):
    """Balloon mask for the given text boxes.

    Contour detection first: outline contours enclosing the text in the ink
    closed at several sizes.  Where no outline encloses the text, region
    growing falls back to the white component the text sits in after dilating
    the ink.  Returns (mask, info): info = [(box, status, area)] with status
    "outline", "grow" or "none".
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    ink = _ink(gray)
    # glyph ink inside the text boxes must not split the balloon interior
    text_ink = np.zeros((h, w), np.uint8)
    for t in text_boxes:
        x0, y0, x1, y1 = [int(v) for v in t]
        text_ink[max(0, y0 - 1):min(h, y1 + 2), max(0, x0 - 1):min(w, x1 + 2)] = 255
    ink = cv2.bitwise_and(ink, cv2.bitwise_not(text_ink))

    boxes_int = []
    for t in text_boxes:
        x0, y0, x1, y1 = [int(v) for v in t]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w - 1, x1), min(h - 1, y1)
        if x1 > x0 and y1 > y0:
            boxes_int.append((x0, y0, x1, y1))
    glyph_ink = _all_glyph_mask(ink, boxes_int)
    ground = _page_ground(gray)

    union = np.zeros((h, w), np.uint8)
    info = []
    for box in boxes_int:
        found = _find_outline(gray, ink, glyph_ink, box)
        status = "outline"
        if found is not None:
            region = found[0]
        else:
            region = _compact_white(gray, ink, box, ground)
            status = "white"
        if region is None or cv2.countNonZero(region) < 800:
            info.append((box, "none", 0))
            continue
        if shrink > 0:
            region = cv2.erode(region, _kernel(shrink))
        union = cv2.bitwise_or(union, region)
        info.append((box, status, int(cv2.countNonZero(region))))
    union = fill_holes(union)
    return union, info


def fill_holes(mask):
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = padded.copy()
    cv2.floodFill(ff, np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8), (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(ff)[1:-1, 1:-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images")
    ap.add_argument("output")
    ap.add_argument("--texts", required=True)
    ap.add_argument("--shrink", type=int, default=SHRINK)
    ap.add_argument("--paint", action="store_true")
    args = ap.parse_args()

    texts = json.load(open(args.texts, encoding="utf-8"))
    src, out_dir = Path(args.images), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [src] if src.is_file() else sorted(
        p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))

    for p in files:
        bgr = imread_u(p)
        if bgr is None:
            print(f"{p.name}: cannot read, skipped")
            continue
        boxes = texts.get(p.name, texts.get(p.stem, []))
        mask, info = segment_with_texts(bgr, boxes, shrink=args.shrink)
        stem = p.stem
        cv2.imencode(".png", mask)[1].tofile(str(out_dir / f"{stem}_mask.png"))
        vis = bgr.copy()
        over = vis.copy()
        over[mask > 0] = (0, 200, 0)
        vis = cv2.addWeighted(over, 0.6, vis, 0.4, 0)
        for box, status, _ in info:
            color = {"outline": (0, 0, 255), "white": (0, 165, 255), "none": (255, 0, 255)}[status]
            cv2.rectangle(vis, box[:2], box[2:], color, 1)
        imwrite_u(out_dir / f"{stem}_overlay.jpg", vis)
        if args.paint:
            painted = bgr.copy()
            painted[mask > 0] = 255
            imwrite_u(out_dir / f"{stem}_painted.jpg", painted)
        counts = {}
        for _, status, _ in info:
            counts[status] = counts.get(status, 0) + 1
        print(f"{p.name}: {counts}, mask {int((mask > 0).sum())} px")


if __name__ == "__main__":
    main()
