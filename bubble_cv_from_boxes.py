"""Plain-OpenCV helpers: ink-outline holes and white regions around a text box.

Used to complete speech balloons the segmentation model leaves partly unpainted.
A balloon outline is a closed dark loop, so its inner contour is the balloon
interior; the hole search runs at several ink-closing sizes because an outline
with hairline gaps only closes at a larger kernel.
"""

import cv2
import numpy as np

GAP = 3          # px the outline is thickened towards the text
RING = 4         # ring distance outside the text box where the fill starts
MAX_FRAC = 0.08  # a region larger than this share of the page is the page ground


def find_bubble_holes(ink, min_area=500):
    """Hole contours of the ink layer, i.e. candidate balloon interiors."""
    contours, hier = cv2.findContours(ink, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    holes = []
    for ci, hi in enumerate(hier[0]):
        if hi[3] == -1:                     # -1 = outer boundary, not a hole
            continue
        area = cv2.contourArea(contours[ci])
        if area >= min_area:
            holes.append((ci, area, cv2.boundingRect(contours[ci])))
    return contours, holes


def build_hole_levels(ink, close_sizes=(5, 9, 13, 17), min_area=500):
    """findContours at several ink-closing levels."""
    levels = []
    for k in close_sizes:
        ik = ink
        if k > 1:
            ik = cv2.morphologyEx(
                ink, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        contours, holes = find_bubble_holes(ik, min_area)
        levels.append((k, contours, holes))
    return levels


def smooth_shrink(mask, shrink):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if shrink > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * shrink + 1, 2 * shrink + 1))
        mask = cv2.erode(mask, k)
    return mask


def _hole_shares_lines(part, box):
    """The hole must cover the middle row of the text box, not just wrap it.

    Otherwise the hole is the surrounding ground (e.g. a ring between a
    scalloped edge and a drawn border).
    """
    x0, y0, x1, y1 = box
    if y1 <= y0 or x1 <= x0:
        return False
    mid = part[(y0 + y1) // 2, x0:x1 + 1]
    return cv2.countNonZero(mid) >= 0.6 * len(mid)


def hole_for_box(gray, ink, levels, box, shrink=0):
    """Smoothed ink hole that wraps a text box, or None.

    The hole must wrap the box and cross its middle row, and most of its
    perimeter must lie on drawn ink (a hole bordered by the panel background
    does not count).  The smallest such hole wins.
    """
    h, w = gray.shape
    x0, y0, x1, y1 = box
    pad = 10
    best = None
    for k_close, contours, holes in levels:
        for ci, area, (hx, hy, hw, hh) in holes:
            if not (hx <= x0 + pad and hy <= y0 + pad
                    and hx + hw >= x1 - pad and hy + hh >= y1 - pad):
                continue
            if hw * hh > 0.25 * h * w:
                continue
            part = np.zeros((h, w), np.uint8)
            cv2.drawContours(part, contours, ci, 255, cv2.FILLED)
            if not _hole_shares_lines(part, box):
                continue
            k = max(3, k_close if k_close > 0 else 5) | 1
            ring = cv2.morphologyEx(part, cv2.MORPH_GRADIENT,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            seg = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            if cv2.countNonZero(cv2.bitwise_and(ring, seg)) < 0.6 * cv2.countNonZero(ring):
                continue
            n = cv2.countNonZero(part)
            if best is None or n < best[0]:
                best = (n, part)
    if best is None:
        return None
    part = smooth_shrink(best[1], shrink)
    return part if cv2.mean(gray, part)[0] >= 150 else None


def _side_ring(box, h, w, ring=RING):
    """Box grown by `ring`, minus the box and its top/bottom strips."""
    x0, y0, x1, y1 = box
    rx0, ry0 = max(0, x0 - ring), max(0, y0 - ring)
    rx1, ry1 = min(w, x1 + ring + 1), min(h, y1 + ring + 1)
    m = np.zeros((h, w), np.uint8)
    m[ry0:ry1, rx0:rx1] = 255
    m[y0:y1 + 1, x0:x1 + 1] = 0
    m[:, rx0:min(w, x0 + 1)] = 0
    m[:, max(0, x1):rx1] = 0
    return m


def _line_region(box, h, w, margin=80):
    """Area a single text's balloon may occupy: the box plus its surroundings."""
    x0, y0, x1, y1 = box
    m = np.zeros((h, w), np.uint8)
    m[max(0, y0 - margin):min(h, y1 + margin + 1),
      max(0, x0 - margin):min(w, x1 + margin + 1)] = 255
    return m


def white_region(gray, box, gap=GAP):
    """White area a single text box sits in: flood fill around the box.

    Glyph ink inside the box is not a wall, everything else drawn is; the fill
    starts from the box sides and the row above it, and is kept only if it stays
    of balloon size.  Returns 0/255 or None.
    """
    h, w = gray.shape
    x0, y0, x1, y1 = box
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text_ink = np.zeros((h, w), np.uint8)
    text_ink[max(0, y0 - 1):min(h, y1 + 2), max(0, x0 - 1):min(w, x1 + 2)] = 255
    wall = cv2.bitwise_and(ink, cv2.bitwise_not(text_ink))
    for g in (gap, 1):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))
        reach = cv2.bitwise_and(cv2.bitwise_not(cv2.dilate(wall, k)),
                                (gray > 128).astype(np.uint8) * 255)
        probe = _side_ring(box, h, w)
        probe[max(0, y0 - RING):max(0, y0), x0:x1 + 1] = 255
        n, labels = cv2.connectedComponents(reach, connectivity=4)
        picked = np.unique(labels[(probe > 0) & (labels > 0)])
        picked = picked[picked > 0]
        if len(picked) == 0:
            continue
        sizes = np.bincount(labels.ravel())
        if sizes[picked].max() > MAX_FRAC * h * w:
            continue
        m = np.isin(labels, picked).astype(np.uint8) * 255
        return cv2.bitwise_and(m, _line_region(box, h, w))
    return None
