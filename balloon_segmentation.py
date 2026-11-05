"""Speech balloon (bubble) segmentation for manga / comic pages.

Model: bubble-pop (https://github.com/letmutex/bubble-pop), ONNX FP16,
Apache-2.0.  See models/model_fp16.onnx.

  input  image  [1, 1024, 768, 1] float16  grayscale, letterboxed, ImageNet-gray normalized
  output logits [1, 1024, 768, 3] float16  (mask, boundary, confidence)

Post-processing follows bubble-pop's inference.py: sigmoid, mask >= 0.5,
8-connectivity instances with area >= 60 and score (mask + confidence) / 2
>= MIN_SCORE.  Bubble interiors are hole-filled.

Text boxes can be passed in (no detector needed) to complete balloons the model
leaves partly unpainted: each box that the mask covers less than COVERAGE_MIN
gets the white area the text sits in added, if that area lies on a balloon.

Library use:

    from balloon_segmentation import BalloonSegmenter
    seg = BalloonSegmenter("models/model_fp16.onnx")
    mask, instances, prob = seg.segment(image_bgr)          # 0/255, page size
    # text boxes: [x0, y0, x1, y1] in page pixels
    mask = seg.complete_with_texts(image_bgr, mask, [[x0, y0, x1, y1], ...])

Command line:

    python balloon_segmentation.py <image_or_folder> <output_folder> [--texts texts.json] [--paint]

texts.json is {"<image file name>": [[x0, y0, x1, y1], ...], ...}.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

import bubble_cv_from_boxes as cvseg

HEIGHT, WIDTH = 1024, 768
NORM_MEAN, NORM_STD = 0.449, 0.226
MASK_THRESHOLD = 0.5
MIN_AREA = 60
MIN_SCORE = 0.65          # instances below this score (mask + confidence) / 2 are dropped
SHRINK = 3                # px the mask is eroded before painting, keeps the drawn outline
COVERAGE_MIN = 0.90       # a text box covered less than this gets completed


def fill_holes(mask):
    """Paint enclosed regions of a 0/255 mask (text strokes inside a bubble)."""
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = padded.copy()
    cv2.floodFill(ff, np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8), (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(ff)[1:-1, 1:-1])


def shrink(mask, px):
    """Erode so the black balloon outline survives the white paint."""
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.erode(mask, k)


def imread_u(path):
    """cv2.imread cannot open non-ASCII paths on Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_u(path, img, quality=92):
    ext = Path(path).suffix
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext.lower() in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    buf.tofile(str(path))


class BalloonSegmenter:
    def __init__(self, model_path=None, min_score=MIN_SCORE):
        if model_path is None:
            model_path = Path(__file__).resolve().parent / "models" / "model_fp16.onnx"
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_path), sess_options=so,
                                            providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.fp16 = "float16" in inp.type
        # this build is NHWC [1, H, W, 1]; accept NCHW too
        self.nhwc = len(inp.shape) == 4 and inp.shape[-1] == 1
        self.min_score = min_score

    def _float_type(self):
        return np.float16 if self.fp16 else np.float32

    def segment(self, bgr, min_score=None):
        """Return (mask, instances) with mask uint8 0/255 at page size.

        instances is a list of (region, score) where region is a bool array.
        """
        mask, _, conf = self.predict(bgr)
        found = self.instances(mask, conf, min_score)
        union = np.zeros(mask.shape, np.uint8)
        for region, _ in found:
            union[region] = 255
        return union, found, mask

    def predict(self, bgr):
        """Return (mask, boundary, confidence) probabilities at page size."""
        h, w = bgr.shape[:2]
        scale = min(WIDTH / w, HEIGHT / h)
        rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
        px, py = (WIDTH - rw) // 2, (HEIGHT - rh) // 2

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(gray, (rw, rh), interpolation=interp)
        canvas = np.full((HEIGHT, WIDTH), 255, np.uint8)
        canvas[py:py + rh, px:px + rw] = resized

        x = (canvas.astype(np.float32) / 255.0 - NORM_MEAN) / NORM_STD
        x = x[None, :, :, None] if self.nhwc else x[None, None]
        logits = self.session.run(None, {self.input_name: x.astype(self._float_type())})[0]
        logits = logits.astype(np.float32)[0]
        if not self.nhwc:
            logits = np.transpose(logits, (1, 2, 0))

        probs = 1.0 / (1.0 + np.exp(-logits))
        planes = [cv2.resize(probs[py:py + rh, px:px + rw, c], (w, h),
                             interpolation=cv2.INTER_LINEAR) for c in range(3)]
        return tuple(planes)

    def complete_with_texts(self, bgr, mask, text_boxes, coverage_min=COVERAGE_MIN):
        """Add the white areas of balloons the mask under-covers.

        For every text box (x0, y0, x1, y1) whose area the mask covers less than
        `coverage_min`, take the white region the text sits in and add it when
        it touches the mask (one balloon split into pieces) and stays of balloon
        size.  Returns (mask, report) with report = [(box, coverage, status)].
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        out = mask.copy()
        touch = cv2.dilate(out, np.ones((9, 9), np.uint8)) > 0
        report = []
        for box in text_boxes:
            x0, y0, x1, y1 = [int(v) for v in box]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w - 1, x1), min(h - 1, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            cov = (out[y0:y1 + 1, x0:x1 + 1] > 0).mean()
            if cov >= coverage_min:
                report.append(((x0, y0, x1, y1), round(float(cov), 2), "covered"))
                continue
            region = cvseg.white_region(gray, (x0, y0, x1, y1))
            if region is None:
                report.append(((x0, y0, x1, y1), round(float(cov), 2), "no-region"))
                continue
            if not (region.astype(bool) & touch).any():
                report.append(((x0, y0, x1, y1), round(float(cov), 2), "not-on-balloon"))
                continue
            # the white region excludes the glyph ink; take the text box too
            region[y0:y1 + 1, x0:x1 + 1] = 255
            out = cv2.bitwise_or(out, region)
            report.append(((x0, y0, x1, y1), round(float(cov), 2), "completed"))
        return out, report

    def instances(self, mask, confidence, min_score=None):
        min_score = self.min_score if min_score is None else min_score
        binary = (mask >= MASK_THRESHOLD).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        result = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
                continue
            region = labels == i
            score = float((mask[region].mean() + confidence[region].mean()) / 2.0)
            if score < min_score:
                continue
            result.append((region, score))
        return result


def mask_for_painting(bgr, seg, shrink_px=SHRINK, fill=True, text_boxes=None):
    """Mask ready for painting text out: completed, hole-filled and shrunk."""
    mask, _, _ = seg.segment(bgr)
    if text_boxes:
        mask, _ = seg.complete_with_texts(bgr, mask, text_boxes)
    if fill:
        mask = fill_holes(mask)
    return shrink(mask, shrink_px)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", help="image file or folder")
    ap.add_argument("output", help="output folder")
    ap.add_argument("--model", default=None, help="ONNX model path (default: models/model_fp16.onnx)")
    ap.add_argument("--min-score", type=float, default=MIN_SCORE)
    ap.add_argument("--shrink", type=int, default=SHRINK)
    ap.add_argument("--no-fill", action="store_true", help="do not fill holes inside balloons")
    ap.add_argument("--texts", default=None,
                    help='JSON with text boxes to complete balloons: {"<image name>": [[x0,y0,x1,y1], ...]}')
    ap.add_argument("--paint", action="store_true", help="also save the page with bubbles painted white")
    args = ap.parse_args()

    texts = {}
    if args.texts:
        with open(args.texts, encoding="utf-8") as f:
            texts = json.load(f)

    src, out_dir = Path(args.images), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [src] if src.is_file() else sorted(
        p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    seg = BalloonSegmenter(args.model, min_score=args.min_score)

    for p in files:
        bgr = imread_u(p)
        if bgr is None:
            print(f"{p.name}: cannot read, skipped")
            continue
        raw, found, _ = seg.segment(bgr)
        report = []
        if p.name in texts:
            raw, report = seg.complete_with_texts(bgr, raw, texts[p.name])
        else:
            raw, report = seg.complete_with_texts(bgr, raw, texts.get(p.stem, []))
        final = shrink(fill_holes(raw) if not args.no_fill else raw, args.shrink)
        cv2.imencode(".png", final)[1].tofile(str(out_dir / f"{p.stem}_mask.png"))

        overlay = bgr.copy()
        colors = bgr.copy()
        for k, (region, score) in enumerate(found):
            color = tuple(int(c) for c in np.random.default_rng(k + 1).integers(64, 240, 3))
            colors[region] = color
            contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2, cv2.LINE_AA)
            x, y, _, _ = cv2.boundingRect(region.astype(np.uint8))
            cv2.putText(overlay, f"{score:.2f}", (x, max(16, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        imwrite_u(out_dir / f"{p.stem}_overlay.jpg",
                  cv2.addWeighted(overlay, 0.7, colors, 0.3, 0))
        if args.paint:
            painted = bgr.copy()
            painted[final > 0] = 255
            imwrite_u(out_dir / f"{p.stem}_painted.jpg", painted)
        for box, cov, status in report:
            if status != "covered":
                print(f"    text {box} coverage {cov:.2f}: {status}")
        print(f"{p.name}: {len(found)} balloons, mask {int((final > 0).sum())} px")


if __name__ == "__main__":
    main()
