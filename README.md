# speech-balloon-segmentation

Speech balloon (bubble) segmentation for manga and comic pages.

- Model: [bubble-pop](https://github.com/letmutex/bubble-pop) (MobileNetV3-Small UNet,
  0.93M parameters), ONNX FP16 build in [models/model_fp16.onnx](models/model_fp16.onnx),
  licensed under Apache-2.0 (see [LICENSE.bubble-pop](LICENSE.bubble-pop)).
- Post-processing: instances below score 0.65 are dropped, balloon interiors are
  hole-filled, the mask is shrunk by 3 px so the drawn outline survives painting.

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

Command line:

```bash
python balloon_segmentation.py samples output --paint
```

Outputs per image: `*_mask.png` (0/255), `*_overlay.jpg` (instances colored),
`*_painted.jpg` (with `--paint`: balloons painted white).

Options: `--min-score 0.65`, `--shrink 3`, `--no-fill`, `--model path/to/model.onnx`.

Python:

```python
from balloon_segmentation import BalloonSegmenter, mask_for_painting

seg = BalloonSegmenter("models/model_fp16.onnx")
mask, instances, mask_prob = seg.segment(image_bgr)   # mask: uint8 0/255, page size
paint_mask = mask_for_painting(image_bgr, seg)         # hole-filled, shrunk by 3 px
```

## Java (B4J / JavaObject)

`java/` is a Maven project around `com.xulihang.BalloonSegmenter`, usable from B4J
through JavaObject (depends on onnxruntime 1.20 and OpenCV 4.9).

```java
BalloonSegmenter seg = new BalloonSegmenter("models/model_fp32.onnx");
Mat mask = seg.getMask(pageMat, textBoxes);   // textBoxes may be null
```

```b4j
Dim seg As JavaObject
seg.InitializeNewInstance("com.xulihang.BalloonSegmenter", Array(modelPath))
Dim mask As JavaObject = seg.RunMethod("getMask", Array(pageMat.JO, boxes))
' boxes: list of maps with X, Y, width, height (also {"geometry": {...}} or [x0,y0,x1,y1])
' mask: org.opencv.core.Mat, 0/255, page size; release it when done
```

Text boxes: `List<Map>` with `X`, `Y`, `width`, `height` (page pixels), a map with a
`geometry` map, or `int[]`/`List` of `[x0, y0, x1, y1]`.

The official model has float16 I/O; the Java project uses `models/model_fp32.onnx`
(same weights, float32 I/O, outputs differ only by rounding). Regenerate it with
`python java/tools/convert_to_fp32.py model_fp16.onnx model_fp32.onnx`.

## Pure CV variant (no model)

`pure_cv_segmentation.py` recovers balloons from the text boxes with classic
OpenCV only (contour detection on closed ink, with a bounded growth fallback):

```bash
python pure_cv_segmentation.py samples output_cv --texts samples/texts.json --paint
```

Slower to get right than the model and less reliable on balloons whose outline
touches other white areas; kept for comparison.

## Model I/O

| | |
| :--- | :--- |
| Input | `image` `[1, 1024, 768, 1]` float16, grayscale (letterboxed, mean 0.449 / std 0.226) |
| Output | `logits` `[1, 1024, 768, 3]`, channels = mask, boundary, confidence |

Known limitations (from the model): transparent balloons and long thin tails may be missed.
