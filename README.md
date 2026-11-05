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

## Model I/O

| | |
| :--- | :--- |
| Input | `image` `[1, 1024, 768, 1]` float16, grayscale (letterboxed, mean 0.449 / std 0.226) |
| Output | `logits` `[1, 1024, 768, 3]`, channels = mask, boundary, confidence |

Known limitations (from the model): transparent balloons and long thin tails may be missed.
