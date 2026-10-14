# simply-cpp-models: licences

These models are not simply-cpp's own work, and they are **not** under
simply-cpp's Apache-2.0 licence. Each keeps its upstream licence, and some
restrict how they may be used. Check the terms below before using them, and
before using them commercially in particular.

## Non-commercial research only: InsightFace

| File | Model |
|---|---|
| `det_10g.onnx` | SCRFD-10GF face detector, from InsightFace's `buffalo_l` model pack |
| `w600k_r50.onnx` | ArcFace ResNet-50 face recognition (WebFace600K), from `buffalo_l` |

Source: https://github.com/deepinsight/insightface (model packs:
https://github.com/deepinsight/insightface/tree/master/python-package).

InsightFace's code is MIT-licensed, but InsightFace makes its pretrained models
(and the training data they were made from) available for **non-commercial
research purposes only**. They are redistributed here unchanged, under those
terms. Commercial use needs a licence from InsightFace (see the licence section
of their README and https://www.insightface.ai).

## AGPL-3.0: Ultralytics YOLO

| File | Model |
|---|---|
| `yolo26n.onnx` | Ultralytics YOLO26 nano object detector (COCO), exported to ONNX |
| `numberplate.onnx` | Number plate detector, trained by Roelof Rossouw on top of `yolo26n` |

Source: https://github.com/ultralytics/ultralytics.

Ultralytics licenses YOLO, including its pretrained weights and models trained
from them, under the **GNU Affero General Public License v3.0**
(`licenses/AGPL-3.0.txt`); `numberplate.onnx` is a derivative of `yolo26n` and
is under the same licence. Software that uses them is bound by the AGPL's
terms, including making its source available to users (also over a network),
unless it has an Ultralytics Enterprise Licence instead:
https://www.ultralytics.com/license.

## Apache-2.0: PaddleOCR

| File | Model |
|---|---|
| `paddle_det_t.onnx`, `paddle_det_s.onnx` | PP-OCR text detection |
| `paddle_rec_t.onnx`, `paddle_rec_s.onnx` (+ `.txt` character lists) | PP-OCR text recognition |
| `paddle_det_monkt_5.onnx`, `paddle_rec_monkt_latin.onnx` (+ `.txt`) | PP-OCRv5 detection and Latin recognition, ONNX conversions published by monkt |
| `paddle_rotate.onnx` | PP-LCNet document orientation classifier |
| `paddle_line_rotate.onnx` | PP-LCNet text line orientation classifier |
| `paddle_unwarp.onnx` | UVDoc document unwarping (PaddleX) |

Sources: https://github.com/PaddlePaddle/PaddleOCR,
https://github.com/PaddlePaddle/PaddleX,
https://huggingface.co/monkt/paddleocr-onnx.

Copyright PaddlePaddle Authors, licensed under the Apache License 2.0
(`licenses/Apache-2.0.txt`). Redistributed as ONNX models, without changes to
their weights.
