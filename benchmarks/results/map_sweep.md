# mAP sweep — pytorch,coreml-fp16,coreml-int8,onnx-fp32 × 640,512,384

data: `dataset/data.yaml` (n=56 val images, split='val') | note: val n=56 ⇒ sub-~1.5pt deltas are within noise.

| config | imgsz | mAP50 | mAP50-95 | artifact size MB | notes |
|---|---:|---:|---:|---:|---|
| pytorch | 640 | 0.9950 | 0.9502 | 18.29 |  |
| pytorch | 512 | 0.9950 | 0.9595 | 18.29 |  |
| pytorch | 384 | 0.9950 | 0.9686 | 18.29 |  |
| coreml-fp16 | 640 | 0.9950 | 0.9567 | 18.17 |  |
| coreml-fp16 | 512 | nan | nan | 18.17 | n/a — fixed-shape @640 |
| coreml-fp16 | 384 | nan | nan | 18.17 | n/a — fixed-shape @640 |
| coreml-int8 | 640 | 0.9950 | 0.9529 | 9.25 |  |
| coreml-int8 | 512 | nan | nan | 9.25 | n/a — fixed-shape @640 |
| coreml-int8 | 384 | nan | nan | 9.25 | n/a — fixed-shape @640 |
| onnx-fp32 | 640 | 0.9950 | 0.9551 | 36.21 |  |
| onnx-fp32 | 512 | nan | nan | 36.21 | n/a — fixed-shape @640 |
| onnx-fp32 | 384 | nan | nan | 36.21 | n/a — fixed-shape @640 |
