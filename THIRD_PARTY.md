# Third-party notices

This project uses the following third-party components.  The list is a
practical summary, not legal advice; review the upstream licenses before
commercial deployment.

| Component | Use | License / note |
|---|---|---|
| Ultralytics YOLO11 | person/vehicle/plate/face detection | AGPL-3.0; commercial use requires an Ultralytics license |
| OpenCV 5.0.0 | image/video processing, YuNet, EAST | Apache-2.0 |
| MediaPipe | 478-point face landmarks, selfie multiclass | Apache-2.0 |
| SAM 2.1 (Meta) | box-prompted face masks | Apache-2.0 (check the upstream model card) |
| jonathandinu/face-parsing | SegFormer face parsing (includes `eye_g` glasses) | see Hugging Face model card |
| RapidOCR / PP-OCR | plate/document text-line masks | Apache-2.0 (RapidOCR; PP-OCR upstream terms) |
| ByteTrack via supervision | data association | MIT (supervision); ByteTrack paper terms |
| One Euro Filter | box smoothing | BSD-style reference implementation; see Casiez et al. |
| Streamlit | web UI | Apache-2.0 |
| Modal | GPU deployment | Modal terms of service |
| Gemini API | optional VLM analysis | Google API terms |
| PyTorch / torchvision | model runtime | BSD-style |

## Model weights

- `yolo11*-seg.pt`, `yolo11n-pose.pt`: Ultralytics (AGPL-3.0).
- `license-plate-finetune-v1n.pt`: see the Hugging Face model card.
- `sam2.1_hiera_tiny.pt`: Meta SAM 2.1 (Apache-2.0).
- `face_landmarker.task`, `selfie_multiclass_256x256.tflite`: Google MediaPipe (Apache-2.0).
- `frozen_east_text_detection.pb`: OpenCV/EAST sample (Unlicense).
- `jonathandinu/face-parsing`: Hugging Face model card.

## Deployment note

The Ultralytics AGPL license is the most important item for a public
network service.  If this project is distributed or operated as a service,
confirm the AGPL obligations or obtain a commercial Ultralytics license.
