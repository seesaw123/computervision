# AI Vision Lab — M4–M6 Teaching Portfolio

Three progressively advanced computer vision teaching modules.

| Grade | Topic | Training |
|---|---|---|
| M4 | Image classification | Train a small CNN from scratch on synthetic circles, squares and triangles |
| M5 | Object detection | Fine-tune YOLO with a custom annotated dataset |
| M6 | Model evaluation | Compare two detectors on the same held-out test dataset |

## Install (Mac/Linux)

Requires Python 3.10–3.12.

```bash
git clone https://github.com/seesaw123/computervision.git
cd computervision
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/train_m4.py --epochs 8
streamlit run app.py
```

M4: This code generates synthetic images, trains and evaluates a CNN. The original downloadable ZIP includes an M4 checkpoint, but checkpoints and synthetic images are excluded from Git to keep the repository small. Retrain locally using the command above. A prior run achieved 69.0% test accuracy on 42 synthetic images; that is not evidence of real-world performance.

M5: Collect and annotate images (see `data/custom_objects/README.md`), copy `data.example.yaml` to `data.yaml`, then train:

```bash
python scripts/train_m5.py --data data/custom_objects/data.yaml --epochs 30 --batch 4 --imgsz 416 --device cpu
```

M6: Train a second checkpoint, then compare on the *same held-out test split*:

```bash
python scripts/evaluate_m6.py --data data/custom_objects/data.yaml --models runs/m5/custom_objects/weights/best.pt /path/to/second/best.pt --split test
```

**Status:** M4 training code and previously measured synthetic experiment; M5/M6 training/evaluation pipelines ready but no real custom dataset, YOLO weights, or M6 comparison results included. Obtain consent and avoid identifiable student images. CPU YOLO fine-tuning can be slow.

Docs: https://docs.ultralytics.com/ and https://docs.pytorch.org/tutorials/
