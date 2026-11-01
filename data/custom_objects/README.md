# Custom object dataset — not included

Choose 2–4 classes such as calculator, marker and notebook. Use consented photographs or properly licensed images.

Create images/train, images/val, images/test and matching labels/train, labels/val, labels/test.

Each image's matching text file uses one YOLO bounding box per line:
`class_id x_center y_center width height` (normalized coordinates 0–1).

Copy data.example.yaml to data.yaml and edit the class names. Split by scene or session to avoid data leakage. Never evaluate with the training split. Begin with 30–50 well-annotated images per class as a pilot, then collect more diverse data based on errors.
