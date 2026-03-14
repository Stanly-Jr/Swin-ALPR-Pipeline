Robust ALPR Pipeline: YOLO + Swin2SR + Custom Deep OCR

A high-performance Automatic License Plate Recognition (ALPR) pipeline that handles low-resolution, noisy, and challenging license plates. This project improves standard OCR accuracy by applying state-of-the-art super-resolution to individual character crops before classification.

🚀 Overview

Traditional ALPR systems often fail when characters are pixelated or distorted. This pipeline solves that by using a three-stage approach:
1. Character Detection: A fine-tuned YOLO model detects bounding boxes for individual characters on the raw image.
2. Super-Resolution Upscaling: Detected crops are padded, extracted, and passed through a fine-tuned **Swin2SR** model to restore edges and clarify the character.
3. ptical Character Recognition (OCR): A custom Deep CNN classifies the restored character.

Additional logic includes dynamic reading-order sorting (handling both single-line and split-line plates), false-positive filtering based on geometric median heights, and rule-based disambiguation (e.g., confusing `0` and `O`, or `8` and `B`).

🧠 Model Architecture

* Detector: YOLOv8 (Custom trained for character detection).
* Super Resolution: Swin2SR (`caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr` base, fine-tuned on plate crops).
* OCR: Custom PyTorch CNN (trained from scratch on 36 alphanumeric classes).

📁 Repository Structure

* `eval_pipeline.py`: Runs evaluation metrics (Plate Accuracy, Character Accuracy, CER, Timing) against a ground-truth CSV dataset.
* `inference_single.py`: Runs the pipeline on a single image and prints the recognized text.
* `inference_batch.py`: Iterates through a target directory of images, processing them one by one with a clean progress bar.

⚙️ Installation & Requirements

Ensure you have Python 3.8+ installed. Install the required dependencies using pip:
