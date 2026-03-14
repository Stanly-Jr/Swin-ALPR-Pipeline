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

Ensure you have Python 3.8+ installed. Install the required dependencies using pip


Note: For optimal performance and Swin2SR inference speed, a CUDA-enabled GPU is highly recommended.

📥 Model Weights
To run the code, you will need the following trained weight files in your project directory:

runs/detect/train7/weights/best.pt (YOLO character detection weights)

swin2sr_v2_plate_epoch2.pth (Fine-tuned Swin2SR weights)

ocr_model_deep_v1.pth (Custom Deep CNN OCR weights)

(Note: If you are cloning this repo, please download the weight files from the Releases tab and place them in the root directory).

💻 Usage
1. Single Image Inference
Update the test_image_path variable in the script to point to your image, then run:

```Bash
python inference_single.py
```
2. Batch Processing
Update the target_dir variable in the script to point to your folder of images. The script will process them and output results sequentially:

```Bash
python inference_batch.py
```
3. Evaluation Setup
To run the evaluation script against the UFPR-ALPR (or custom) dataset, ensure you have a labels.csv file with filename and plate columns. Update the paths in the main execution block:

```Bash
python eval_pipeline.py
```
🔍 Pipeline Details
Smart Cropping: Adds a 20% margin to YOLO bounding boxes so the Swin2SR model has surrounding context to accurately restore edges.

Tight Trim: After upscaling, the pipeline shaves off 10% of the margins to provide the OCR CNN with a perfectly centered, tight crop of the character.

Illumination Handling: Includes utility functions (is_dark_on_dark, is_red_background) to dynamically normalize contrast and invert colors for specific challenging plate formats before they hit the OCR model.

🤝 Acknowledgments

Dataset utilized for testing: UFPR-ALPR Dataset

Swin2SR by caidas

YOLO by Ultralytics
