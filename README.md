[![Brand detection using YOLO](https://miro.medium.com/v2/resize:fit:800/1*pRtS_Zhl09gsDrcShqlnvQ.png)](https://medium.com/@6431322821/brand-detection-using-yolov8-b1155e1a5fa4)

# BrandI – Deep Learning for Logo Detection

BrandI is a computer vision system that detects brand logos in real-world images using deep learning. It is built to support marketing, PR, and brand safety teams in tracking brand exposure across public visuals and social media.

## Key Features

- Real-time logo detection using YOLOv8
- Fine-tuned on 27 branded logos (~1,000+ images)
- High performance with strong precision and recall
- Pre-trained model and Streamlit-based UI for image testing

## Setup Instructions

To create a new Conda virtual environment and install the required dependencies:

```bash
# Create a new Conda environment
conda create -n brandi-env python=3.10

# Activate the environment
conda activate brandi-env

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

## Streamlit App – Test an Image

To launch the Streamlit interface:

```bash
streamlit run main.py
```

Upload an image and detect logos directly in the browser.

## Preprocess Before Prediction

Make sure to preprocess your test image before passing it to the model:

```python
from PIL import Image
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

def preprocess_and_predict(image_path, model_path, save_path="/flickr_logo_yolo_project/predicted_images/", conf=0.3):
    """
    Preprocesses a test image to match YOLOv8 training style and runs prediction.
    - Resizes to 416x416
    - Converts to RGB
    - Runs YOLOv8 prediction
    - Saves and displays result
    """

    # Load model
    model = YOLO(model_path)

    # Preprocess image: RGB and resize to 416x416
    img = Image.open(image_path).convert("RGB").resize((416, 416))
    temp_path = "temp_resized.jpg"
    img.save(temp_path)

    # Predict on the resized image
    results = model.predict(source=temp_path, conf=conf, imgsz=416)

    # Visualize and save output
    result_img = results[0].plot()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, result_img)

    # Show inline
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Prediction")
    plt.show()

    return results[0]

model_path = "" #Add model path
test_image = "" #Add test image path
preprocess_and_predict(test_image, model_path)
```

## Train a New Model

1. Download the dataset:
   [flickr_logo_yolo_project.zip](https://drive.google.com/file/d/1Bc-x3Gk00WPzIdOz7Een_lVr2hto_yQU/view?usp=sharing)

2. Unzip the file and place the contents in your working directory.

3. Train using the following notebook:

```bash
python training_dataset.ipynb
```

Trained weights will be saved in the `models/` directory.

## Project Structure

```
BrandI/
├── main.py                   # Streamlit app for inference
├── models/                   # Folder for saved YOLOv8 weights
│   └── best.pt               # Latest trained model
├── README.md                 # Project documentation
├── requirements.txt          # Dependencies
├── runtime.txt               # Python version
├── trained_data.zip          # Pretrained model weights and outputs
├── training_dataset.ipynb    # Notebook for training the model
```

## Model Performance

| Metric         | Value     |
|----------------|-----------|
| Precision      | 96.9%     |
| Recall         | 95.7%     |
| mAP@0.5        | 92.1%     |
| mAP@0.5:0.95   | 81.5%     |
| Epochs         | 30        |

Framework: PyTorch  
Model: YOLOv8  
UI: Streamlit  
Image preprocessing: Pillow

## Known Limitations

- Limited to 27 logo classes
- May miss small, obscured, or low-resolution logos
- No sentiment or context analysis

## Potential Improvements

- Caption generation with transformer models
- Synthetic data generation using GANs
- Scalable API deployment for enterprise use

## Authors

Tara Teylouni  
Afonso Vaz Santos  
Samir Barakat  
Nour Sewilam  
Yotaro Enomoto
