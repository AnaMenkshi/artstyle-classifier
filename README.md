# artstyle-classifier
AI project to classify artworks by painting style.

# 🎨 ArtStyle Classifier

A machine learning project leveraging AI and deep learning (CNNs) to classify paintings into 5 distinct artistic styles.  
Built by Ana Menkshi and Fatjona Murrani, this system also includes an interactive web app for real-time predictions.

## Features
- Classifies paintings into 5 major artistic styles
- Uses a fine-tuned **ResNet50** backbone with custom layers
- Includes an easy-to-use **web app** (Gradio / Streamlit)
- Deployed on Hugging Face Spaces for public use

Try the App: [ArtStyle Classifier on Hugging Face](https://huggingface.co/spaces/menkshi/ARTAF)
The username:art ,the pasword:art.
---

## Dataset
- Dataset: `painter-by-numbers.zip`
- Images: `.jpg` files of paintings
- Metadata: `.csv` file with painting info
- Filtered to 5 distinct styles to improve model learning

---

##  Preprocessing & Augmentation
- Images resized to **224x224** pixels
- Normalized pixel values `[0, 1]`
- Augmentation techniques:
  - Rotation (±10°)
  - Shear (5%)
  - Horizontal & Vertical Flips
  - Rescaling
- Dataset split: **80% training / 20% validation**

---

##  Model Architecture

- **Base Model:** ResNet50 (`include_top=False`)
- **Custom Layers:**
  - Flatten ➔ Dense(512, He init) ➔ Dropout ➔ BatchNorm ➔ ReLU
  - Dense(16) ➔ Dropout ➔ BatchNorm ➔ ReLU
  - Output Layer: Softmax (5 classes)

**Training Config:**
- Optimizer: Adam (lr=0.0001)
- Loss: Categorical Crossentropy
- Batch size: 32
- Metrics: Accuracy

---

##  Web App
- Built with **Gradio** 
- Features:
  - Upload a painting ➔ Get style prediction
  - User-friendly and responsive interface
- Deployed via **Hugging Face Spaces**

---

##  Installation

1. Clone this repository:
```bash
git clone https://github.com/AnaMenkshi/artstyle-classifier.git
cd artstyle-classifier


