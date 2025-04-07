# 🍅 Tomato Leaf Disease Classification 🌿

This project classifies tomato leaf diseases using a Convolutional Neural Network (CNN) based on the VGG16 architecture. Early and accurate detection of these diseases can help farmers take quick action and improve crop productivity.

---

## 🧠 Overview

Using a fine-tuned **VGG16** model, this classifier identifies tomato leaves into the following 10 categories:

- Tomato Bacterial Spot  
- Tomato Early Blight  
- Tomato Late Blight  
- Tomato Leaf Mold  
- Tomato Septoria Leaf Spot  
- Tomato Spider Mites (Two-Spotted Spider Mite)  
- Tomato Target Spot  
- Tomato Mosaic Virus  
- Tomato Yellow Leaf Curl Virus  
- Tomato Healthy

---

## 📁 Dataset

The dataset is sourced from the PlantVillage Dataset on Kaggle, containing over 11,000 labeled images of tomato leaves with and without diseases.

### Preprocessing:
- Image resizing to 224x224 (input size for VGG16)
- Normalization
- Data augmentation (rotation, zoom, horizontal flip, etc.)

---

## ⚙️ Tech Stack

- Python 🐍  
- TensorFlow / Keras  
- OpenCV  
- NumPy, Matplotlib, Pandas  
- Streamlit (for GUI)

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/tomato-leaf-disease-classification.git
   cd tomato-leaf-disease-classification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**
   ```bash
   python train.py
   ```

4. **Predict using a single image**
   ```bash
   python predict.py --image path_to_image.jpg
   ```

5. **Run the Streamlit web app**
   ```bash
   streamlit run app.py
   ```

---

## 📊 Results

- **Model Architecture:** VGG16 (pretrained on ImageNet, fine-tuned on PlantVillage Tomato subset)
- **Test Accuracy:** **98.82%**
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam
- **Epochs Trained:** (Add your number here)

### Evaluation Metrics:
- Confusion Matrix
- Precision, Recall, F1-score
- Accuracy Graphs and Loss Curves

---

## ✅ Features

- Pre-trained VGG16 with fine-tuning
- High-accuracy classification (98.82%)
- Real-time or static image prediction
- Interactive web app using Streamlit
- Clean UI and easy deployment

---
