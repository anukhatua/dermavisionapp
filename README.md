# Deep Learning–Based Classification System for Human Skin Diseases

**Live Demo:** [DermavisionApp on Hugging Face Spaces](https://huggingface.co/spaces/anukhatua15/dermavision)  
**GitHub Repository:** https://github.com/Anushka0615/dermavisionapp.git  

---

## Project Overview  
This project presents an AI-based system for **automated classification of human skin diseases** using deep learning.  
A lightweight **MobileNetV2** model was trained on the **ISIC 2019 dataset** and then fine tuned and augmented to identify four common skin disease categories with **88.24 % accuracy**.  

The **DermavisionApp** web interface allows users to upload a skin image and receive instant predictions with confidence scores.  
Additional features include optional patient metadata (age, sex, medical history) and a **“Find a Dermatologist”** locator that redirects users to Google Maps.

---

## Objectives  
- Build an accurate CNN-based skin disease classifier.  
- Preprocess and analyze dermatological images for feature extraction.  
- Evaluate performance using Accuracy, Precision, Recall, and F1-Score.  
- Deploy a real-time, user-friendly web app for diagnosis assistance.  

---

## Methodology  
- **Architecture:** MobileNetV2 (Transfer Learning)  
- **Loss Function:** Sparse Categorical Cross-Entropy  
- **Optimizer:** Adam (with learning-rate scheduling)  
- **Metrics:** Accuracy | Precision | Recall | F1-Score | Confusion Matrix  
- **Frameworks / Tools:** Python | TensorFlow/Keras | NumPy | Pandas | Matplotlib | Seaborn  
- **Development Environment:** Google Colab / Jupyter Notebook  
- **Deployment Framework:** Flask  

---

## Dataset 
- **Dataset Used:** [ISIC 2019 Challenge Dataset](https://www.isic-archive.com)  
- **Classes:** 4 skin disease categories  
- **Preprocessing:** resizing (224×224), normalization, augmentation, class balancing.  

**References:**  
BCN_20000 Dataset: (c) Department of Dermatology, Hospital Clínic de Barcelona
HAM10000 Dataset: (c) by ViDIR Group, Department of Dermatology, Medical University of Vienna; https://doi.org/10.1038/sdata.2018.161
MSK Dataset: (c) Anonymous; https://arxiv.org/abs/1710.05006; https://arxiv.org/abs/1902.03368

---

## Model & Performance  
| Metric | Value |
|:--|:--|
| Architecture | MobileNetV2 |
| Accuracy | **88.24 %** |
| Precision | Balanced across classes |
| Recall | Balanced across classes |
| Environment | Google Colab |

- Minor misclassifications between visually similar diseases.  
- Patient metadata improves reliability.  
- Confidence score aids interpretability.  

---

## Repository Structure
dermavisionapp/
├── app.py # Flask application
├── final_model.h5 # Trained MobileNetV2 model
├── requirements.txt # Python dependencies
├── runtime.txt # Runtime environment
├── Procfile # Deployment configuration
├── static/ # CSS, JS, assets
├── templates/ # HTML templates
├── uploads/ # Uploaded user images
├── feedback/ # User feedback storage
└── README.md


## Run Locally  
```bash
git clone https://github.com/Anushka0615/dermavisionapp.git
cd dermavisionapp
pip install -r requirements.txt
python app.py
```
---

## Note on Training Code
> The **training scripts are not included** in this repository.  
> The `final_model.h5` file was trained separately in Google Colab using the **ISIC 2019 dataset** and **MobileNetV2** architecture.  
> Only the **final trained model** and **deployment files** are provided here for demonstration and application use.

---

## Features
- **Instant Prediction:** Upload an image and get real-time classification results with confidence percentage.  
- **Patient Metadata Input:** Option to add age, sex, and medical history to refine predictions.  
- **Dermatologist Finder:** Redirects users to Google Maps to locate nearby dermatologists.  
- **Confidence Display:** Shows how sure the model is for each prediction.  
- **Clean Flask UI:** Simple, user-friendly web interface suitable for all devices.

---

## Future Enhancements
- Expand to include more skin disease categories.  
- Integrate **Grad-CAM** or similar visualizations for explainability.  
- Add **multi-language** and **mobile-responsive** versions.  
- Connect with medical databases for continuous learning and updates.  
- Improve model interpretability through patient data integration.

---

## Authors
**Anushka Khatua** – M.Tech in AI & DS (KIIT University, Bhubaneswar)   
*Advanced Industry Integrated Program – LTIMindTree & KIIT University*

---

## License
This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute this project with proper attribution.

---
