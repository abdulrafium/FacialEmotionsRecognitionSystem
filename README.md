# EmotionSense AI – Facial Emotion Recognition System

EmotionSense AI is a deep learning–based Facial Emotion Recognition application developed using Streamlit, TensorFlow/Keras, and OpenCV. The system detects human emotions from facial images and live webcam streams and displays prediction confidence through an interactive web interface.

---

## Dataset Information

This project uses the **FER2013 dataset** downloaded from **Kaggle**.

- Dataset: FER2013  
- Source: Kaggle Facial Expression Recognition Challenge  
- Image size: 48×48  

Note: The dataset is not included in this repository due to size limitations and must be downloaded manually from Kaggle.

---

## Model & Weights

- The project folder contains pre-trained weights named `best_model_weights.h5`
- These weights are trained on the FER2013 dataset
- The Streamlit application loads these weights directly for inference

---

## Model Training (Optional)

To retrain or customize the model:

- Use the Google Colab notebook `FER_DL_FinalProject.ipynb`
- Run the notebook in Google Colab
- Download the FER2013 dataset from Kaggle
- Import the dataset in Colab and train the model
- Save the trained weights and replace `best_model_weights.h5` if needed

---

## Features

- Real-time facial emotion detection using webcam
- Emotion prediction from uploaded images
- Face detection using OpenCV Haar Cascades
- Emotion confidence visualization
- Prediction history logging
- Optional voice feedback using browser speech synthesis

---

## Tools & Technologies

- Python
- Streamlit
- TensorFlow / Keras
- MobileNetV2
- OpenCV
- Plotly
- Google Colab (for training)

---

## Project Structure

EmotionSense-AI/
├── app.py
├── best_model_weights.h5
├── FER_DL_FinalProject.ipynb
├── predictions.json
├── requirements.txt
└── README.md


## How to Run

1. Navigate to your project folder:
   ```bash
   cd path/to/EmotionSense-AI

2. Create a virtual environment:
   python -m venv venv

3. Activate the virtual environment (windows): 
   venv\Scripts\activate

4. Install dependencies from requirements.txt:
   pip install -r requirements.txt

5. Run the Streamlit application:
   streamlit run app.py

## Notes
- Ensure the weights file best_model_weights.h5 is present in the project folder.
- To retrain the model, run FER_DL_FinalProject.ipynb in Google Colab with the FER2013 dataset from Kaggle.

## Author
Abdul Rafiu
Department of Computer Science