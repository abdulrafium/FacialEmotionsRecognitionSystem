# app.py
import streamlit as st
import cv2
from keras.applications import MobileNetV2
from keras import models, layers, regularizers
import numpy as np
import time
import os
import json
from datetime import datetime
import plotly.graph_objects as go
import uuid
from pathlib import Path
from streamlit.components.v1 import html as st_html
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import threading

# ---------------- Page config ----------------
st.set_page_config(
    page_title="EmotionSense AI",
    page_icon="😊",
    layout="wide",
)

# -------------------files -------------------
CANDIDATE_WEIGHT_FILES = [
    "best_model_weights.h5",
    #"best_model.weights.h5",
    #"best_model_weights.weights.h5",
    #"best_model.weights.h5",
    #"best_model.h5",
    #"best_model.keras"
]

# choose first existing
MODEL_WEIGHTS_PATH = None
for fn in CANDIDATE_WEIGHT_FILES:
    if os.path.exists(fn):
        MODEL_WEIGHTS_PATH = fn
        break

# If none found, still set default so error message is clear
if MODEL_WEIGHTS_PATH is None:
    MODEL_WEIGHTS_PATH = "best_model_weights.h5"  # default — user can change

PREDICTIONS_FILE = "predictions.json"
ASSETS_DIR = Path("assets")
SPLASH_GIF = ASSETS_DIR / "loading.gif"
LOGO = ASSETS_DIR / "logo.png"

# Ensure predictions file exists
if not os.path.exists(PREDICTIONS_FILE):
    with open(PREDICTIONS_FILE, "w") as f:
        json.dump([], f)

# ---------------- Utility functions ----------------
def save_prediction(entry):
    try:
        with open(PREDICTIONS_FILE, "r+") as f:
            data = json.load(f)
            data.append(entry)
            f.seek(0)
            json.dump(data, f, indent=2)
    except Exception as e:
        st.error(f"Failed to save prediction: {e}")

def load_predictions(limit=50):
    try:
        with open(PREDICTIONS_FILE, "r") as f:
            data = json.load(f)
        return list(reversed(data))[:limit]
    except:
        return []

def preprocess_image_for_model(img_bgr):
    """
    Input: BGR image (as from cv2)
    Output: numpy array shape (1,48,48,3), float32, normalized 0-1
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (48, 48))
    arr = img_resized.astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape: (1,48,48,3)
    return arr

def speak_text_js(text):
    escaped = text.replace('"', '\\"')
    js = f"""
    <script>
    const speak = () => {{
        const utterance = new SpeechSynthesisUtterance("{escaped}");
        utterance.lang = 'en-US';
        speechSynthesis.cancel();
        speechSynthesis.speak(utterance);
    }};
    speak();
    </script>
    """
    return js

# ---------------- Load model (weights-only if possible) ----------------
@st.cache_resource(show_spinner=False)
def load_emotion_model(weights_path):
    # If a full saved model file (.keras or .h5 non-weights) exists, try load_model first
    try:
        if weights_path.endswith(".keras") or (weights_path.endswith(".h5") and "weights" not in weights_path):
            # attempt to load full model (fallback)
            from keras.models import load_model
            m = load_model(weights_path)
            return m
    except Exception:
        # continue to try weights loading
        pass

    # Rebuild the architecture (must match training)
    feature_base = MobileNetV2(input_shape=(48,48,3), include_top=False, weights='imagenet')
    for layer in feature_base.layers[:100]:
        layer.trainable = False

    emotion_model = models.Sequential([
        feature_base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(5, activation='softmax')
    ])

    # Try several possible weight filenames if the provided path doesn't exist
    if not os.path.exists(weights_path):
        # try find any .h5 file that contains 'best' or 'weights'
        for fn in os.listdir():
            if fn.endswith(".h5") and ("best" in fn or "weights" in fn):
                weights_path = fn
                break

    # Final check
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found. Tried: {weights_path}")

    # load weights into rebuilt model
    emotion_model.load_weights(weights_path)
    return emotion_model

# Load model with friendly message
try:
    model = load_emotion_model(MODEL_WEIGHTS_PATH)
    st.success(f"Model loaded from: {MODEL_WEIGHTS_PATH}")
except Exception as e:
    st.error(f"Failed to load model weights from {MODEL_WEIGHTS_PATH}: {e}")
    st.stop()

EMOTION_LABELS = ["Happy", "Neutral", "Sad", "Angry", "Surprise"]

def predict_emotion_from_face(face_bgr):
    x = preprocess_image_for_model(face_bgr)
    preds = model.predict(x, verbose=0).reshape(-1)
    idx = int(np.argmax(preds))
    return {
        "emotion": EMOTION_LABELS[idx],
        "confidence": float(preds[idx]),
        "vector": [float(p) for p in preds]
    }

# Face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ---------------- Splash screen ----------------
if "splash_done" not in st.session_state:
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:center;flex-direction:column;margin-top:20px;">
        <h1 style="font-family:Inter, sans-serif;font-size:40px;color:#2b6cb0;margin:6px 0;">EmotionSense AI</h1>
        <h3 style="color:#4a5568;margin:0;">Facial Emotion Recognition System</h3>
        <p style="color:#718096;margin-top:6px;">Developed by <strong>Abdul Rafiu</strong></p>
    </div>
    """, unsafe_allow_html=True)
    if SPLASH_GIF.exists():
        st.image(str(SPLASH_GIF), width=220)
    progress_text = st.empty()
    progress_bar = st.progress(0)
    for i in range(1,101):
        progress_text.markdown(f"<div style='text-align:center;color:#2d3748'>Initializing... {i}%</div>", unsafe_allow_html=True)
        progress_bar.progress(i)
        time.sleep(0.006)
    progress_bar.empty()
    progress_text.empty()
    st.session_state.splash_done = True
    st.rerun()

# ---------------- Sidebar ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Live Camera", "Upload Image", "History", "About"])
st.sidebar.markdown("---")
speak_toggle = st.sidebar.checkbox("🔊 Enable voice output (browser)", value=True)
auto_save_toggle = st.sidebar.checkbox("💾 Auto-save predictions", value=True)
st.sidebar.markdown("---")
st.sidebar.caption("Tip: allow camera access for Live Camera.")

# ---------------- PAGE: Home ----------------
if page == "Home":
    col1, col2 = st.columns([1, 2])
    with col1:
        if LOGO.exists():
            st.image(str(LOGO), width=140)
        st.markdown("### Welcome to EmotionSense AI")
        st.markdown(
            """
            **Quick actions**
            - Use **Live Camera** to detect emotions in real-time.  
            - Use **Upload Image** to test single photos.  
            - View **History** to see last detections.
            """
        )
        st.info("This app runs the model locally. Voice uses your browser's speech synthesis.")
    with col2:
        st.markdown("### Recent predictions")
        recent = load_predictions(limit=5)
        if recent:
            for item in recent[:5]:
                t = item.get("timestamp", "")
                src = item.get("source", "")
                emo = item.get("emotion", "")
                conf = item.get("confidence", 0)
                st.markdown(f"**{t}** • {src} → **{emo}** ({conf*100:.1f}%)")
        else:
            st.write("No predictions yet. Try Upload Image or Live Camera.")

# ---------------- PAGE: Upload Image ----------------
elif page == "Upload Image":
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image file (jpg, png)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Use getvalue() to avoid buffer consumption issues
        file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Uploaded file could not be read as image.")
        else:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded image", use_container_width=True)
            st.markdown("### Detection")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
            if len(faces) == 0:
                st.warning("No faces found in the image.")
            else:
                results = []
                for (x, y, w, h) in faces:
                    face = image[y:y+h, x:x+w]
                    pred = predict_emotion_from_face(face)
                    results.append(pred)
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)
                    label = f"{pred['emotion']} ({pred['confidence']*100:.1f}%)"
                    cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Result", use_container_width=True)

                # Combine softmax vectors by averaging if multiple faces
                avg_vector = np.mean([r["vector"] for r in results], axis=0)
                top_idx = int(np.argmax(avg_vector))
                final_emotion = EMOTION_LABELS[top_idx]
                final_conf = float(avg_vector[top_idx])

                # Plot percentages
                fig = go.Figure([go.Bar(x=EMOTION_LABELS, y=list(avg_vector))])
                fig.update_layout(title="Emotion probabilities", yaxis=dict(tickformat=".2f"), height=350)
                st.plotly_chart(fig, use_container_width=True)

                # Save and speak
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                entry = {
                    "id": str(uuid.uuid4()),
                    "timestamp": timestamp,
                    "source": "upload",
                    "emotion": final_emotion,
                    "confidence": final_conf,
                    "vector": [float(v) for v in avg_vector.tolist()]
                }
                if auto_save_toggle:
                    save_prediction(entry)
                    st.success(f"Saved: {final_emotion} ({final_conf*100:.1f}%)")
                if speak_toggle:
                    text = f"You look {final_emotion}"
                    st_html(speak_text_js(text), height=0)

# ---------------- PAGE: Live Camera ----------------
elif page == "Live Camera":
    st.header("Live Camera — Real-time Detection")
    st.info("📸 Camera will stream continuously without flickering. Click STOP in the video widget to stop.")
    
    # Initialize session state
    if "emotion_history" not in st.session_state:
        st.session_state.emotion_history = []
    if "last_saved_time" not in st.session_state:
        st.session_state.last_saved_time = 0
    
    # Video processor class for streamlit-webrtc
    class EmotionVideoProcessor:
        def __init__(self):
            self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self.last_predictions = []
            self.lock = threading.Lock()
            self.frame_count = 0
            self.skip_frames = 2  # Process every 3rd frame for speed
            self.last_faces = []  # Store last detected faces for stability
            self.last_emotion = {}  # Store last emotion per face
        
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Skip frames to improve performance
            self.frame_count += 1
            should_detect = self.frame_count % self.skip_frames == 0
            
            if should_detect:
                # Detect faces (more stable parameters)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Apply histogram equalization for better detection
                gray = cv2.equalizeHist(gray)
                faces = self.face_detector.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=7,  # Higher = more stable, fewer false positives
                    minSize=(80, 80),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                if len(faces) > 0:
                    self.last_faces = faces
                    
                    agg_vectors = []
                    for idx, (x, y, w, h) in enumerate(faces):
                        face = img[y:y+h, x:x+w]
                        pred = predict_emotion_from_face(face)
                        agg_vectors.append(pred["vector"])
                        
                        # Store emotion for this face position
                        face_key = f"{idx}"
                        self.last_emotion[face_key] = pred
                    
                    # Store predictions
                    with self.lock:
                        self.last_predictions = agg_vectors
            
            # Always draw the last detected faces (even on skipped frames for stability)
            if len(self.last_faces) > 0:
                for idx, (x, y, w, h) in enumerate(self.last_faces):
                    face_key = f"{idx}"
                    if face_key in self.last_emotion:
                        pred = self.last_emotion[face_key]
                        label = f"{pred['emotion']} ({pred['confidence']*100:.1f}%)"
                        
                        # Draw stable rectangle and text
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                        
                        # Background for text (makes it more readable)
                        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(img, (x, y-text_height-10), (x+text_width, y), (0, 255, 0), -1)
                        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    # Create columns
    cam_col1, cam_col2 = st.columns([2, 1])
    
    with cam_col2:
        st.write("### Settings")
        st.caption("Adjust detection sensitivity")
        sensitivity_info = st.empty()
        sensitivity_info.info("Face detection: Standard")
        
        st.markdown("---")
        st.write("### Status")
        status_display = st.empty()
        status_display.info("⚫ Waiting for camera...")
    
    with cam_col1:
        # WebRTC configuration
        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        # Create video processor instance
        processor = EmotionVideoProcessor()
        
        # Start webrtc streamer
        ctx = webrtc_streamer(
            key="emotion-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            video_processor_factory=lambda: processor,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 30}
                }, 
                "audio": False
            },
            async_processing=True,
        )
        
        # Monitor and save predictions
        if ctx.state.playing:
            status_display.success("🟢 Camera Active - Detecting emotions...")
            
            # Check for predictions periodically (less frequently)
            placeholder = st.empty()
            while ctx.state.playing:
                time.sleep(3)  # Check every 3 seconds to reduce overhead
                
                with processor.lock:
                    if len(processor.last_predictions) > 0:
                        # Aggregate predictions
                        agg = np.mean(np.array(processor.last_predictions), axis=0)
                        top_idx = int(np.argmax(agg))
                        final_emotion = EMOTION_LABELS[top_idx]
                        final_conf = float(agg[top_idx])
                        
                        # Save prediction
                        now = time.time()
                        if (now - st.session_state.last_saved_time) > 2:
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            entry = {
                                "id": str(uuid.uuid4()),
                                "timestamp": timestamp,
                                "source": "camera",
                                "emotion": final_emotion,
                                "confidence": final_conf,
                                "vector": [float(v) for v in agg.tolist()]
                            }
                            
                            if auto_save_toggle:
                                save_prediction(entry)
                                status_display.success(f"✅ {final_emotion} ({final_conf*100:.1f}%) - {len(processor.last_predictions)} face(s)")
                            
                            if speak_toggle:
                                st_html(speak_text_js(f"You look {final_emotion}"), height=0)
                            
                            st.session_state.last_saved_time = now
                        
                        processor.last_predictions = []  # Clear after processing
        else:
            status_display.info("⚫ Camera stopped. Click START to begin.")

# ---------------- PAGE: History ----------------
elif page == "History":
    st.header("Prediction History")
    history_data = load_predictions(limit=200)
    if not history_data:
        st.write("No saved predictions yet.")
    else:
        for item in history_data:
            t = item.get("timestamp", "")
            src = item.get("source", "")
            emo = item.get("emotion", "")
            conf = item.get("confidence", 0)
            with st.expander(f"{t} • {src} • {emo} ({conf*100:.1f}%)"):
                st.json(item)
        if st.button("Clear history (delete file)"):
            try:
                os.remove(PREDICTIONS_FILE)
                with open(PREDICTIONS_FILE, "w") as f:
                    json.dump([], f)
                st.success("History cleared.")
            except Exception as e:
                st.error(f"Could not clear history: {e}")

# ---------------- PAGE: About ----------------
elif page == "About":
    st.header("About this project")
    st.markdown(
        """
        **EmotionSense AI** is a Facial Emotion Recognition prototype by **Abdul Rafiu**.
        - Model input: 48x48 RGB images (derived from faces).
        - Backend: TensorFlow / Keras model (MobileNetV2-based fine-tuned).
        - This Streamlit app demonstrates webcam & image inference, history logging, voice feedback, and charts.
        """
    )
    st.markdown("**Notes**")
    st.markdown("- Ensure your weights file (e.g. best_model_weights.h5) is in the same folder as app.py.")
    st.markdown("- Allow camera access in the browser for Live Camera.")
