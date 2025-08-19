import streamlit as st
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import tempfile
import os

# --- 1. UI Configuration and Styling ---
st.set_page_config(page_title="Drowsiness Detection", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
body {
    background: linear-gradient(to right, #232526, #414345);
    font-family: 'Poppins', sans-serif;
}
.main-container {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
}
h1, h2, h3, p, label { color: #FFFFFF; }
h1 {
    text-align: center;
    font-weight: 600;
    font-size: 3rem;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.2);
}
p { text-align: center; font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)

# --- 2. Load Models and Classifiers (Updated for TFLite) ---
@st.cache_resource
def load_all_models():
    """Loads the TFLite model and OpenCV classifiers."""
    interpreter = tflite.Interpreter(model_path='./models/model.tflite')
    interpreter.allocate_tensors()
    
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    return interpreter, face_cascade, eye_cascade

interpreter, face_cascade, eye_cascade = load_all_models()

# --- 3. Core Detection Logic (Updated for TFLite) ---
def process_frame(frame, drowsy_counter_ref):
    IMG_SIZE = (80, 80)
    ALARM_THRESHOLD = 20
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    status = "Alert"
    color = (0, 255, 0)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) >= 2:
            ex, ey, ew, eh = eyes[0]
            eye_region = roi_gray[ey:ey+eh, ex:ex+ew]
            
            if eye_region.size > 0:
                eye_image_resized = cv2.resize(eye_region, (IMG_SIZE))
                eye_image_normalized = eye_image_resized / 255.0
                
                # Prepare input for TFLite model
                model_input = np.array(eye_image_normalized, dtype=np.float32).reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)
                
                # Get input and output details
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                # Set tensor, invoke, and get prediction
                interpreter.set_tensor(input_details[0]['index'], model_input)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

                if prediction > 0.5:
                    drowsy_counter_ref['count'] += 1
                    status = "Drowsy"
                    color = (0, 0, 255)
                    if drowsy_counter_ref['count'] > ALARM_THRESHOLD:
                        cv2.putText(frame, "!!! DROWSINESS ALERT !!!", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    drowsy_counter_ref['count'] = 0
                
                cv2.putText(frame, f"Status: {status} ({prediction:.2f})", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        break
    return frame, status

# --- 4. Video Transformer for Live Webcam ---
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.drowsy_counter_ref = {'count': 0}

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        processed_img, _ = process_frame(img, self.drowsy_counter_ref)
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# --- 5. Main Streamlit UI Layout (Unchanged) ---
with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown("<h1>Drowsiness Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<p>This AI-powered application monitors for signs of driver fatigue in real-time. <br> Choose an option below to begin monitoring.</p>", unsafe_allow_html=True)
    st.markdown("---", unsafe_allow_html=True)
    st.sidebar.title("Options")
    app_mode = st.sidebar.selectbox("Choose the app mode:", ["Live Webcam Feed", "Upload a Video"])
    if app_mode == "Live Webcam Feed":
        st.header("Live Webcam Feed")
        webrtc_streamer(key="drowsiness-detection", video_transformer_factory=VideoTransformer, media_stream_constraints={"video": True, "audio": False})
    elif app_mode == "Upload a Video":
        st.header("Upload and Analyze a Video")
        uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            cap = cv2.VideoCapture(video_path)
            st_frame = st.empty()
            drowsy_counter_ref = {'count': 0}
            total_frames, drowsy_frames = 0, 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                total_frames += 1
                processed_frame, status = process_frame(frame, drowsy_counter_ref)
                if status == "Drowsy": drowsy_frames += 1
                st_frame.image(processed_frame, channels="BGR")
            cap.release()
            os.remove(video_path)
            st.success("Video analysis complete!")
            if total_frames > 0:
                drowsiness_percentage = (drowsy_frames / total_frames) * 100
                st.metric(label="Drowsiness Percentage", value=f"{drowsiness_percentage:.2f}%")
    st.markdown("<br><hr><p style='font-size: 0.9rem;'>Developed by Rupsha Das</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
