import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd
from datetime import datetime
import hashlib

# --- Page Config ---
st.set_page_config(
    page_title="Smart Attendance",
    page_icon="😷",
    layout="centered"
)

st.title("😷 Smart Attendance")
st.write("Snap a photo to log attendance. A mask is required for entry.")

# --- Session State ---
if "attendance_log" not in st.session_state:
    st.session_state.attendance_log = pd.DataFrame(
        columns=["Timestamp", "Status", "Access"]
    )

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_img_hash" not in st.session_state:
    st.session_state.last_img_hash = None

if "last_frame" not in st.session_state:
    st.session_state.last_frame = None

# --- Load Models ---
@st.cache_resource
def load_ai_models():
    mask_model = load_model("face_mask_detector_final.keras")
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    return mask_model, face_cascade

try:
    mask_model, face_cascade = load_ai_models()
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

# --- Camera Input ---
camera_image = st.camera_input("📷 Capture Image")

# Clear when no image
if camera_image is None:
    st.session_state.last_result = None
    st.session_state.last_frame = None
    st.session_state.last_img_hash = None

# --- Show Result Badge ---
if st.session_state.last_result == "granted":
    st.success("✅ Access Granted — Attendance Logged!")
elif st.session_state.last_result == "denied":
    st.error("🚨 Access Denied — Please Wear a Mask")

# --- SAFE IMAGE DISPLAY (FIXED) ---
try:
    if (
        st.session_state.last_frame is not None
        and isinstance(st.session_state.last_frame, Image.Image)
    ):
        st.image(st.session_state.last_frame, caption="Scan Result", use_container_width=True)
except:
    st.session_state.last_frame = None

# --- Process Image ---
if camera_image is not None:

    img_bytes = camera_image.getvalue()
    img_hash = hashlib.md5(img_bytes).hexdigest()

    if img_hash != st.session_state.last_img_hash:

        image = Image.open(camera_image)
        frame = np.array(image)

        # Convert properly
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, 1.1, 5, minSize=(60, 60)
        )

        if len(faces) == 0:
            st.warning("⚠️ No face detected")
        else:
            is_masked = True  # assume safe

            for (x, y, w, h) in faces:
                face_crop = frame_bgr[y:y+h, x:x+w]
                resized = cv2.resize(face_crop, (224, 224))
                normalized = resized / 255.0
                reshaped = np.reshape(normalized, (1, 224, 224, 3))

                pred = mask_model.predict(reshaped, verbose=0)[0]
                label_idx = np.argmax(pred)

                if label_idx != 0:
                    is_masked = False

                color = (0, 255, 0) if label_idx == 0 else (255, 0, 0)
                label = "Mask" if label_idx == 0 else "No Mask"

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # --- FIX: Convert to RGB before displaying ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.session_state.last_frame = Image.fromarray(frame_rgb)

            st.session_state.last_result = "granted" if is_masked else "denied"
            st.session_state.last_img_hash = img_hash

            # Log entry
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_entry = pd.DataFrame([{
                "Timestamp": timestamp,
                "Status": "Masked" if is_masked else "Unmasked",
                "Access": "Granted" if is_masked else "Denied"
            }])

            st.session_state.attendance_log = pd.concat(
                [st.session_state.attendance_log, new_entry],
                ignore_index=True
            )

            st.rerun()

# --- Attendance Table ---
st.subheader("📋 Attendance Log")

log = st.session_state.attendance_log

if not log.empty:
    st.dataframe(log, use_container_width=True)

    csv = log.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download CSV",
        csv,
        "attendance.csv",
        "text/csv"
    )
else:
    st.info("No records yet.")