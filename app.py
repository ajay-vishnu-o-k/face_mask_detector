import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd
from datetime import datetime
import hashlib

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Smart Attendance",
    page_icon="😷",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    :root {
        --bg-primary:   #0d0f14;
        --bg-secondary: #151820;
        --bg-card:      #1c2030;
        --accent-teal:  #00e5c3;
        --accent-blue:  #4f8ef7;
        --text-primary: #e8eaf2;
        --text-muted:   #7a8099;
        --green:        #00c48c;
        --red:          #ff4d6d;
        --border:       rgba(255,255,255,0.07);
        --radius:       14px;
    }

    html { -webkit-text-size-adjust: 100%; touch-action: manipulation; }

    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main .block-container {
        background-color: var(--bg-primary) !important;
        font-family: 'DM Sans', sans-serif;
        color: var(--text-primary);
    }

    .main .block-container {
        padding: 1rem 1rem 2rem !important;
        max-width: 100% !important;
    }

    @media (min-width: 768px) {
        .main .block-container {
            padding: 2rem 3rem 3rem !important;
            max-width: 720px !important;
            margin: 0 auto !important;
        }
    }

    [data-testid="stHeader"] { background-color: var(--bg-primary) !important; }

    @media (max-width: 767px) {
        [data-testid="stSidebarNav"],
        [data-testid="collapsedControl"] { display: none !important; }
    }

    h1 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 800 !important;
        font-size: clamp(1.5rem, 5vw, 2.2rem) !important;
        background: linear-gradient(135deg, var(--accent-teal), var(--accent-blue));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }

    h2, h3 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: clamp(1rem, 3.5vw, 1.4rem) !important;
        color: var(--text-primary) !important;
    }

    p, .stMarkdown p {
        color: var(--text-muted) !important;
        font-weight: 300;
        font-size: clamp(0.85rem, 2.5vw, 0.97rem);
        line-height: 1.6;
    }

    [data-testid="stCameraInput"] > div {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 0.75rem;
        width: 100% !important;
        box-sizing: border-box;
    }

    [data-testid="stCameraInput"] video,
    [data-testid="stCameraInput"] canvas,
    [data-testid="stCameraInput"] img {
        width: 100% !important;
        height: auto !important;
        border-radius: 10px;
    }

    [data-testid="stCameraInput"] label {
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        color: var(--accent-teal) !important;
        font-size: clamp(0.85rem, 2.5vw, 1rem);
    }

    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--accent-teal), var(--accent-blue)) !important;
        color: #0d0f14 !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 1.6rem !important;
        min-height: 48px !important;
        font-size: clamp(0.85rem, 2.5vw, 1rem) !important;
        width: 100% !important;
        letter-spacing: 0.3px;
        transition: opacity 0.2s ease;
        cursor: pointer;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover { opacity: 0.85; }

    div[data-testid="stSuccess"] {
        background-color: rgba(0,196,140,0.12) !important;
        border-left: 4px solid var(--green) !important;
        border-radius: 10px !important;
    }
    div[data-testid="stSuccess"] p,
    div[data-testid="stSuccess"] span { color: var(--green) !important; }

    div[data-testid="stError"] {
        background-color: rgba(255,77,109,0.12) !important;
        border-left: 4px solid var(--red) !important;
        border-radius: 10px !important;
    }
    div[data-testid="stError"] p,
    div[data-testid="stError"] span { color: var(--red) !important; }

    div[data-testid="stWarning"] {
        background-color: rgba(255,183,77,0.10) !important;
        border-left: 4px solid #ffb74d !important;
        border-radius: 10px !important;
    }

    div[data-testid="stInfo"] {
        background-color: rgba(79,142,247,0.10) !important;
        border-left: 4px solid var(--accent-blue) !important;
        border-radius: 10px !important;
    }

    /* suppress Streamlit deprecation warnings */
    div[data-testid="stDeprecationWarning"],
    .stAlert[kind="warning"] { display: none !important; }

    [data-testid="stDataFrame"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch;
        width: 100% !important;
        display: block !important;
    }
    [data-testid="stDataFrame"] table {
        min-width: 400px;
        font-size: clamp(0.75rem, 2vw, 0.9rem);
    }

    hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

    [data-testid="stImage"] p {
        color: var(--text-muted) !important;
        font-size: 0.82rem;
        text-align: center;
    }

    .result-badge {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.6rem;
        padding: 1rem 1.25rem;
        border-radius: var(--radius);
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: clamp(1rem, 3.5vw, 1.25rem);
        margin: 1rem 0;
        letter-spacing: 0.2px;
    }
    .result-badge.granted {
        background: rgba(0,196,140,0.15);
        border: 1.5px solid var(--green);
        color: var(--green);
    }
    .result-badge.denied {
        background: rgba(255,77,109,0.15);
        border: 1.5px solid var(--red);
        color: var(--red);
    }
</style>
""", unsafe_allow_html=True)


# ── 2. Header ──────────────────────────────────────────────────────────────────
st.title("😷 Smart Attendance")
st.write("Snap a photo to log attendance. A mask is required for entry.")


# ── 3. Session state ───────────────────────────────────────────────────────────
if "attendance_log" not in st.session_state:
    st.session_state.attendance_log = pd.DataFrame(
        columns=["Timestamp", "Status", "Access"]
    )
if "last_result" not in st.session_state:
    st.session_state.last_result = None       # "granted" | "denied" | None

# FIX: track the hash of the last processed image to prevent duplicate logging
if "last_img_hash" not in st.session_state:
    st.session_state.last_img_hash = None

# FIX: store the annotated frame to redisplay after rerun without re-processing
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None


# ── 4. Load models ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_ai_models():
    mask_model   = load_model("face_mask_detector_final.keras")
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    return mask_model, face_cascade

try:
    mask_model, face_cascade = load_ai_models()
except Exception as e:
    st.error(f"Model load failed — make sure **face_mask_detector_final.keras** is present.\n\n`{e}`")
    st.stop()


# ── 6. Camera input ────────────────────────────────────────────────────────────
camera_image = st.camera_input("📷  Look at the camera and snap a photo")

# ── FIX: clear stored results as soon as user clears the photo ─────────────────
if camera_image is None:
    st.session_state.last_result   = None
    st.session_state.last_frame    = None
    st.session_state.last_img_hash = None

# ── 5. Result badge + last frame (only shown while a photo is present) ─────────
if st.session_state.last_result == "granted":
    st.markdown(
        '<div class="result-badge granted">✅ &nbsp;Access Granted — Attendance Logged!</div>',
        unsafe_allow_html=True,
    )
elif st.session_state.last_result == "denied":
    st.markdown(
        '<div class="result-badge denied">🚨 &nbsp;Access Denied — Please Wear a Mask</div>',
        unsafe_allow_html=True,
    )

if st.session_state.last_frame is not None:
    st.image(st.session_state.last_frame, caption="Scan Result", use_container_width=True)

if camera_image is not None:
    # ── FIX: hash the image bytes — only process if it's a NEW photo ──
    img_bytes = camera_image.getvalue()
    img_hash  = hashlib.md5(img_bytes).hexdigest()

    if img_hash != st.session_state.last_img_hash:
        # New photo — run detection
        image     = Image.open(camera_image)
        frame     = np.array(image)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) == 0:
            st.warning("⚠️ No face detected — step closer and try again.")
        else:
            is_masked = False

            for (x, y, w, h) in faces:
                face_crop  = frame_bgr[y:y+h, x:x+w]
                resized    = cv2.resize(face_crop, (224, 224))
                normalized = resized / 255.0
                reshaped   = np.reshape(normalized, (1, 224, 224, 3))

                pred      = mask_model.predict(reshaped, verbose=0)[0]
                label_idx = np.argmax(pred)

                # ── FIX: check both class orderings from the model ──
                # Print to logs so you can verify: 0=Masked or 0=Unmasked
                # Based on your notebook: class_indices shows WithMask=0, WithoutMask=1
                is_masked = (label_idx == 0)

                # RGB colors (frame is RGB from PIL)
                color_rgb = (0, 196, 140) if is_masked else (255, 77, 109)
                label_txt = "Masked - Granted" if is_masked else "No Mask - Denied"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color_rgb, 3)
                bg_y1 = max(y - 36, 0)
                cv2.rectangle(frame, (x, bg_y1), (x + w, y), color_rgb, cv2.FILLED)
                cv2.putText(
                    frame, label_txt, (x + 6, y - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                )

                # Log ONE entry per photo
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                new_entry = pd.DataFrame([{
                    "Timestamp": timestamp,
                    "Status":    "Masked"  if is_masked else "Unmasked",
                    "Access":    "Granted" if is_masked else "Denied",
                }])
                st.session_state.attendance_log = pd.concat(
                    [st.session_state.attendance_log, new_entry], ignore_index=True
                )

            # Persist state for after rerun
            st.session_state.last_img_hash = img_hash
            st.session_state.last_result   = "granted" if is_masked else "denied"
            st.session_state.last_frame    = frame

            st.rerun()


# ── 7. Attendance log ──────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Live Attendance Database")

log = st.session_state.attendance_log

if not log.empty:
    def highlight_access(val):
        if val == "Granted":
            return "background-color:rgba(0,196,140,0.15);color:#00c48c;font-weight:600;"
        if val == "Denied":
            return "background-color:rgba(255,77,109,0.15);color:#ff4d6d;font-weight:600;"
        return ""

    styled_df = log.style.applymap(highlight_access, subset=["Access"])
    st.dataframe(styled_df, use_container_width=True)

    csv = log.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇  Download Attendance CSV",
        data=csv,
        file_name="attendance_log.csv",
        mime="text/csv",
    )
else:
    st.info("ℹ️ No attendance records yet today.")