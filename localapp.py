
# localapp.py
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import pyautogui
import time
import operator

import os

# ---------------------------
# Model loading (cached)
# ---------------------------
@st.cache_resource
def load_model():
    import tensorflow as tf
    with open("gesture-model.json", "r") as json_file:
        model_json = json_file.read()
    try:
        loaded_model = tf.keras.models.model_from_json(model_json)
    except TypeError:
        loaded_model = tf.keras.models.model_from_json(
            model_json, custom_objects={'Sequential': tf.keras.models.Sequential}
        )
    loaded_model.load_weights("gesture-model.h5")
    return loaded_model



# Categories
categories = {
    0: 'palm',
    1: 'fist',
    2: 'thumbs-up',
    3: 'thumbs-down',
    4: 'index-right',
    5: 'index-left',
    6: 'no-gesture'
}

# ---------------------------
# Global CSS for nicer look
# ---------------------------
def inject_global_css():
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    /* Page background gradient + subtle animation */
    .stApp {
      font-family: 'Poppins', sans-serif;
      background: radial-gradient(1200px 600px at 10% 20%, rgba(59,130,246,0.10), transparent 10%),
                  radial-gradient(900px 500px at 90% 80%, rgba(16,185,129,0.08), transparent 10%),
                  linear-gradient(180deg, #071029 0%, #0b1220 100%);
      background-attachment: fixed;
      color: #e6eef7;
    }

    /* Glass card style for columns and side panels */
    .glass {
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.06);
      backdrop-filter: blur(6px);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 6px 20px rgba(2,6,23,0.5);
    }

    /* Buttons */
    .stButton>button {
      border-radius: 10px;
      padding: 8px 14px;
      font-weight: 600;
      box-shadow: 0 8px 18px rgba(2,6,23,0.35);
    }

    /* Small badge */
    .badge {
      display:inline-block;
      padding:6px 10px;
      border-radius:999px;
      font-weight:700;
    }

    /* Gesture / action colored badges */
    .g-badge { background: linear-gradient(90deg,#60a5fa,#3b82f6); color:white; }
    .a-badge { background: linear-gradient(90deg,#34d399,#10b981); color:white; }
    .warn-badge { background: linear-gradient(90deg,#fb7185,#f97316); color:white; }

    /* Title tweaks */
    .main-title {
      background: linear-gradient(to right, #60a5fa, #2dd4bf);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      font-weight: 800;
      margin-bottom: 4px;
    }
    .subtitle {
      color: #FFFFFF;
      margin-top: 2px;
      font-size: 13px;
    }

    /* Small utility spacing */
    .small-space { margin-top:8px; }
    .large-space { margin-top:18px; }

    /* Animated glowing blob behind header (bare-bones) */
    .blob {
      position: absolute;
      width: 220px;
      height: 220px;
      filter: blur(52px);
      opacity: 0.9;
      border-radius: 50%;
      z-index: -1;
      transform: translate(-30px, -10px);
      background: radial-gradient(circle at 30% 30%, rgba(59,130,246,0.25), transparent 40%),
                  radial-gradient(circle at 70% 70%, rgba(16,185,129,0.18), transparent 40%);
    }

    /* make streamlit markdown html blocks look good */
    .header-area { position: relative; overflow: hidden; padding: 18px; border-radius: 12px; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ---------------------------
# Animated header using GSAP (in an iframe via components.html)
# ---------------------------
def render_animated_header():
    header_html = """
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/gsap.min.js"></script>

    <div style="position:relative;padding:14px;border-radius:12px;" class="header-area glass">
      <div class="blob" id="blob"></div>
      <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;text-align:center;">
        <div>
          <div class="main-title" id="mainTitle" style="font-size:32px; letter-spacing:1px; background: linear-gradient(to right, #60a5fa, #2dd4bf); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800;">Hand Gesture Media Controller</div>
        </div>
        <div>
          <div class="badge" id="liveBadge" style="background:linear-gradient(90deg,#06b6d4,#3b82f6);color:white;font-size:14px;">LIVE DEMO</div>
        </div>
      </div>
      <div style="display:flex;justify-content:center;gap:12px;margin-top:16px;align-items:center;">
        <button id="ctaStart" class="px-6 py-3 rounded-lg" style="background:linear-gradient(90deg,#7c3aed,#06b6d4);color:white;border:none;font-weight:700;cursor:pointer;font-size:16px;box-shadow:0 4px 14px rgba(0,0,0,0.25);">Start Gesture Mode</button>
      </div>
      <div style="text-align:center;color:#cfe8ff;font-size:14px;margin-top:12px;">Tip: Place your hand inside the blue rectangle on the feed.</div>
    </div>

    <script>
      // subtle entrance animations
      gsap.from("#mainTitle", { y: -12, opacity: 0, duration: 0.9, ease: "power3.out" });
      gsap.from("#mainSub", { y: -6, opacity: 0, duration: 0.8, delay: 0.12 });
      gsap.from("#liveBadge", { scale: 0.9, opacity: 0, duration: 0.6, delay: 0.2 });
      gsap.from("#ctaStart", { y: 6, opacity: 0, duration: 0.7, delay: 0.28, ease: "back.out(1.2)" });

      // gentle floating animation for the blob background
      gsap.to("#blob", { x: 40, y: 10, duration: 6, repeat: -1, yoyo: true, ease: "sine.inOut" });

      // connect the header CTA to the Streamlit button (best-effort: finds the first Start-like button)
      document.getElementById("ctaStart").addEventListener("click", () => {
        const allButtons = Array.from(document.querySelectorAll("button"));
        const btn = allButtons.find(b => /Start Web Camera|Start Web Camera \\/ Video|Start Web Camera \\/ Video/i.test(b.innerText) || /Start Web Camera|Start Web Camera \\/ Video|Start Web Camera/i.test(b.innerText));
        if (btn) btn.click();
      });
    </script>
    """
    # set a little extra height to accommodate visuals
    components.html(header_html, height=170, scrolling=False)

# ---------------------------
# Helper: ensure output folder
# ---------------------------
def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# ---------------------------
# App main
# ---------------------------
def main():
    st.set_page_config(page_title="Gesture Media Controller", layout="wide")
    inject_global_css()
    render_animated_header()

    # Sidebar: Settings & Demo options
    with st.sidebar:
        st.header("Settings")
        cam_index = st.number_input("Camera index (0 is default)", min_value=0, max_value=10, value=0, step=1)
        pred_interval = st.slider("Prediction interval (frames)", min_value=1, max_value=20, value=8)
        conf_thresh = st.slider("Confidence threshold", min_value=0.1, max_value=1.0, value=0.75, step=0.01)
        buffer_size = st.slider("Gesture buffer size", min_value=1, max_value=6, value=2)
        cooldown_time = st.slider("Cooldown (seconds) for non-volume gestures", min_value=0.0, max_value=5.0, value=4.0, step=0.1)

        st.markdown("---")
        st.header("Demo / Video")
        use_sample = st.checkbox("Use sample video instead of webcam", value=False)
        uploaded_file = None
        if use_sample:
            uploaded_file = st.file_uploader("Upload video (mp4) or leave blank to use local sample", type=["mp4", "avi", "mov"])

        record_demo = st.checkbox("Record demo output to demo_output.mp4", value=False)

        st.markdown("---")
        st.markdown("Made By:")
        st.markdown("- Rishi Raj Sharma  \n- S Madan  \n- MD Faiz  \n- Tejaswini")

    # Main area: pages
    st.markdown("<div class='glass' style='padding:12px;'>", unsafe_allow_html=True)
    st.markdown("## Gesture Control")
    st.markdown("Place your hand inside the blue rectangle on the camera feed. Use the legend to test gestures.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='glass small-space' style='padding:12px;'>
    <strong>Gestures legend</strong><br/>
    - ✋ <strong>palm</strong> : Play / Pause &nbsp;&nbsp;
    - ✊ <strong>fist</strong> : Mute &nbsp;&nbsp;
    - 👍 <strong>thumbs-up</strong> : Volume Up &nbsp;&nbsp;
    - 👎 <strong>thumbs-down</strong> : Volume Down &nbsp;&nbsp;
    - 👉 <strong>index-right</strong> : Forward &nbsp;&nbsp;
    - 👈 <strong>index-left</strong> : Rewind
    </div>
    """, unsafe_allow_html=True)

    # Controls and placeholders
    col_left, col_right = st.columns([2, 1])
    with col_left:
        start_btn = st.button("Start Web Camera / Video")
        stop_btn = st.button("Stop (reload page to fully stop)")
        roi_placeholder = st.image(np.zeros((120, 120)), caption="Processed ROI (thresholded)")
        feed_placeholder = st.image(np.zeros((480, 640, 3), dtype=np.uint8), caption="Webcam / Video Feed")

    with col_right:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("### Live Status")
        gesture_badge = st.empty()
        action_badge = st.empty()
        conf_box = st.empty()
        progress_place = st.empty()
        cooldown_box = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass small-space'>", unsafe_allow_html=True)
        st.markdown("*Tips*")
        st.markdown("- Use good, front-facing light.")
        st.markdown("- Keep background simple inside ROI.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Start loop on button
    if start_btn:
        # Input source: webcam or uploaded/sample video
        if use_sample:
            if uploaded_file is not None:
                # save uploaded file to a temp location and use it
                temp_path = "temp_uploaded_demo.mp4"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
                source = temp_path
            else:
                # fallback to local 'sample_video.mp4' if present, otherwise error
                if os.path.exists("sample_video.mp4"):
                    source = "sample_video.mp4"
                else:
                    st.error("No sample video found. Upload a video or place 'sample_video.mp4' in the app folder.")
                    source = None
        else:
            source = cam_index  # webcam index

        if source is not None:
            with st.spinner("Loading model..."):
                loaded_model = load_model()
            
            # Open capture
            camera = cv2.VideoCapture(source)
            # if webcam, set resolution; if file, keep default
            if not use_sample:
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # Setup recording writer if requested
            writer = None
            if record_demo:
                ensure_dir("outputs/")
                out_path = "outputs/demo_output.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(out_path, fourcc, 20.0, (640, 480))

            # Prediction loop variables
            frame_count = 0
            PREDICTION_INTERVAL = pred_interval
            CONFIDENCE_THRESHOLD = conf_thresh
            COOLDOWN_TIME = cooldown_time
            last_action_time = 0
            action_triggered = False
            current_gesture = 'no-gesture'
            gesture_buffer = []
            BUFFER_SIZE = buffer_size
            displayed_action = "Waiting..."

            # small animation: reveal the live status with JS (best-effort)
            try:
                while True:
                    ret, frame = camera.read()
                    if not ret:
                        st.warning("Stream ended or cannot read from source.")
                        break

                    frame = cv2.flip(frame, 1) if not use_sample else frame

                    # region-of-interest (right-top)
                    x1 = int(0.5 * frame.shape[1])
                    y1 = 10
                    x2 = frame.shape[1] - 10
                    y2 = int(0.5 * frame.shape[1])
                    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255, 0, 0), 3)

                    roi = frame[y1:y2, x1:x2]
                    roi_resized = cv2.resize(roi, (120, 120))
                    gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
                    _, test_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

                    # Show frames in Streamlit
                    roi_placeholder.image(test_image, clamp=False)
                    feed_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    frame_count += 1
                    action = "NO ACTION"
                    confidence = 0.0
                    current_time = time.time()

                    if frame_count % PREDICTION_INTERVAL == 0:
                        result = loaded_model.predict(test_image.reshape(1, 120, 120, 1), verbose=0)
                        prediction = {
                            'palm': float(result[0][0]),
                            'fist': float(result[0][1]),
                            'thumbs-up': float(result[0][2]),
                            'thumbs-down': float(result[0][3]),
                            'index-right': float(result[0][4]),
                            'index-left': float(result[0][5]),
                            'no-gesture': float(result[0][6])
                        }

                        pred_sorted = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
                        top_pred = pred_sorted[0][0]
                        confidence = pred_sorted[0][1]

                        gesture_buffer.append(top_pred)
                        if len(gesture_buffer) > BUFFER_SIZE:
                            gesture_buffer.pop(0)
                        if len(gesture_buffer) >= BUFFER_SIZE:
                            stable_gesture = max(set(gesture_buffer), key=gesture_buffer.count)
                        else:
                            stable_gesture = top_pred

                        if confidence >= CONFIDENCE_THRESHOLD:
                            current_gesture = stable_gesture
                        else:
                            current_gesture = 'no-gesture'
                            gesture_buffer.clear()

                        cooldown_active = action_triggered and (current_time - last_action_time) < COOLDOWN_TIME

                        # Actions mapping
                        if current_gesture == 'thumbs-up':
                            action = "VOLUME UP"
                            pyautogui.press('volumeup')
                        elif current_gesture == 'thumbs-down':
                            action = "VOLUME DOWN"
                            pyautogui.press('volumedown')
                        elif current_gesture != 'no-gesture' and not cooldown_active:
                            if current_gesture == 'palm':
                                action = "PLAY/PAUSE"
                                pyautogui.press('playpause')
                                last_action_time = current_time
                                action_triggered = True
                            elif current_gesture == 'fist':
                                action = "MUTE"
                                pyautogui.press('volumemute')
                                last_action_time = current_time
                                action_triggered = True
                            elif current_gesture == 'index-right':
                                action = "FORWARD"
                                pyautogui.press('nexttrack')
                                last_action_time = current_time
                                action_triggered = True
                            elif current_gesture == 'index-left':
                                action = "REWIND"
                                pyautogui.press('prevtrack')
                                last_action_time = current_time
                                action_triggered = True

                    # Cooldown display
                    cooldown_text = "Ready"
                    if action_triggered:
                        t_since = current_time - last_action_time
                        if t_since < COOLDOWN_TIME:
                            cooldown_text = f"Cooldown: {COOLDOWN_TIME - t_since:.1f}s"
                        else:
                            action_triggered = False
                            cooldown_text = "Ready"

                    # Update status widgets with colored badges
                    gesture_badge.markdown(
                        f"<div class='badge g-badge'>Gesture: <strong style='margin-left:8px'>{current_gesture}</strong></div>",
                        unsafe_allow_html=True
                    )
                    
                    # Update displayed action only if a new action is triggered
                    if action != "NO ACTION":
                        displayed_action = action

                    # action badge color: green for actions, warn for NO ACTION (or waiting)
                    if displayed_action != "Waiting...":
                        action_badge.markdown(
                            f"<div class='badge a-badge'>Last Action: <strong style='margin-left:8px'>{displayed_action}</strong></div>",
                            unsafe_allow_html=True
                        )
                    else:
                        action_badge.markdown(
                            f"<div class='badge warn-badge'>Action: <strong style='margin-left:8px'>{displayed_action}</strong></div>",
                            unsafe_allow_html=True
                        )

                    # Confidence UI: progress bar + numeric
                    pct = int(confidence * 100)
                    conf_box.markdown(f"*Confidence:* {confidence:.2f}")
                    progress_place.progress(pct)

                    cooldown_box.markdown(f"*Status:* {cooldown_text}")

                    # Optionally record output (draw simple overlay then write)
                    if writer is not None:
                        out_frame = frame.copy()
                        cv2.putText(out_frame, f'Gesture: {current_gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                        cv2.putText(out_frame, f'Action: {action}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                        out_frame_resized = cv2.resize(out_frame, (640, 480))
                        writer.write(out_frame_resized)

                    # tiny sleep to be gentle on CPU
                    time.sleep(0.02)

            except Exception as e:
                st.error(f"Stopped due to error: {e}")
            finally:
                camera.release()
                if writer is not None:
                    writer.release()
                cv2.destroyAllWindows()
                st.success("Camera / Video stopped. If you recorded, check outputs/demo_output.mp4")

    # stop button note
    if stop_btn:
        st.warning("To fully stop the video loop, please stop the Streamlit server (Ctrl+C) or refresh the page.")


if __name__ == "__main__":
    main()
