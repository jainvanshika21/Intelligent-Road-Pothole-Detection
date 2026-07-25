import time
from pathlib import Path
import sys
import tempfile
import os
import subprocess
import cv2
import pandas as pd
import streamlit as st
import plotly.express as px

from collections import defaultdict


# ------------------ PAGE CONFIG ------------------

st.set_page_config(
    page_title="Intelligent Pothole Detection",
    page_icon="🛣️",
    layout="wide"
)


# ------------------ CUSTOM CSS ------------------

st.markdown(
    """
    <style>

    /* Main Layout */

    .block-container{
        padding-top:2rem;
        padding-left:1rem;
        padding-right:2rem;
    }

    /* Smooth Page Animation */

    .main{
        animation:fadein .4s;
    }

    @keyframes fadein{

        from{
            opacity:0;
            transform:translateY(8px);
        }

        to{
            opacity:1;
            transform:translateY(0px);
        }

    }

    /* Buttons */

    .stButton>button{

        width:100%;
        height:48px;

        border-radius:10px;

        font-weight:600;

        transition:.25s;
    }

    .stButton>button:hover{

        transform:scale(1.02);

    }

    /* Sidebar */

    section[data-testid="stSidebar"]{

        border-right:1px solid rgba(128,128,128,.25);

    }

    /* Metric Cards */

    div[data-testid="stMetric"]{

        border:1px solid rgba(128,128,128,.25);

        border-radius:12px;

        padding:16px;

        box-shadow:0 2px 6px rgba(0,0,0,.08);

    }

    /* Metric Labels */

    div[data-testid="stMetricLabel"]{

        font-weight:600;

        font-size:15px;

    }

    /* Metric Values */

    div[data-testid="stMetricValue"]{

        font-size:30px;

        font-weight:700;

    }

    /* Tabs */

    button[data-baseweb="tab"]{

        font-weight:600;

        font-size:15px;

    }

    /* Tables */

    [data-testid="stDataFrame"]{

        border-radius:10px;

        overflow:hidden;

    }

    /* Alerts */

    div[data-testid="stAlert"]{

        border-radius:12px;

    }

    </style>
    """,
    unsafe_allow_html=True
)


# ------------------ PROJECT PATH ------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ------------------ IMPORT PROJECT MODULES ------------------

from src.config import MODELS_DIR
from src.detector import PotholeDetector, draw_detections
from src.storage import save_upload
from src.video import get_video_meta, open_video, read_frame


# ------------------ K-FOLD RESULT ------------------

def calculate_kfold_average():

    base_path = PROJECT_ROOT / "runs" / "detect" / "kfold_results"

    scores = []

    for i in range(5):

        file_path = base_path / f"fold_{i}" / "results.csv"

        if file_path.exists():

            try:

                df = pd.read_csv(file_path)

                if "metrics/mAP50(B)" in df.columns:

                    map50 = df["metrics/mAP50(B)"].max()

                    scores.append(map50)

            except Exception as e:

                print(f"Error reading {file_path}: {e}")


    if len(scores) > 0:

        return scores, sum(scores) / len(scores)

    else:

        return None, None


# ------------------ SESSION STATE ------------------

def init_state():

    defaults = {

        "video_path": None,

        "processing": False,

        "detection_started": False,

        "detection_completed": False,

        "frame_idx": 0,

        "frame_skip": 3,

        "detections": [],

        "meta": None,

        "model_path": str(MODELS_DIR / "pothole.pt"),

        "conf": 0.1,

        "output_video_path": None,

        "output_video_bytes": None,

    }


    for k, v in defaults.items():

        if k not in st.session_state:

            st.session_state[k] = v


init_state()


# ------------------ UI HEADER ------------------

st.markdown(
    """
    <h1 style="
    text-align:left;
    margin-top:0px;
    margin-bottom:5px;
    ">
    🛣️ Intelligent Road Pothole Detection System
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="margin-top:0px;">
    Real-time pothole detection using YOLOv8 and Streamlit
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<div style='height:5px'></div>",
    unsafe_allow_html=True
)

st.info(
    "Upload a road video to detect potholes in real time. "
    "The application provides live detection, analytics, "
    "downloadable reports, and model evaluation."
)


# ------------------ SIDEBAR ------------------

with st.sidebar:

    st.markdown(
        """
        <h2 style="margin-bottom:10px;">
        🛣️ Pothole Detector
        </h2>
        """,
        unsafe_allow_html=True
    )


    st.subheader("📹 Upload Video")


    uploaded = st.file_uploader(
        "Upload MP4",
        type=["mp4"]
    )


    # ------------------ HANDLE UPLOAD ------------------

    if uploaded is not None:

        # Save only when a new file is uploaded
        if (
            st.session_state.video_path is None
            or st.session_state.get("uploaded_file_name") != uploaded.name
        ):

            # Reset previous results
            st.session_state.processing = False
            st.session_state.detection_started = False
            st.session_state.detection_completed = False

            st.session_state.frame_idx = 0
            st.session_state.detections = []

            st.session_state.output_video_path = None
            st.session_state.output_video_bytes = None


            saved_path = save_upload(
                uploaded.name,
                uploaded.getvalue()
            )


            st.session_state.video_path = saved_path

            st.session_state.uploaded_file_name = uploaded.name

            st.session_state.meta = get_video_meta(saved_path)


            st.success(
                f"✅ {uploaded.name} uploaded successfully!"
            )


    # ------------------ VIDEO INFORMATION ------------------

    if st.session_state.meta is not None:

        st.info(
            f"""
            🎥 Video Information

            Frames: {st.session_state.meta.total_frames}

            FPS: {st.session_state.meta.fps:.2f}
            """
        )


    # ------------------ CONTROLS ------------------

    st.subheader("⚙️ Controls")


    if st.button("▶️ Start Detection"):

        if st.session_state.video_path is None:

            st.warning(
                "Please upload a video first."
            )

        else:

            st.session_state.processing = True

            st.session_state.detection_started = True

            st.session_state.detection_completed = False

            st.session_state.frame_idx = 0

            st.session_state.detections = []

            st.session_state.output_video_path = None

            st.session_state.output_video_bytes = None

            st.rerun()


    st.subheader("🤖 Model Settings")


    st.session_state.conf = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=st.session_state.conf,
        step=0.05
    )


    st.session_state.frame_skip = st.slider(
        "Frame Skip",
        min_value=1,
        max_value=10,
        value=st.session_state.frame_skip,
        step=1,
        help="Higher values process fewer frames for faster detection."
    )


    st.markdown("---")


    st.subheader("🤖 Model Information")


    st.write(
        """
        **Model:** YOLOv8

        **Framework:** Ultralytics

        **Task:** Object Detection

        **Input:** Road Video

        **Output:** Pothole Bounding Boxes
        """
    )


    st.markdown("---")


    st.caption(
        "AI Vision System\nYOLOv8 + OpenCV"
    )


# ============================================================
# ------------------ VIDEO PROCESSING ------------------------
# ============================================================

st.subheader("📹 Processing Video")


if st.session_state.video_path is None:

    st.info("Upload video first")


else:

    # --------------------------------------------------------
    # PROCESS VIDEO ONLY WHEN DETECTION WAS STARTED
    # --------------------------------------------------------

    if (
        st.session_state.detection_started
        and st.session_state.processing
    ):

        progress_bar = st.progress(0)

        status_text = st.empty()


        try:

            # ------------------ LOAD MODEL ------------------

            detector = PotholeDetector(
                model_path=st.session_state.model_path,
                conf=st.session_state.conf,
            )


            # ------------------ OPEN INPUT VIDEO ------------------

            cap = open_video(
                st.session_state.video_path
            )


            if cap is None or not cap.isOpened():

                st.error(
                    "❌ Could not open uploaded video."
                )

                st.session_state.processing = False

                st.stop()


            # ------------------ VIDEO INFORMATION ------------------

            total_frames = st.session_state.meta.total_frames

            fps = st.session_state.meta.fps

            width = int(
                cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            )

            height = int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )


            # Safety fallback
            if fps <= 0:

                fps = 25.0


            if width <= 0 or height <= 0:

                st.error(
                    "❌ Could not read video dimensions."
                )

                cap.release()

                st.session_state.processing = False

                st.stop()


            # ------------------------------------------------
            # CREATE UNIQUE OUTPUT VIDEO
            # ------------------------------------------------

            output_dir = Path(
                tempfile.gettempdir()
            ) / "pothole_detection_outputs"

            output_dir.mkdir(
                parents=True,
                exist_ok=True
            )


            output_path = (
                output_dir
                / f"detected_output_{int(time.time())}.mp4"
            )


            # ------------------------------------------------
            # CREATE VIDEO WRITER
            # ------------------------------------------------

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )


            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (width, height)
            )


            if not writer.isOpened():

                cap.release()

                st.error(
                    "❌ Could not create output video. "
                    "The video codec may not be supported."
                )

                st.session_state.processing = False

                st.stop()


            # ------------------------------------------------
            # PROCESS VIDEO FRAME BY FRAME
            # ------------------------------------------------

            while True:

                ret, frame = read_frame(cap)


                if not ret:

                    break


                current_frame = (
                    st.session_state.frame_idx
                )


                # --------------------------------------------
                # RUN YOLO DETECTION
                # --------------------------------------------

                if (
                    current_frame
                    % st.session_state.frame_skip
                    == 0
                ):

                    detections = detector.detect(
                        frame
                    )


                    annotated = draw_detections(
                        frame,
                        detections
                    )


                    # ------------------------------
                    # STORE DETECTIONS
                    # ------------------------------

                    for d in detections:

                        timestamp = round(
                            current_frame / fps,
                            2
                        )


                        st.session_state.detections.append(
                            {
                                "timestamp (sec)": timestamp,

                                "frame": current_frame,

                                "label": d.label,

                                "confidence": round(
                                    d.confidence,
                                    2
                                ),
                            }
                        )


                else:

                    annotated = frame


                # --------------------------------------------
                # WRITE FRAME TO OUTPUT VIDEO
                # --------------------------------------------

                writer.write(
                    annotated
                )


                # --------------------------------------------
                # UPDATE FRAME COUNTER
                # --------------------------------------------

                st.session_state.frame_idx += 1


                # --------------------------------------------
                # UPDATE PROGRESS
                # --------------------------------------------

                if total_frames > 0:

                    progress = (
                        st.session_state.frame_idx
                        / total_frames
                    )

                else:

                    progress = 0


                progress_bar.progress(
                    min(progress, 1.0)
                )


                status_text.markdown(
                    f"""
                    ### 🔍 Analyzing Video with YOLOv8

                    **Progress:** {progress * 100:.1f}%

                    **Frame:** {
                        st.session_state.frame_idx
                    }/{total_frames}
                    """
                )


            # ------------------------------------------------
            # IMPORTANT:
            # RELEASE WRITER BEFORE READING VIDEO
            # ------------------------------------------------

            writer.release()

            cap.release()


            # ------------------------------------------------
            # VERIFY OUTPUT VIDEO
            # ------------------------------------------------

            if (
                not output_path.exists()
                or output_path.stat().st_size == 0
            ):

                st.error(
                    "❌ Detection completed but "
                    "the output video file is empty."
                )

                st.session_state.processing = False

                progress_bar.empty()

                status_text.empty()

                st.stop()

                        # ------------------------------------------------
            # RELEASE VIDEO WRITER AND CAPTURE
            # ------------------------------------------------

            writer.release()
            cap.release()


            # ------------------------------------------------
            # VERIFY ORIGINAL OUTPUT VIDEO
            # ------------------------------------------------

            if (
                not output_path.exists()
                or output_path.stat().st_size == 0
            ):

                st.error(
                    "❌ Detection completed, but the output "
                    "video file was not created correctly."
                )

                st.session_state.processing = False

                progress_bar.empty()
                status_text.empty()

                st.stop()


            # ------------------------------------------------
            # CONVERT VIDEO TO BROWSER-COMPATIBLE H.264
            # ------------------------------------------------

            h264_output_path = (
                output_dir
                / f"detected_output_h264_{int(time.time())}.mp4"
            )


            ffmpeg_command = [
                "ffmpeg",
                "-y",
                "-i",
                str(output_path),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(h264_output_path),
            ]


            try:

                result = subprocess.run(
                    ffmpeg_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )


                if result.returncode != 0:

                    st.error(
                        "❌ FFmpeg video conversion failed."
                    )

                    st.code(
                        result.stderr
                    )

                    st.session_state.processing = False

                    progress_bar.empty()
                    status_text.empty()

                    st.stop()


            except FileNotFoundError:

                st.error(
                    "❌ FFmpeg is not installed on the server."
                )

                st.info(
                    "Please add 'ffmpeg' to packages.txt "
                    "and redeploy your Streamlit app."
                )

                st.session_state.processing = False

                progress_bar.empty()
                status_text.empty()

                st.stop()


            # ------------------------------------------------
            # VERIFY H.264 OUTPUT VIDEO
            # ------------------------------------------------

            if (
                not h264_output_path.exists()
                or h264_output_path.stat().st_size == 0
            ):

                st.error(
                    "❌ H.264 output video was not created."
                )

                st.session_state.processing = False

                progress_bar.empty()
                status_text.empty()

                st.stop()


            # ------------------------------------------------
            # READ H.264 VIDEO AS BYTES
            # ------------------------------------------------

            with open(
                h264_output_path,
                "rb"
            ) as video_file:

                video_bytes = video_file.read()


            # ------------------------------------------------
            # SAVE VIDEO IN SESSION STATE
            # ------------------------------------------------

            st.session_state.output_video_path = (
                str(h264_output_path)
            )

            st.session_state.output_video_bytes = (
                video_bytes
            )


            # ------------------------------------------------
            # MARK PROCESSING COMPLETE
            # ------------------------------------------------

            st.session_state.processing = False

            st.session_state.detection_completed = True

            st.session_state.detection_started = False


            progress_bar.empty()
            status_text.empty()


            # ------------------------------------------------
            # DISPLAY SUCCESS MESSAGE
            # ------------------------------------------------

            st.success(
                "✅ Detection Completed!"
            )


            # ------------------------------------------------
            # DISPLAY PROCESSED VIDEO
            # ------------------------------------------------

            st.subheader(
                "🎬 Processed Video"
            )


            st.video(
                st.session_state.output_video_bytes
            )

            # ------------------------------------------------
            # MARK PROCESSING COMPLETE
            # ------------------------------------------------

            st.session_state.processing = False

            st.session_state.detection_completed = True

            st.session_state.detection_started = False


            progress_bar.empty()

            status_text.empty()


            st.success(
                "✅ Detection Completed!"
            )


            # ------------------------------------------------
            # DISPLAY PROCESSED VIDEO
            # ------------------------------------------------

            st.subheader(
                "🎬 Processed Video"
            )


            st.video(
                st.session_state.output_video_bytes
            )


        except Exception as e:

            # ----------------------------------------------
            # CLEANUP ON ERROR
            # ----------------------------------------------

            try:

                if "writer" in locals():

                    writer.release()

            except Exception:

                pass


            try:

                if "cap" in locals():

                    cap.release()

            except Exception:

                pass


            st.session_state.processing = False

            st.session_state.detection_started = False


            progress_bar.empty()

            status_text.empty()


            st.error(
                f"❌ Error while processing video: {str(e)}"
            )

# ============================================================
# ------------------ SUMMARY METRICS --------------------------
# ============================================================

if (
    st.session_state.detection_completed
    and st.session_state.detections
):

    st.success(
        "✅ Video processing completed successfully!"
    )


    df_summary = pd.DataFrame(
        st.session_state.detections
    )


    total_detections = len(
        df_summary
    )


    avg_conf = df_summary[
        "confidence"
    ].mean()


    max_conf = df_summary[
        "confidence"
    ].max()


    frames_processed = (
        st.session_state.frame_idx
    )


    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "🕳️ Total Detections",
        total_detections
    )


    c2.metric(
        "🎞️ Frames Processed",
        frames_processed
    )


    c3.metric(
        "📈 Avg Confidence",
        f"{avg_conf:.2f}"
    )


    c4.metric(
        "🎯 Highest Confidence",
        f"{max_conf:.2f}"
    )


# ============================================================
# ------------------ RESET BUTTON -----------------------------
# ============================================================

if (
    not st.session_state.processing
    and st.session_state.video_path is not None
):

    if st.button(
        "🔄 Analyze Another Video"
    ):

        # Reset everything
        st.session_state.processing = False

        st.session_state.detection_started = False

        st.session_state.detection_completed = False

        st.session_state.frame_idx = 0

        st.session_state.detections = []

        st.session_state.video_path = None

        st.session_state.meta = None

        st.session_state.output_video_path = None

        st.session_state.output_video_bytes = None

        st.session_state.uploaded_file_name = None

        st.rerun()


# ============================================================
# ------------------ TABS ------------------------------------
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "📄 Detection Report",
        "📊 Analytics Dashboard",
        "📈 Model Evaluation"
    ]
)


# ============================================================
# ------------------ REPORT TAB -------------------------------
# ============================================================

with tab1:


    st.subheader(
        "📋 Detection Results"
    )


    if st.session_state.detections:


        df = pd.DataFrame(
            st.session_state.detections
        )


        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )


        csv = df.to_csv(
            index=False
        )


        st.download_button(

            label="⬇ Download Detection Results",

            data=csv,

            file_name="detections.csv",

            mime="text/csv"

        )


    else:

        st.info(
            "Upload a video and start detection to view results."
        )


# ============================================================
# ------------------ ANALYTICS TAB ----------------------------
# ============================================================

with tab2:


    if st.session_state.detections:


        st.subheader(
            "📊 Detection Analytics"
        )


        df = pd.DataFrame(
            st.session_state.detections
        )


        frame_df = (

            df.groupby("frame")

            .size()

            .reset_index(
                name="Potholes"
            )

            .rename(
                columns={
                    "frame": "Frame"
                }
            )

        )


        st.dataframe(

            frame_df,

            use_container_width=True,

            hide_index=True

        )


        # --------------------------------------------
        # LINE CHART
        # --------------------------------------------

        st.subheader(
            "📈 Pothole Detection Trend"
        )


        fig_line = px.line(

            frame_df,

            x="Frame",

            y="Potholes",

            markers=True,

            line_shape="linear",

            title="Potholes Detected Per Frame"

        )


        fig_line.update_layout(

            xaxis_title="Frame Number",

            yaxis_title="Number of Potholes",

            template="plotly",

            height=420

        )


        fig_line.update_traces(

            marker=dict(
                size=9
            )

        )


        # --------------------------------------------
        # HISTOGRAM
        # --------------------------------------------

        st.subheader(
            "📈 Detection Confidence Distribution"
        )


        fig_hist = px.histogram(

            df,

            x="confidence",

            nbins=10,

            title="Detection Confidence Distribution"

        )


        fig_hist.update_layout(

            xaxis_title="Confidence Score",

            yaxis_title="Number of Detections",

            template="plotly",

            height=420

        )


        fig_hist.update_traces(

            opacity=0.8

        )


        # --------------------------------------------
        # DISPLAY CHARTS
        # --------------------------------------------

        col1, col2 = st.columns(2)


        with col1:

            st.plotly_chart(

                fig_line,

                use_container_width=True

            )


        with col2:

            st.plotly_chart(

                fig_hist,

                use_container_width=True

            )


    else:

        st.info(
            "No analytics available. Run detection first."
        )


# ============================================================
# ------------------ MODEL EVALUATION -------------------------
# ============================================================

with tab3:


    st.header(
        "📌 Model Evaluation (Training Performance)"
    )


    fold_scores, avg_map = (
        calculate_kfold_average()
    )


    if fold_scores is not None:


        st.subheader(
            "Fold-wise mAP50"
        )


        for i, score in enumerate(
            fold_scores
        ):

            st.write(
                f"Fold {i}: {score:.3f}"
            )


        st.metric(

            "Average mAP50 (5-Fold)",

            f"{avg_map * 100:.1f}%"

        )


        kfold_df = pd.DataFrame({

            "Fold": [
                f"Fold {i}"
                for i in range(
                    len(fold_scores)
                )
            ],

            "mAP50": fold_scores

        })


        fig_bar = px.bar(

            kfold_df,

            x="Fold",

            y="mAP50",

            text=kfold_df[
                "mAP50"
            ].round(3),

            title="5-Fold Cross Validation Performance"

        )


        fig_bar.update_traces(

            texttemplate="%{text:.3f}"

        )


        fig_bar.update_layout(

            template="plotly",

            height=450

        )


        fig_bar.update_traces(

            textposition="outside"

        )


        st.plotly_chart(

            fig_bar,

            use_container_width=True

        )


    else:

        st.warning(
            "K-Fold results not found."
        )


# ============================================================
# ------------------ FOOTER ----------------------------------
# ============================================================

st.markdown("---")


st.caption(
    "Developed using Python, Streamlit, OpenCV, and YOLOv8"
)