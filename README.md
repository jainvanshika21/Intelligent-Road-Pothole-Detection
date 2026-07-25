# 🛣️ Intelligent Road Pothole Detection System

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-success)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit)

> An end-to-end Computer Vision application for detecting road potholes from uploaded road videos using a custom-trained YOLOv8 model, OpenCV, and Streamlit, with browser-compatible processed video playback, interactive analytics, detection reports, CSV export, and model performance evaluation.

---

# 🌐 Live Demo

🚀 **[Open the Live Pothole Detection App](https://jainvanshika21-intelligent-road-pothole-appstreamlit-app-2mc2qo.streamlit.app/)**

The application is deployed using Streamlit and allows users to upload road videos, run AI-based pothole detection, view bounding-box predictions, explore interactive analytics, review detection reports, and download results as CSV.

> **Note:** The application is designed for uploaded road videos containing road surfaces and potential potholes.

---

# 📌 Project Overview

The **Intelligent Road Pothole Detection System** is an AI-powered Computer Vision application designed to automatically detect potholes from road videos.

The system uses a custom-trained **YOLOv8 object detection model** to identify potholes in video frames. **OpenCV** is used for video reading, frame processing, annotation, and video generation, while **Streamlit** provides an interactive web-based interface.

After processing, the system generates a browser-compatible processed video with pothole bounding boxes and confidence scores. Users can also explore detection statistics, view detailed detection reports, download results as CSV, and evaluate model performance using 5-Fold Cross Validation.

---

# ⭐ Project Highlights

* 🚀 AI-powered pothole detection using YOLOv8
* 🎥 Road video processing using OpenCV
* 📦 Bounding box visualization with confidence scores
* ▶️ Browser-compatible processed video playback
* 📊 Interactive analytics dashboard
* 📄 Downloadable CSV detection reports
* 🎯 Adjustable confidence threshold
* ⏭️ Adjustable frame skip for video processing
* 🧠 5-Fold Cross Validation for model evaluation
* 📈 Fold-wise mAP@50 performance visualization
* 🌐 Deployed as a live Streamlit web application
* 💻 Interactive and user-friendly interface

---

# 🎯 Project Objective

The objective of this project is to develop an automated Computer Vision system capable of identifying potholes from road videos using Deep Learning and Object Detection.

The system aims to:

* Reduce the need for manual pothole inspection
* Automatically identify potholes in road footage
* Provide visual detection results
* Generate structured detection reports
* Analyze pothole detection patterns
* Provide confidence-based detection insights
* Evaluate model performance using validation techniques

---

# 🚀 Features

## 🎥 Video Processing

* Upload `.mp4` road videos
* Extract and process video frames using OpenCV
* Process video frames using YOLOv8
* Generate annotated output videos
* Display processed videos directly in the browser
* Use H.264 encoding for browser-compatible playback

## 🤖 AI-Based Detection

* Custom-trained YOLOv8 object detection model
* Automatic pothole detection from video frames
* Adjustable confidence threshold
* Bounding box visualization
* Confidence score display

## 📊 Interactive Analytics

* Total pothole detections
* Number of processed frames
* Average detection confidence
* Highest detection confidence
* Frame-wise detection trends
* Detection confidence distribution
* Interactive visualizations

## 📄 Detection Reports

* Generate structured detection results
* Display detection timestamps
* Display video frame numbers
* Display confidence scores
* Export detection data as CSV
* Download reports for further analysis

## 🧠 Model Evaluation

* 5-Fold Cross Validation
* Fold-wise mAP@50 performance
* Average mAP@50
* Performance comparison across folds
* Interactive model performance visualization

---

# 🧠 Model Details

The project uses a custom-trained **YOLOv8 object detection model** for pothole detection.

### Detection Class

* `Pothole`

### Model Output

For each detected pothole, the system provides:

* Bounding box visualization
* Detection label
* Confidence score
* Detection timestamp
* Video frame number

---

# 🤖 Model Evaluation

The project includes model evaluation using **5-Fold Cross Validation**.

The evaluation dashboard presents:

* Fold-wise mAP@50
* Average mAP@50
* Performance comparison across folds

The results are visualized using an interactive chart to compare model performance across individual folds.

---

# 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Computer Vision | OpenCV |
| Deep Learning | Ultralytics YOLOv8 |
| Web Framework | Streamlit |
| Data Analysis | Pandas, NumPy |
| Visualization | Plotly, Matplotlib |
| Machine Learning | Scikit-learn |
| Video Encoding | FFmpeg / H.264 |

---

# 🎥 Demo Video

The project demonstration showcases:

* Road video upload
* YOLOv8 pothole detection
* Bounding box predictions
* Processed video playback
* Detection summary
* Detection report
* Interactive analytics
* Model performance evaluation

▶️ **[Watch the Pothole Detection Demo](assets/Pothole_Detection_Demo.mp4)**

---

# 📂 Project Structure

```text
Intelligent-Road-Pothole-Detection/
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   ├── config.py
│   ├── detector.py
│   ├── storage.py
│   └── video.py
│
├── scripts/
│
├── models/
│   └── pothole.pt
│
├── data/
│
├── runs/
│   └── detect/
│       └── kfold_results/
│
├── assets/
│   ├── 01_Home_Dashboard.png
│   ├── 02_Video_Upload.png
│   ├── 03_Live_Pothole_Detection.png
│   ├── 04_Detection_Report.png
│   ├── 05_Analytics_Dashboard.png
│   ├── 06_Model_Performance.png
│   └── Pothole_Detection_Demo.mp4
│
├── requirements.txt
├── packages.txt
├── README.md
└── .gitignore
```

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/jainvanshika21/Intelligent-Road-Pothole-Detection.git
```

## 2. Move into the Project Directory

```bash
cd Intelligent-Road-Pothole-Detection
```

## 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## 4. Install FFmpeg

FFmpeg is required for converting processed videos into browser-compatible H.264 format.

### Windows

Install FFmpeg and make sure it is available in your system PATH.

### Streamlit Cloud

The project includes a `packages.txt` file containing:

```text
ffmpeg
```

This allows Streamlit Cloud to install FFmpeg during deployment.

## 5. Launch the Application

```bash
streamlit run app/streamlit_app.py
```

---

# 🎥 How to Use

1. Open the deployed application or run it locally.
2. Upload a road video in `.mp4` format.
3. Adjust the confidence threshold if required.
4. Adjust the frame skip value if required.
5. Start the pothole detection process.
6. Wait for the video processing to complete.
7. View the processed video with pothole bounding boxes.
8. Review detection summary metrics.
9. Explore the Detection Report.
10. Explore the Analytics Dashboard.
11. Download detection results as a CSV file.
12. View model evaluation results.

---

# 🔄 System Workflow

```text
                 Road Video
                      │
                      ▼
                Upload Video
                      │
                      ▼
           Frame Extraction (OpenCV)
                      │
                      ▼
              YOLOv8 Detection
                      │
                      ▼
          Bounding Box Visualization
                      │
                      ▼
        Temporary Processed Video
                      │
                      ▼
        FFmpeg H.264 Video Conversion
                      │
                      ▼
        Browser-Compatible Video Output
                      │
                      ▼
             Detection Results
                │           │
                │           │
                ▼           ▼
            Analytics    CSV Report
                │
                ▼
          Model Evaluation
```

---

# 📊 Analytics Dashboard

The application provides interactive analytics including:

* 📌 Total number of pothole detections
* 🎞️ Number of frames processed
* 📈 Average detection confidence
* 🎯 Highest detection confidence
* 📉 Frame-wise pothole detection trends
* 📊 Detection confidence distribution
* 📋 Detection results table
* 📄 Downloadable CSV report

These visualizations help analyze detection patterns and understand model predictions across the processed road video.

---

# 📸 Application Preview

## 🏠 Home Dashboard

![Home Dashboard](assets/01_Home_Dashboard.png)

---

## 🎥 Video Upload

![Video Upload](assets/02_Video_Upload.png)

---

## 🚧 Live Pothole Detection

![Live Detection](assets/03_Live_Pothole_Detection.png)

---

## 📋 Detection Report

![Detection Report](assets/04_Detection_Report.png)

---

## 📊 Analytics Dashboard

![Analytics Dashboard](assets/05_Analytics_Dashboard.png)

---

## 🤖 Model Performance

![Model Performance](assets/06_Model_Performance.png)

---

# 📦 Outputs

The system generates:

* ✅ Pothole detection results
* ✅ Bounding box visualizations
* ✅ Confidence scores
* ✅ Detection timestamps
* ✅ Video frame numbers
* ✅ Detection summary metrics
* ✅ Browser-compatible processed video
* ✅ Downloadable CSV reports
* ✅ Interactive analytics
* ✅ Model evaluation results

---

# ⚠️ Limitations

* Detection performance depends on video quality, lighting, camera angle, and road conditions.
* Extremely blurry or low-resolution videos may reduce detection accuracy.
* The model may produce false positives on road surfaces with patterns visually similar to potholes.
* The application is designed for road videos and may not provide meaningful results for unrelated non-road videos.
* The current system focuses on pothole detection and does not classify pothole severity.
* GPS-based pothole location tracking is not currently implemented.
* The application is designed for uploaded road videos rather than continuous live camera feeds.

---

# 🔮 Future Enhancements

* 📍 GPS-based pothole mapping
* 🗺️ Interactive pothole mapping dashboard
* ☁️ Cloud deployment improvements
* 📱 Mobile application
* 🎥 Live camera-based detection
* 🚧 Pothole severity classification
* 🌍 Smart City infrastructure integration
* 📍 Automatic location tagging
* 📊 Historical pothole tracking
* 🧠 Improved detection using larger and more diverse datasets

---

# 📚 Dependencies

The main Python dependencies used in the project are:

```text
streamlit>=1.30
opencv-python-headless==4.10.0.84
numpy>=1.24
pandas>=2.0
ultralytics>=8.0
scikit-learn>=1.3
matplotlib>=3.7
PyYAML>=6.0
plotly>=5.0
```

Install all Python dependencies using:

```bash
pip install -r requirements.txt
```

FFmpeg is installed separately for video conversion and browser-compatible playback.

For Streamlit Cloud deployment, the required system package is specified in:

```text
packages.txt
```

with:

```text
ffmpeg
```

---

# ☁️ Deployment

The application is deployed using **Streamlit Cloud**.

The deployment configuration uses:

* `requirements.txt` for Python dependencies
* `packages.txt` for the FFmpeg system dependency
* `app/streamlit_app.py` as the Streamlit application entry point

The live application can be accessed here:

🚀 **[Open Live Pothole Detection App](https://jainvanshika21-intelligent-road-pothole-appstreamlit-app-2mc2qo.streamlit.app/)**

---

# 👩‍💻 Author

## Vanshika Jain

**MCA Student | Computer Vision & Data Analytics Enthusiast**

* 🔗 [GitHub](https://github.com/jainvanshika21)
* 🔗 [LinkedIn](https://www.linkedin.com/in/vanshika-jain-17007128a/)

### Skills

`Python` • `YOLOv8` • `OpenCV` • `Streamlit` • `Computer Vision` • `Pandas` • `NumPy` • `Plotly` • `Scikit-learn` • `Machine Learning` • `Deep Learning`

---

# 🌟 Support

If you found this project useful, consider giving the repository a ⭐ on GitHub.