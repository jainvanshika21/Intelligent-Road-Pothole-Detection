# 🛣️ Intelligent Road Pothole Detection System

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-success)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit)

> **Built as an end-to-end Computer Vision project demonstrating Deep Learning, Video Processing, Interactive Analytics, and Model Evaluation using YOLOv8 and Streamlit.**

An AI-powered Computer Vision application that detects road potholes from uploaded videos using **YOLOv8**, **OpenCV**, and **Streamlit**. The system performs real-time detection, visualizes results with bounding boxes, generates downloadable reports, provides interactive analytics, and evaluates model performance using **5-Fold Cross Validation**.

---

# ⭐ Project Highlights

- 🚀 Developed an AI-powered pothole detection system using YOLOv8.
- 🎥 Detects potholes from uploaded road videos in real time.
- 📦 Displays live bounding boxes for detected potholes.
- 📊 Interactive analytics dashboard with visual insights.
- 📄 Generates downloadable CSV detection reports.
- 🎯 Adjustable confidence threshold for improved detection.
- 🧠 Evaluated model performance using 5-Fold Cross Validation.
- 💻 Built a responsive web interface using Streamlit.

---

# 🎥 Demo Video

Click below to watch the project demo.

▶️ **Demo:** [Pothole_Detection_Demo.mp4](assets/Pothole_Detection_Demo.mp4)

---

# 🚀 Features

- 🎥 Upload MP4 road videos
- 🤖 Real-time pothole detection using YOLOv8
- 📦 Live pothole detection with bounding boxes
- 📊 Interactive Analytics Dashboard
- 📈 Detection confidence distribution
- 📉 Frame-wise pothole detection trend
- 📄 Download detection reports (CSV)
- 🎯 Adjustable confidence threshold
- 🧠 5-Fold Cross Validation evaluation
- ⚡ User-friendly Streamlit interface

---

# 🛠️ Tech Stack

| Category | Technologies |
|-----------|--------------|
| Programming | Python |
| Computer Vision | OpenCV |
| Deep Learning | Ultralytics YOLOv8 |
| Web Framework | Streamlit |
| Data Analysis | Pandas, NumPy |
| Visualization | Plotly, Matplotlib |

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
│
├── data/
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
├── README.md
└── .gitignore
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/jainvanshika21/Intelligent-Road-Pothole-Detection.git
```

Move into the project directory

```bash
cd Intelligent-Road-Pothole-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Launch the application

```bash
streamlit run app/streamlit_app.py
```

---

# 🎥 How to Use

1. Launch the Streamlit application.
2. Upload a road video (`.mp4`).
3. Adjust the confidence threshold.
4. Click **Start Detection**.
5. View live pothole detection with bounding boxes.
6. Explore the Analytics Dashboard.
7. Download the detection report as a CSV file.

---

# 🔄 Workflow

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
Detection Report
      │
      ├────────► Analytics Dashboard
      │
      └────────► Model Evaluation
```

---

# 📊 Analytics Dashboard

The application provides:

- 📌 Total pothole detections
- 📈 Detection confidence distribution
- 📉 Frame-wise pothole detection trend
- 📋 Detection report table
- 📄 Downloadable CSV report

---

# 🤖 Model Evaluation

The YOLOv8 model is evaluated using **5-Fold Cross Validation**.

Metrics displayed include:

- mAP@50
- Fold-wise performance
- Average mAP@50
- Performance comparison chart

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

# 📦 Output

The system generates:

- ✅ Live pothole detection
- ✅ Bounding box visualization
- ✅ Detection reports
- ✅ Downloadable CSV
- ✅ Interactive analytics dashboard
- ✅ Model evaluation dashboard

---

# 📈 Future Enhancements

- 📍 GPS-based pothole mapping
- ☁️ Cloud deployment
- 📱 Mobile application
- 🎥 Live camera detection
- 🚧 Pothole severity classification
- 🌍 Smart City integration

---

# 📚 Dependencies

```text
streamlit
opencv-python
ultralytics
numpy
pandas
plotly
matplotlib
```

Install using

```bash
pip install -r requirements.txt
```

---

# 👩‍💻 Author

**Vanshika Jain**

MCA Student

### Skills

Python • YOLOv8 • OpenCV • Streamlit • Computer Vision • Pandas • NumPy • Plotly • Machine Learning • Deep Learning

---

# 🌟 Support

If you found this project useful, please consider giving it a ⭐ on GitHub.
