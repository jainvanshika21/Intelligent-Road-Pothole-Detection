# 🛣️ Intelligent Road Pothole Detection System

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-success)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow)

An AI-powered Computer Vision application that detects road potholes from uploaded videos using **YOLOv8**, **OpenCV**, and **Streamlit**. The system performs real-time detection, visualizes results with bounding boxes, generates downloadable reports, provides analytics, and evaluates model performance using **5-Fold Cross Validation**.

---

## 🚀 Features

- 🎥 Upload MP4 road videos
- 🤖 Real-time pothole detection using YOLOv8
- 📦 Live detection with bounding boxes
- 📊 Interactive Analytics Dashboard
- 📈 Detection confidence distribution
- 📉 Frame-wise pothole detection trend
- 📄 Download detection reports (CSV)
- 🎯 Adjustable confidence threshold
- 🧠 5-Fold Cross Validation model evaluation
- ⚡ Clean and interactive Streamlit interface

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Programming | Python |
| Computer Vision | OpenCV |
| Deep Learning | Ultralytics YOLOv8 |
| Web Framework | Streamlit |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib |

---

## 📂 Project Structure

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
│   ├── make_dataset.py
│   ├── train_fold.py
│   └── kfold_create.py
│
├── models/
│   └── pothole.pt
│
├── data/
│   ├── uploads/
│   └── outputs/
│
├── runs/
│   └── detect/
│       └── kfold_results/
│
├── reports/
├── screenshots/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Intelligent-Road-Pothole-Detection.git
```

Move into the project folder

```bash
cd Intelligent-Road-Pothole-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app/streamlit_app.py
```

---

# 🎥 How to Use

1. Launch the Streamlit application.
2. Upload a road video (`.mp4`).
3. Adjust the confidence threshold if required.
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
Detection Results
     │
     ├────────► Analytics Dashboard
     │
     └────────► Model Performance (5-Fold CV)
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

## 🏠 Home Page

> *(Add `screenshots/home.png` here)*

---

## 🎥 Live Detection

> *(Add `screenshots/live_detection.png` here)*

---

## 📊 Analytics Dashboard

> *(Add `screenshots/analytics.png` here)*

---

## 🤖 Model Performance

> *(Add `screenshots/model_performance.png` here)*

---

# 📦 Output

The system generates:

- ✅ Live pothole detection
- ✅ Bounding box visualization
- ✅ Detection report
- ✅ Downloadable CSV
- ✅ Analytics dashboard
- ✅ Model evaluation dashboard

---

# 📈 Future Enhancements

- 📍 GPS-based pothole mapping
- ☁️ Cloud deployment
- 📱 Mobile application
- 🎥 Live camera detection
- 🚧 Pothole severity classification
- 🌍 Smart city integration

---

# 📚 Dependencies

```text
streamlit
opencv-python
ultralytics
numpy
pandas
matplotlib
```

Install using:

```bash
pip install -r requirements.txt
```

---

# 👩‍💻 Author

**Vanshika Jain**

MCA Student

**Skills:** Python • Computer Vision • YOLOv8 • OpenCV • Streamlit • Data Analysis

---

## 🌟 Support

If you found this project useful, consider giving it a ⭐ on GitHub.