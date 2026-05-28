# YOLOv8 Nano Crowd Head Detection System

A lightweight AI-powered crowd head detection and counting system built using **YOLOv8 Nano**, trained on event crowd imagery and deployed as a live web application using **Flask**.

🔗 Live Demo:
[YOLOv8 Nano Crowd Detection App](https://huggingface.co/spaces/Findmeanickpls/yolov8n-Nano_headcount)

---

# Overview

This project focuses on real-time crowd head detection in semi-outdoor, indoor, and outdoor event environments using a custom-trained YOLOv8 Nano model.

The system was designed to explore lightweight crowd monitoring and crowd counting while maintaining efficient inference speed on modest hardware. It includes:

* Custom dataset preprocessing
* Image segmentation pipeline
* Data annotation workflow
* YOLOv8 Nano training
* Flask-based deployment
* Web upload interface for live detection

Unlike generic object detection demos, this project was specifically developed around event crowd analysis and dense human head detection challenges such as:

* Occlusion
* Scale variation
* Complex backgrounds
* Outdoor lighting inconsistencies
* Dense crowd overlap

---

# Live Deployment

The application is publicly deployed on Render:

🔗 https://huggingface.co/spaces/Findmeanickpls/yolov8n-Nano_headcount

Users can:

* Upload crowd images
* Run head detection
* View detected results
* Estimate crowd density visually

---

# Project Motivation

Crowd analysis systems are increasingly used in:

* Event management
* Smart surveillance
* Occupancy monitoring
* Public safety
* Crowd engagement analytics

This project explores how a lightweight detection model like YOLOv8 Nano performs in practical crowd scenarios while remaining deployable through a simple web interface.

---

# Model Architecture

The project uses:

* **YOLOv8 Nano**
* Ultralytics framework
* Python + Flask deployment

YOLOv8 Nano was selected because of:

* Fast inference speed
* Lightweight deployment size
* Real-time suitability
* Lower computational overhead

---

# Dataset & Annotation Workflow

The dataset consisted of crowd event imagery containing multiple visible human heads in semi-outdoor and outdoor scenes.

## Annotation

Bounding boxes were manually annotated using:

* Makesense.ai

Annotations focused specifically on:

* Human heads
* Dense crowd regions
* Partial occlusions

---

# Image Preprocessing Pipeline

A custom Python preprocessing pipeline was implemented to improve small-object detection performance.

## Preprocessing Included

* Image segmentation into 4 quadrants
* Dataset restructuring
* Data augmentation
* Label alignment
* Bounding box preservation

Quadrant segmentation helped improve:

* Small head visibility
* Dense crowd learning
* Local feature extraction

---

# Training Details

| Parameter   | Value               |
| ----------- | ------------------- |
| Model       | YOLOv8 Nano         |
| Epochs      | 24                  |
| Framework   | Ultralytics         |
| Task        | Head Detection      |
| Environment | Crowd Event Imagery |

---

# Performance Summary

| Metric          | Result    |
| --------------- | --------- |
| mAP50           | 0.584     |
| mAP50-95        | 0.168     |
| Recall          | 0.611     |
| Inference Speed | 89–225 ms |

## Observations

* Stronger performance in semi-outdoor/indoor environments
* Reduced localisation accuracy in dense outdoor scenes
* False positives increased in highly crowded outdoor images
* Efficient inference on lightweight hardware

---

# Web Application

The deployment layer was developed using Flask.

## Features

* Image upload interface
* Detection visualisation
* Bounding box rendering
* Crowd count estimation
* Lightweight deployment workflow

The goal was to keep the interface simple, accessible, and easy to deploy without requiring GPU infrastructure.

---

# Tech Stack

## AI / Computer Vision

* YOLOv8 Nano
* Ultralytics
* OpenCV
* NumPy
* Pillow

## Backend

* Flask
* Gunicorn

## Deployment

* Render

---

# Project Structure

```bash
.
├── web interface/
│   ├── app.py
│   ├── templates/
│   ├── uploads/
│   ├── results/
│   ├── requirements.txt
│   └── runs/
│
├── runs/
├── head counting model.ipynb
├── yolov8n.pt
└── README.md
```

---

# Running Locally

## Clone Repository

```bash
git clone https://github.com/findmeanickpls/Yolov8-nano.git
cd Yolov8-nano
```

---

## Install Requirements

```bash
pip install -r requirements.txt
```

---

## Run Flask App

```bash
cd "web interface"
python app.py
```

---

## Open Browser

```bash
http://127.0.0.1:5000
```

---

# Deployment Notes

The application is deployed using:

* Flask
* Gunicorn
* Render

## Start Command

```bash
gunicorn app:app
```

---

# Current Limitations

The current system still faces several challenges:

* Outdoor overfitting
* False positives in dense scenes
* Reduced localisation precision
* No batch image processing
* Limited generalisation across varied crowd scales

---

# Future Improvements

Planned improvements include:

* Better outdoor dataset balancing
* Larger YOLOv8 variants
* Batch image upload support
* Video-based crowd tracking
* Real-time stream processing
* Improved non-maximum suppression
* Confidence threshold optimisation
* Cloud storage integration

---

# Why This Project Matters

This project demonstrates that lightweight object detection systems can still provide meaningful crowd analysis capabilities without requiring large-scale infrastructure.

It also highlights the practical challenges of deploying crowd detection systems in real-world environments where:

* crowd density varies,
* visibility changes,
* and ethical considerations become important.

---

# Author

Developed by Sheriffdeen

GitHub: https://github.com/findmeanickpls

---

# Disclaimer

This project was developed for educational and research purposes only.

It is not intended for production-grade surveillance deployment without further optimisation, testing, and ethical review.

