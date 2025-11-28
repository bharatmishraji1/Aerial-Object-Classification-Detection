🕊️ Aerial Object Detection (Bird vs Drone)
Python TensorFlow YOLOv8 Streamlit License Platform

A complete end-to-end Aerial Object Detection System capable of:

Classifying Bird vs Drone using MobileNetV2 Transfer Learning
Detecting drones/birds in real images using YOLOv8
Providing real-time camera inference
Showing model explanations using Grad-CAM
Rendering metadata, insights, confidence plots and detection overlays
Running inside an elegant Streamlit dashboard UI
Features
Classification
MobileNetV2 Transfer Learning model
Achieves 98–100% accuracy on test dataset
Grad-CAM heatmaps for explainable AI
Confidence bars & probability comparison
Detection
YOLOv8 (best.pt) integration
Bounding box rendering without .plot()
Real-time camera detection
Fully offline capable
User Interface
Modern dashboard layout
Profile sidebar with GitHub & LinkedIn
Upload, live camera, predictions & analysis
Works on PC & phone
Project Structure

Aerial-Object-Detection/
│
├──  classification/
│   ├──  custom_cnn.py
│   ├──  transfer_learning.py
│   ├──  evaluate.py
│   ├──  model_comparison.png
│   ├──  final_custom_cnn_model.keras
│   └──  final_transfer_learning_model.keras
│
├──  detection/
│   ├──  Train.ipynb
│   ├──  train_yolo.py
│   ├──  bird_drone.yaml
│   └──  yolov8s.pt
│
├──  streamlit_app/
│   ├──  app.py
│   │
│   ├──  models/
│   │   ├──  final_custom_cnn_model.keras
│   │   ├──  final_custom_cnn_model_yolo.keras
│   │   ├──  final_transfer_learning_model.keras
│   │   ├──  final_transfer_learning_model_yolo.keras
│   │   │
│   │   ├──  Checkpoint/
│   │   │   ├──  CNN_Classification/
│   │   │   │   └──  best_custom_cnn.keras  (LFS)
│   │   │   │
│   │   │   ├──  Transfer_Classification/
│   │   │   │   └──  best_transfer_learning.keras (LFS)
│   │   │   │
│   │   │   └──  YoloV8_Detection/
│   │   │       └──  best.pt (LFS)
│   │
│   └──  assets/   (icons, screenshots)
│
├──  report/
│   ├── Final_Report..docx
│   └── Project Title.docx
│
├──  BD_py.jpg
├── 📄 exec.txt
├── 📄 README.md
├── 📄 README.dataset.txt
├── 📄 README.roboflow.txt
├── 📄 .gitignore
└── 📄 .gitattributes


🏛️ Software Architecture Diagram
                   ┌───────────────────────┐
                   │     Input Images      │
                   │ (Drone / Bird Images) │
                   └──────────┬────────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │  Preprocessing     │
                    │ (Resize, Normalize)│
                    └──────────┬─────────┘
                              │
               ┌──────────────┼────────────────┐
               ▼              ▼                ▼
     ┌────────────────┐ ┌───────────────┐ ┌────────────────┐
     │ YOLOv8 Model   │ │ Custom CNN     │ │ Transfer Learn │
     │ (Detection)    │ │ (Classification)│ │ (EfficientNet) │
     └───────┬────────┘ └────────┬────────┘ └───────┬────────┘
             │                   │                    │
             └──────────┬────────┴────────────┬──────┘
                        ▼                     ▼
               ┌─────────────────┐   ┌───────────────────┐
               │ Streamlit UI    │   │ Performance Metrics│
               │ (Real-time App) │   │ Accuracy, Loss     │
               └─────────┬───────┘   └──────────┬────────┘
                         ▼                      ▼
                ┌─────────────────┐    ┌──────────────────┐
                │ Final Prediction │    │ Reports & Plots  │
                └─────────────────┘    └──────────────────┘
→• FlowChart(PipeLine)
        ┌──────────────┐
        │ Upload Image │
        └───────┬──────┘
                ▼
        ┌──────────────────┐
        │ Preprocessing    │
        │ (resize, scale)  │
        └───────┬──────────┘
                ▼
       ┌─────────────────────┐
       │ YOLOv8 Detection    │
       └───────┬─────────────┘
               ▼
       ┌─────────────────────┐
       │ Crop detected ROI   │
       └───────┬─────────────┘
               ▼
   ┌──────────────────────────────┐
   │ CNN/Transfer Learning Class  │
   └─────────┬────────────────────┘
             ▼
     ┌───────────────┐
     │ Final Output  │
     │ (Bird / Drone)│
     └───────────────┘
📦 Installation
1. Clone the Repository

git clone [https://github.com/bharatmishraji1/Aerial-Object-Detection.git](https://github.com/bharatmisrhaji1/Aerial-Object-Detection.git)
cd Aerial-Object-Detection

2. Install Dependencies

pip install -r requirements.txt

3. Run the Streamlit App

streamlit run streamlit_app/app.py

🧠 Model Performance
Model	Accuracy	Precision	Recall	Notes
MobileNetV2 Transfer Learning	⭐ 98–100%	High	High	Final classifier used
Custom CNN	89–92%	Medium	Medium	Baseline model
YOLOv8	–	–	–	Used for detection, not classification
🎯 Outputs
🖼️ Insert Output Image Here
(https://github.com/bharatmishraji1/Aerial-Object-Detection/blob/main/BD_py.jpg)

🧪 How It Works
💡 Classification Pipeline
Input → Resize (224×224)
Normalize [0–1]
MobileNetV2 pretrained backbone
Dense classifier head
Sigmoid → Bird / Drone
🎯 Detection Pipeline
YOLOv8 loads best.pt
Runs inferencing
Generates bounding boxes & labels
Rendered manually using Pillow
🔥 Real-Time Camera Pipeline
Streamlit → OpenCV Frame Capture
Classification + YOLO detection
Live result display
🧑‍💻 Author
Bharat Mishra
Platform	Link
🔗 GitHub	https://github.com/bharatmishraji1
🔗 LinkedIn	http://www.linkedin.com/in/bharat-mishra-974a6b1b6
📜 Changelog
v1.0.0
Added MobileNetV2 classifier
Added YOLOv8 detection engine
Added Grad-CAM visualization
Added real-time webcam inference
Full Streamlit UI created
📄 License
This project is licensed under the MIT License.

⭐ Support
If this project helped you, consider giving it a ⭐ on GitHub!


---








