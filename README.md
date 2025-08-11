# FaceRecognizationAPI

A Python + Flask based API for **face recognition** using the [`face_recognition`](https://github.com/Sd8698621/FaceRecognizationAPI) library and OpenCV.  
It supports **face detection, recognition, and labeling** with a simple RESTful API interface.

---

## 🚀 Features

- Detect faces in images
- Recognize faces from a trained dataset
- Store and load known face encodings using `pickle`
- Upload images via API or HTML form
- Easy-to-extend Flask backend
- Session-based user handling

---

## 📂 Project Structure

FaceRecognizationAPI/
│
├── app.py # Main Flask application
├── dataset/ # Folder for storing known face images
├── upload/ # Temporary upload folder
├── flask_session_dir/ # Session storage
├── templates/ # HTML templates for testing via browser
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## 🛠 Requirements

- Python 3.7+
- Flask
- OpenCV (`cv2`)
- face_recognition
- numpy

Install dependencies using:

###⚙️ Usage
1️⃣ Clone the repository
