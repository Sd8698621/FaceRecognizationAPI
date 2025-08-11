import os
import cv2
import base64
import pickle
import face_recognition
from flask import (
    Flask, request, redirect, flash, render_template,
    session, url_for
)
from werkzeug.utils import secure_filename
import numpy as np
from flask_session import Session

# --- Setup directories ---
upload_dir = "upload"
dataset_dir = "dataset"
session_dir = "flask_session_dir"
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(upload_dir, exist_ok=True)
os.makedirs(session_dir, exist_ok=True)

# --- Configurations ---
encodings_file = "face_encodings.pkl"
MAX_WIDTH = 800
MATCH_THRESHOLD = 0.45
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# --- Flask app setup ---
app = Flask(__name__)
app.secret_key = "supersecretkey"

# Flask-Session config for server-side sessions
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = session_dir
app.config["SESSION_PERMANENT"] = False
Session(app)

# --- Globals ---
known_encodings = []
known_names = []

# --- Helper Functions ---

def load_encodings():
    """Load known face encodings and names from disk."""
    global known_encodings, known_names
    if os.path.exists(encodings_file):
        with open(encodings_file, "rb") as f:
            data = pickle.load(f)
            known_encodings = data.get("encodings", [])
            known_names = data.get("names", [])

def save_encodings():
    """Save known face encodings and names to disk."""
    with open(encodings_file, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def resize_image(image):
    """Resize image maintaining aspect ratio if width exceeds MAX_WIDTH."""
    h, w = image.shape[:2]
    if w > MAX_WIDTH:
        scale = MAX_WIDTH / w
        return cv2.resize(image, (MAX_WIDTH, int(h * scale)))
    return image

def is_encoding_known(encoding):
    """Check if the face encoding is already known."""
    if not known_encodings:
        return False
    matches = face_recognition.compare_faces(
        known_encodings, encoding, tolerance=MATCH_THRESHOLD
    )
    return any(matches)

def save_face_image(image, face_location, person_name):
    """Save cropped face image to dataset directory under person's folder."""
    top, right, bottom, left = face_location
    person_dir = os.path.join(dataset_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)
    count = len(os.listdir(person_dir)) + 1
    save_path = os.path.join(person_dir, f"{count}.jpg")
    face_crop = image[top:bottom, left:right]
    cv2.imwrite(save_path, face_crop)
    return save_path

def encode_faces_from_image(image_path):
    """Load image, resize, detect faces and encode them."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image file.")
    image = resize_image(image)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image, model="cnn")
    encodings = face_recognition.face_encodings(rgb_image, face_locations)
    return encodings, face_locations, image

def encode_face_to_base64(image, face_location):
    """Encode cropped face image to base64 string for embedding in HTML."""
    top, right, bottom, left = face_location
    face_crop = image[top:bottom, left:right]
    _, buffer = cv2.imencode(".jpg", face_crop)
    return base64.b64encode(buffer).decode("utf-8")

def serialize_encodings(encodings):
    """Serialize and base64 encode a list of face encodings."""
    return base64.b64encode(pickle.dumps(encodings)).decode("utf-8")

def deserialize_encodings(enc_str):
    """Deserialize base64 encoded face encodings."""
    return pickle.loads(base64.b64decode(enc_str.encode("utf-8")))


# --- Routes ---

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part in the request.", "error")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("No file selected.", "error")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_dir, filename)
            file.save(filepath)

            try:
                encodings, face_locations, image = encode_faces_from_image(filepath)

                if not encodings:
                    flash("No faces detected in the image.", "error")
                    os.remove(filepath)
                    return redirect(request.url)

                faces_data = []
                unknown_encodings = []
                unknown_locations = []

                # Filter only unknown faces for labeling
                for i, encoding in enumerate(encodings):
                    if not is_encoding_known(encoding):
                        unknown_encodings.append(encoding)
                        unknown_locations.append(face_locations[i])
                        face_b64 = encode_face_to_base64(image, face_locations[i])
                        faces_data.append({
                            "index": len(faces_data),
                            "image_b64": face_b64,
                        })

                # Save session data for label step
                session["uploaded_image_path"] = filepath
                session["unknown_encodings"] = serialize_encodings(unknown_encodings)
                session["unknown_locations"] = pickle.dumps(unknown_locations).hex()

                # If all faces are known, no need to label
                if not unknown_encodings:
                    os.remove(filepath)
                    flash(f"All {len(encodings)} faces are already known.", "info")
                    return redirect(request.url)

                return render_template("train/label_faces.html", faces=faces_data)

            except Exception as e:
                # Cleanup file on error
                if os.path.exists(filepath):
                    os.remove(filepath)
                flash(f"Error processing image: {e}", "error")
                return redirect(request.url)
        else:
            flash("Unsupported file format. Please upload PNG, JPG, or JPEG.", "error")
            return redirect(request.url)

    # GET method
    return render_template("train/index.html")


@app.route("/label", methods=["POST"])
def label_faces():
    # Verify session data exists
    if "uploaded_image_path" not in session or "unknown_encodings" not in session or "unknown_locations" not in session:
        flash("Session expired or invalid. Please upload image again.", "error")
        return redirect(url_for("upload_image"))

    image_path = session.pop("uploaded_image_path")
    unknown_encodings = deserialize_encodings(session.pop("unknown_encodings"))
    unknown_locations = pickle.loads(bytes.fromhex(session.pop("unknown_locations")))

    image = cv2.imread(image_path)
    if image is None:
        flash("Uploaded image is missing or corrupted.", "error")
        return redirect(url_for("upload_image"))

    new_names = []
    for i in range(len(unknown_encodings)):
        field_name = f"name_{i}"
        name = request.form.get(field_name, "").strip()
        if not name:
            name = "Unknown"
        new_names.append(name)

    global known_encodings, known_names
    added_faces = []

    for i, encoding in enumerate(unknown_encodings):
        if not is_encoding_known(encoding):
            known_encodings.append(encoding)
            known_names.append(new_names[i])
            save_face_image(image, unknown_locations[i], new_names[i])
            added_faces.append(new_names[i])

    save_encodings()

    # Clean up uploaded image after processing
    if os.path.exists(image_path):
        os.remove(image_path)

    if added_faces:
        flash(f"Training complete. Added faces: {', '.join(added_faces)}", "success")
    else:
        flash("No new faces were added.", "info")

    return redirect(url_for("upload_image"))


if __name__ == "__main__":
    load_encodings()
    app.run(host="0.0.0.0", port=3000, debug=True)
