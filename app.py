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

def get_face_name(encoding):
    """Get the name of a known face encoding, or None if unknown."""
    if not known_encodings:
        return None
    matches = face_recognition.compare_faces(
        known_encodings, encoding, tolerance=MATCH_THRESHOLD
    )
    face_distances = face_recognition.face_distance(known_encodings, encoding)
    
    if any(matches):
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            return known_names[best_match_index]
    return None

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

def process_face_recognition(encodings, face_locations):
    """Process face encodings and return recognized names and unknown faces info."""
    recognized_names = []
    unknown_encodings = []
    unknown_locations = []
    
    for i, encoding in enumerate(encodings):
        if known_encodings:
            # Compare with all known faces
            matches = face_recognition.compare_faces(
                known_encodings, encoding, tolerance=MATCH_THRESHOLD
            )
            face_distances = face_recognition.face_distance(known_encodings, encoding)
            
            if any(matches):
                # Find the best match
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    recognized_names.append(known_names[best_match_index])
                    continue
        
        # Face is unknown
        unknown_encodings.append(encoding)
        unknown_locations.append(face_locations[i])
    
    return recognized_names, unknown_encodings, unknown_locations

def format_names_message(names, message_type="recognized"):
    """Format a list of names into a user-friendly message."""
    if not names:
        return ""
    
    # Remove duplicates while preserving order
    unique_names = list(dict.fromkeys(names))
    
    if len(unique_names) == 1:
        return f"{message_type.capitalize()} face: {unique_names[0]}"
    else:
        return f"{message_type.capitalize()} faces: {', '.join(unique_names)}"

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

                # Process face recognition
                recognized_names, unknown_encodings, unknown_locations = process_face_recognition(
                    encodings, face_locations
                )

                # Prepare faces data for labeling (only unknown faces)
                faces_data = []
                for i, location in enumerate(unknown_locations):
                    face_b64 = encode_face_to_base64(image, location)
                    faces_data.append({
                        "index": i,
                        "image_b64": face_b64,
                    })

                # Save session data for label step
                session["uploaded_image_path"] = filepath
                session["unknown_encodings"] = serialize_encodings(unknown_encodings)
                session["unknown_locations"] = pickle.dumps(unknown_locations).hex()

                # Handle different scenarios
                if not unknown_encodings and not recognized_names:
                    # This shouldn't happen if faces were detected, but just in case
                    os.remove(filepath)
                    flash("No faces could be processed.", "error")
                    return redirect(request.url)
                
                elif not unknown_encodings:
                    # All faces are recognized
                    os.remove(filepath)
                    message = format_names_message(recognized_names, "recognized")
                    flash(message, "info")
                    return redirect(request.url)
                
                elif not recognized_names:
                    # All faces are unknown
                    flash(f"Found {len(unknown_encodings)} unknown face(s). Please label them below.", "info")
                    return render_template("train/label_faces.html", faces=faces_data)
                
                else:
                    # Mixed: some recognized, some unknown
                    recognized_message = format_names_message(recognized_names, "recognized")
                    unknown_count = len(unknown_encodings)
                    flash(f"{recognized_message}. Found {unknown_count} unknown face(s) - please label them below.", "info")
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
    required_session_keys = ["uploaded_image_path", "unknown_encodings", "unknown_locations"]
    if not all(key in session for key in required_session_keys):
        flash("Session expired or invalid. Please upload image again.", "error")
        return redirect(url_for("upload_image"))

    # Retrieve session data
    image_path = session.pop("uploaded_image_path")
    unknown_encodings = deserialize_encodings(session.pop("unknown_encodings"))
    unknown_locations = pickle.loads(bytes.fromhex(session.pop("unknown_locations")))

    # Validate image file
    image = cv2.imread(image_path)
    if image is None:
        flash("Uploaded image is missing or corrupted.", "error")
        return redirect(url_for("upload_image"))

    # Extract names from form
    new_names = []
    for i in range(len(unknown_encodings)):
        field_name = f"name_{i}"
        name = request.form.get(field_name, "").strip()
        if not name:
            name = "Unknown"
        new_names.append(name)

    # Add new faces to known encodings
    global known_encodings, known_names
    added_faces = []
    skipped_faces = []

    for i, encoding in enumerate(unknown_encodings):
        # Double-check if face is still unknown (safety check)
        if not is_encoding_known(encoding):
            known_encodings.append(encoding)
            known_names.append(new_names[i])
            save_face_image(image, unknown_locations[i], new_names[i])
            added_faces.append(new_names[i])
        else:
            # Face was somehow already added (rare edge case)
            existing_name = get_face_name(encoding)
            skipped_faces.append(f"{new_names[i]} (already exists as {existing_name})")

    # Save updated encodings
    save_encodings()

    # Clean up uploaded image
    if os.path.exists(image_path):
        os.remove(image_path)

    # Provide feedback to user
    messages = []
    if added_faces:
        added_message = format_names_message(added_faces, "added")
        messages.append(f"Training complete. {added_message}")
    
    if skipped_faces:
        messages.append(f"Skipped faces: {', '.join(skipped_faces)}")
    
    if not added_faces and not skipped_faces:
        messages.append("No new faces were added.")

    # Flash appropriate message
    if added_faces:
        flash(" | ".join(messages), "success")
    else:
        flash(" | ".join(messages), "info")

    return redirect(url_for("upload_image"))


@app.route("/dataset")
def view_dataset():
    """View all known faces in the dataset."""
    dataset_info = {}
    
    if os.path.exists(dataset_dir):
        for person_folder in os.listdir(dataset_dir):
            person_path = os.path.join(dataset_dir, person_folder)
            if os.path.isdir(person_path):
                image_count = len([f for f in os.listdir(person_path) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                dataset_info[person_folder] = image_count
    
    return render_template("train/dataset.html", dataset=dataset_info, 
                         total_people=len(dataset_info),
                         total_images=sum(dataset_info.values()))


@app.route("/reset", methods=["POST"])
def reset_dataset():
    """Reset the entire face recognition dataset."""
    try:
        # Remove encodings file
        if os.path.exists(encodings_file):
            os.remove(encodings_file)
        
        # Clear global variables
        global known_encodings, known_names
        known_encodings.clear()
        known_names.clear()
        
        # Optionally remove dataset directory (uncomment if you want to delete face images too)
        # import shutil
        # if os.path.exists(dataset_dir):
        #     shutil.rmtree(dataset_dir)
        # os.makedirs(dataset_dir, exist_ok=True)
        
        flash("Face recognition dataset has been reset successfully.", "success")
    except Exception as e:
        flash(f"Error resetting dataset: {e}", "error")
    
    return redirect(url_for("upload_image"))


if __name__ == "__main__":
    load_encodings()
    print(f"Loaded {len(known_names)} known faces: {', '.join(known_names) if known_names else 'None'}")
    app.run(host="0.0.0.0", port=3000, debug=True)
