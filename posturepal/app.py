from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, session, flash
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, Length
import firebase_admin
from firebase_admin import credentials, auth, firestore
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from playsound import playsound
import threading
import time

# ===========================
# Flask App Initialization
# ===========================
app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Replace with your own secret key

# ===========================
# Firebase Initialization (Fixed)
# ===========================
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred)

# Initialize Firestore
db = firestore.client()

# ===========================
# YOLO and MediaPipe Initialization
# ===========================
model = YOLO('best.pt')
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# ===========================
# Chatbot Initialization
# ===========================
GROQ_API_KEY = "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"
model_chatbot = ChatGroq(model="Gemma2-9b-It", groq_api_key=GROQ_API_KEY)
parser = StrOutputParser()

# ===========================
# Flask-WTF Forms
# ===========================
class SignupForm(FlaskForm):
    full_name = StringField('Full Name', validators=[DataRequired(), Length(min=3, max=50)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    submit = SubmitField('Sign Up')

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    submit = SubmitField('Login')

# ===========================
# Routes: Authentication
# ===========================
@app.route('/')
def home():
    """Render home page."""
    if 'user' in session:
        return render_template('home.html', user=session['user'])
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = SignupForm()

    if form.validate_on_submit():
        full_name = form.full_name.data
        email = form.email.data
        password = form.password.data

        try:
            # Create user in Firebase Authentication
            user = auth.create_user(
                email=email,
                password=password,
                display_name=full_name  # Store full name in auth profile
            )

            # Save user details in Firestore Database
            user_data = {
                "full_name": full_name,
                "email": email,
                "uid": user.uid
            }
            
            db.collection('users').document(user.uid).set(user_data)

            flash('Account created successfully!', 'success')
            return redirect(url_for('login'))

        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')

    return render_template('signup.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data
        
        try:
            user = auth.get_user_by_email(email)
            user_token = auth.create_custom_token(user.uid)
            
            # ✅ Fetch full name from Firestore
            user_doc = db.collection('users').document(user.uid).get()
            
            if user_doc.exists:
                full_name = user_doc.to_dict().get('full_name', 'User')  # Fallback to 'User'
            else:
                full_name = 'User'

            # ✅ Store full name in the session
            session['user'] = {
                'email': email,
                'uid': user.uid,
                'token': user_token,
                'full_name': full_name
            }
            
            flash('Logged in successfully!', 'success')
            return redirect(url_for('home'))

        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
    
    return render_template('login.html', form=form)


@app.route('/logout')
def logout():
    """Logout route."""
    session.pop('user', None)
    flash('Logged out successfully!', 'info')
    return redirect(url_for('login'))

# ===========================
# Posture Analysis Variables
# ===========================
rep_count = 0
exercise_state = None
last_sound_time = 0  # Timestamp of the last sound
sound_cooldown = 5  # Cooldown period in seconds

# ===========================
# Helper Functions
# ===========================
def play_alert_sound():
    """Play alert sound with cooldown."""
    global last_sound_time
    current_time = time.time()
    if current_time - last_sound_time > sound_cooldown:
        last_sound_time = current_time
        threading.Thread(target=lambda: playsound("alert_sound.mp3")).start()

def calculate_angle(a, b, c):
    """Calculate angle between three points."""
    ab = np.array([b[0] - a[0], b[1] - a[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    dot_product = np.dot(ab, bc)
    mag_ab = np.linalg.norm(ab)
    mag_bc = np.linalg.norm(bc)

    if mag_ab == 0 or mag_bc == 0:
        return 0

    angle = np.degrees(np.arccos(dot_product / (mag_ab * mag_bc)))
    return angle

def analyze_frame(frame):
    """Analyze frame and return processed frame with posture feedback."""
    global rep_count, exercise_state

    results = model(frame)
    exercise_type = 'Unknown'

    if len(results[0].boxes):
        sorted_boxes = sorted(results[0].boxes, key=lambda x: x.conf, reverse=True)
        class_id = int(sorted_boxes[0].cls)
        exercise_type = {0: 'bicep curl', 1: 'push-up', 2: 'squat'}.get(class_id, 'Unknown')

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(image_rgb)

    if pose_results.pose_landmarks:
        landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in pose_results.pose_landmarks.landmark]

        # Joint Points
        left_shoulder = landmarks[11]
        left_elbow = landmarks[13]
        left_wrist = landmarks[15]
        left_hip = landmarks[23]
        left_knee = landmarks[25]
        left_ankle = landmarks[27]

        if exercise_type == "bicep curl":
            angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            if angle < 60 and exercise_state == "down":
                rep_count += 1
                exercise_state = "up"
            elif angle > 140:
                exercise_state = "down"
        elif exercise_type == "push-up":
            angle = calculate_angle(left_shoulder, left_hip, left_knee)
        elif exercise_type == "squat":
            angle = calculate_angle(left_hip, left_knee, left_ankle)

        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.putText(frame, f"Exercise: {exercise_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def generate_frames():
    """Stream video feed."""
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        frame = analyze_frame(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video feed route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

# Set the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Read and analyze the image
            frame = cv2.imread(filepath)
            if frame is None:
                flash('Invalid image file')
                return redirect(request.url)
            analyzed_frame = analyze_frame(frame)
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename)
            cv2.imwrite(result_path, analyzed_frame)

            # Show the analyzed image
            return render_template('upload.html', result_image='uploads/' + 'result_' + filename)
    return render_template('upload.html')

# ===========================
# Run App
# ===========================
if __name__ == '__main__':
    app.run(debug=True)
