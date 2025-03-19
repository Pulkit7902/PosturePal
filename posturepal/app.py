from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

# Initialize Flask app
app = Flask(__name__)

# Initialize YOLO and MediaPipe models
model = YOLO('best.pt')
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize chat model
GROQ_API_KEY = "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"
model_chatbot = ChatGroq(model="Gemma2-9b-It", groq_api_key=GROQ_API_KEY)
parser = StrOutputParser()

# Rep counting
rep_count = 0
exercise_state = None


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
    """Analyze frame and return processed frame with feedback and rep counting."""
    global rep_count, exercise_state

    results = model(frame)
    if len(results[0].boxes):
        sorted_boxes = sorted(results[0].boxes, key=lambda x: x.conf, reverse=True)
        class_id = int(sorted_boxes[0].cls)
        exercise_type = {0: 'bicep curl', 1: 'push-up', 2: 'squat'}.get(class_id, 'Unknown')
    else:
        exercise_type = 'Unknown'

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(image_rgb)

    if pose_results.pose_landmarks:
        landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in pose_results.pose_landmarks.landmark]

        # Bicep curl joint points
        left_shoulder = landmarks[11]
        left_elbow = landmarks[13]
        left_wrist = landmarks[15]

        # Push-up joint points
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

            feedback = f"Bicep Curl - Reps: {rep_count}"

        elif exercise_type == "push-up":
            angle = calculate_angle(left_shoulder, left_hip, left_knee)

            if angle < 90 and exercise_state == "down":
                rep_count += 1
                exercise_state = "up"
            elif angle > 160:
                exercise_state = "down"

            feedback = f"Push-up - Reps: {rep_count}"

        elif exercise_type == "squat":
            angle = calculate_angle(left_hip, left_knee, left_ankle)

            if angle < 70 and exercise_state == "down":
                rep_count += 1
                exercise_state = "up"
            elif angle > 140:
                exercise_state = "down"

            feedback = f"Squat - Reps: {rep_count}"

        else:
            feedback = "Unknown exercise"

        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    else:
        feedback = "No pose detected"

    # Add feedback and rep count to the frame
    cv2.putText(frame, f"Exercise: {exercise_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, feedback, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

# def analyze_frame(frame):
#     """Analyze frame and return processed frame with feedback."""
#     global rep_count, exercise_state

#     results = model(frame)

#     # Object Detection - Identify the Exercise Type
#     if len(results[0].boxes):
#         sorted_boxes = sorted(results[0].boxes, key=lambda x: x.conf, reverse=True)
#         class_id = int(sorted_boxes[0].cls)
#         exercise_type = {0: 'bicep curl', 1: 'push-up', 2: 'squat'}.get(class_id, 'Unknown')
#     else:
#         exercise_type = 'Unknown'

#     # Pose Estimation
#     image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     pose_results = pose.process(image_rgb)

#     if pose_results.pose_landmarks:
#         landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in pose_results.pose_landmarks.landmark]

#         # Choose exercise form feedback
#         if exercise_type == "bicep curl":
#             feedback = "Good form" if np.random.rand() > 0.3 else "Bad form"
#         elif exercise_type == "push-up":
#             feedback = "Keep it up!" if np.random.rand() > 0.3 else "Fix your posture"
#         elif exercise_type == "squat":
#             feedback = "Great squats!" if np.random.rand() > 0.3 else "Lower yourself properly"
#         else:
#             feedback = "Unknown exercise"

#         # Draw landmarks
#         mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#     else:
#         feedback = "No pose detected"

#     # Add feedback on the frame
#     cv2.putText(frame, f"Exercise: {exercise_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     cv2.putText(frame, feedback, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     return frame


def generate_frames():
    """Capture frames, apply posture analysis, and stream them."""
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Apply posture analysis before streaming
            frame = analyze_frame(frame)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/webcam')
def webcam():
    """Render webcam page."""
    return render_template('webcam.html')


@app.route('/video_feed')
def video_feed():
    """Route for video feed with posture analysis."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Render home page."""
    return render_template('file.html')


@app.route('/query', methods=['POST'])
def query():
    """Handle chatbot queries."""
    user_query = request.form.get("query")

    if user_query:
        messages = [
            SystemMessage(content="You are a fitness bot. Provide detailed explanations."),
            HumanMessage(content=user_query)
        ]

        try:
            response = parser.invoke(model_chatbot.invoke(messages))
            return jsonify({"response": response})
        except Exception as e:
            return jsonify({"error": str(e)})

    return jsonify({"error": "No query provided"})


if __name__ == '__main__':
    app.run(debug=True)

