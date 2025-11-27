import cv2
import numpy as np
import mediapipe as mp
from pythonosc import udp_client
import sys
import math
import tkinter as tk
from tkinter import ttk
import threading
import time

# Initialize OSC client
osc_client = udp_client.SimpleUDPClient("127.0.0.1", 4567)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles




# Global variables
current_camera = None
running = False
loading = False
frame_width = 640
frame_height = 480
window_name = "Face Tracking - Smile Detection"

def stop_current_stream():
    global current_camera, running
    if current_camera is not None:
        running = False
        current_camera.release()
        cv2.destroyAllWindows()
    current_camera = None

def process_frame():
    """Process a single frame from the camera"""
    global current_camera, running, frame_width, frame_height
    
    if not running or current_camera is None:
        return
    
    success, img = current_camera.read()
    if not success:
        print("Failed to read frame")
        stop_current_stream()
        update_ui_state()
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    smile_value = 0.5  # Default neutral
    expression = "No face detected"
    corner_lift = 0

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw face mesh
            mp_draw.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            
            # Calculate smile/frown value
            smile_value, expression, corner_lift = calculate_smile_frown(
                face_landmarks, frame_width, frame_height
            )
            
            # Draw mouth landmarks for visualization
            left_mouth = face_landmarks.landmark[61]
            right_mouth = face_landmarks.landmark[291]
            upper_lip = face_landmarks.landmark[13]
            lower_lip = face_landmarks.landmark[14]
            
            # Convert normalized coordinates to pixel coordinates
            left_pt = (int(left_mouth.x * frame_width), int(left_mouth.y * frame_height))
            right_pt = (int(right_mouth.x * frame_width), int(right_mouth.y * frame_height))
            upper_pt = (int(upper_lip.x * frame_width), int(upper_lip.y * frame_height))
            lower_pt = (int(lower_lip.x * frame_width), int(lower_lip.y * frame_height))
            
            # Draw mouth corners and lips
            cv2.circle(img, left_pt, 5, (0, 255, 0), -1)
            cv2.circle(img, right_pt, 5, (0, 255, 0), -1)
            cv2.circle(img, upper_pt, 5, (255, 0, 0), -1)
            cv2.circle(img, lower_pt, 5, (255, 0, 0), -1)
            
            # Display values on screen
            cv2.putText(img, f'Expression: {expression}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f'Smile Value: {smile_value:.2f}', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f'Corner Lift: {corner_lift:.2f}', (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    else:
        cv2.putText(img, 'No face detected', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Send OSC messages
    osc_client.send_message("/face/smile", smile_value)
    osc_client.send_message("/face/expression", 1.0 if expression == "Smiling" else 0.0)

    cv2.imshow(window_name, img)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Esc key
        stop_current_stream()
        update_ui_state()
        return
    
    # Check if window was closed by clicking X
    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            stop_current_stream()
            update_ui_state()
            return
    except:
        # Window doesn't exist anymore
        stop_current_stream()
        update_ui_state()
        return
    
    # Schedule next frame processing
    if running:
        root.after(10, process_frame)

def start_face_tracking(camera_index):
    global current_camera, running, loading, frame_width, frame_height
    
    stop_current_stream()
    loading = True
    update_ui_state()
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open webcam {camera_index}.")
        loading = False
        update_ui_state()
        return

    current_camera = cap
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    running = True
    loading = False
    update_ui_state()
    
    # Start processing frames using Tkinter's after() method
    root.after(10, process_frame)

def calculate_smile_frown(face_landmarks, img_width, img_height):
    """
    Calculate smile vs frown based on mouth landmarks
    Returns: smile_value (0-1), where 0 is frown, 0.5 is neutral, 1 is smile
    """
    # Key mouth landmarks for smile detection
    # Left mouth corner: 61
    # Right mouth corner: 291
    # Upper lip center: 13
    # Lower lip center: 14
    # Nose tip: 1
    
    left_mouth = face_landmarks.landmark[61]
    right_mouth = face_landmarks.landmark[291]
    upper_lip = face_landmarks.landmark[13]
    lower_lip = face_landmarks.landmark[14]
    nose_tip = face_landmarks.landmark[1]
    
    # Calculate mouth width
    mouth_width = np.hypot(
        (right_mouth.x - left_mouth.x) * img_width,
        (right_mouth.y - left_mouth.y) * img_height
    )
    
    # Calculate average y position of mouth corners
    mouth_corners_y = (left_mouth.y + right_mouth.y) / 2
    
    # Calculate mouth center y position
    mouth_center_y = (upper_lip.y + lower_lip.y) / 2
    
    # Calculate the vertical distance from nose to mouth corners
    nose_to_mouth_corners = (mouth_corners_y - nose_tip.y) * img_height
    
    # Calculate if corners are raised (smile) or lowered (frown)
    # When smiling, mouth corners are higher than mouth center
    # When frowning, mouth corners are lower than mouth center
    corner_lift = (mouth_center_y - mouth_corners_y) * img_height
    
    # Normalize the value to 0-1 range
    # Positive corner_lift = smile, negative = frown
    smile_threshold = 5  # pixels
    frown_threshold = 3
    
    if corner_lift > smile_threshold:
        smile_value = min(1.0, 0.5 + (corner_lift / 20))  # Smiling
        expression = "Smiling"
    elif corner_lift < -frown_threshold:
        smile_value = max(0.0, 0.5 + (corner_lift / 20))  # Frowning
        expression = "Frowning"
    else:
        smile_value = 0.5  # Neutral
        expression = "Neutral"
    
    return smile_value, expression, corner_lift


def update_ui_state():
    if loading:
        status_label.config(text="Loading...")
        start_button.config(state=tk.DISABLED)
        camera_menu.config(state=tk.DISABLED)
    elif running:
        status_label.config(text="Running")
        start_button.config(state=tk.NORMAL)
        camera_menu.config(state=tk.NORMAL)
    else:
        status_label.config(text="Stopped")
        start_button.config(state=tk.NORMAL)
        camera_menu.config(state=tk.NORMAL)

# Create the main window
root = tk.Tk()
root.title("Camera Selection")

# Create a list of camera options
camera_options = [f"Camera {i}" for i in range(10)]  # Assuming up to 10 cameras

# Create and pack the dropdown menu
camera_var = tk.StringVar(root)
camera_var.set(camera_options[0])  # Set default value
camera_menu = ttk.Combobox(root, textvariable=camera_var, values=camera_options)
camera_menu.pack(pady=10)

# Create button frame for Start and Stop buttons
button_frame = ttk.Frame(root)
button_frame.pack(pady=10)

# Create and pack the start button
start_button = ttk.Button(button_frame, text="Start", command=lambda: start_face_tracking(int(camera_var.get().split()[1])))
start_button.pack(side=tk.LEFT, padx=5)

# Create and pack the stop button
stop_button = ttk.Button(button_frame, text="Stop", command=stop_current_stream)
stop_button.pack(side=tk.LEFT, padx=5)

# Create and pack the status label (throbber)
status_label = ttk.Label(root, text="Stopped")
status_label.pack(pady=10)

# Handle window close event
def on_closing():
    stop_current_stream()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Start the Tkinter event loop
root.mainloop()
