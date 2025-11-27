import cv2
import numpy as np
import math
from pythonosc import udp_client
import tkinter as tk
from tkinter import ttk

# Initialize OSC client
osc_client = udp_client.SimpleUDPClient("127.0.0.1", 4567)

# Global variables
current_camera = None
running = False
loading = False

def stop_current_stream():
    global current_camera, running
    running = False
    if current_camera is not None:
        current_camera.release()
    cv2.destroyAllWindows()
    current_camera = None
    update_ui_state()

def start_sticker_tracking(camera_index):
    global current_camera, running, loading
    
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
    running = True
    loading = False
    update_ui_state()

    while running:
        success, img = cap.read()
        if not success:
            break

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Narrow red ranges
        lower_red1 = np.array([0, 150, 150])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 150, 150])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out tiny blobs
        min_area = 300
        contours = [c for c in contours if cv2.contourArea(c) > min_area]

        # Keep only the two largest
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        centers = []
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))
                cv2.circle(img, (cx, cy), 10, (0, 255, 0), -1)

        if len(centers) == 2:
            (x1, y1), (x2, y2) = centers
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            dx = x2 - x1
            dy = y2 - y1
            angle = math.degrees(math.atan2(dy, dx))
            normalized_angle = abs(math.cos(math.radians(angle)))

            cv2.putText(img, f'Angle: {angle:.1f} deg', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f'AngleNorm: {normalized_angle:.2f}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            osc_client.send_message("/sax/angle", normalized_angle)
        else:
            cv2.putText(img, 'Need exactly 2 stickers', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            osc_client.send_message("/sax/angle", 0)

  
        cv2.imshow("Sticker Tracking", img)

        # Allow ESC to stop
        if cv2.waitKey(1) & 0xFF == 27:
            stop_current_stream()
            break

    stop_current_stream()

def update_ui_state():
    if loading:
        status_label.config(text="Loading...")
        start_button.config(state=tk.DISABLED)
        stop_button.config(state=tk.DISABLED)
        camera_menu.config(state=tk.DISABLED)
    elif running:
        status_label.config(text="Running")
        start_button.config(state=tk.DISABLED)
        stop_button.config(state=tk.NORMAL)
        camera_menu.config(state=tk.DISABLED)
    else:
        status_label.config(text="Stopped")
        start_button.config(state=tk.NORMAL)
        stop_button.config(state=tk.DISABLED)
        camera_menu.config(state=tk.NORMAL)

# Tkinter UI
root = tk.Tk()
root.title("Camera Selection")

camera_options = [f"Camera {i}" for i in range(10)]
camera_var = tk.StringVar(root)
camera_var.set(camera_options[0])
camera_menu = ttk.Combobox(root, textvariable=camera_var, values=camera_options)
camera_menu.pack(pady=10)

start_button = ttk.Button(root, text="Start", command=lambda: start_sticker_tracking(int(camera_var.get().split()[1])))
start_button.pack(pady=10)

stop_button = ttk.Button(root, text="Stop", command=stop_current_stream)
stop_button.pack(pady=10)

status_label = ttk.Label(root, text="Stopped")
status_label.pack(pady=10)

root.mainloop()
