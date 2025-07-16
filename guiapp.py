import tkinter as tk
from tkinter import Entry, Button, Label
import cv2
import math
import requests
from ultralytics import YOLO
from PIL import Image, ImageTk
import datetime
import pygame

model = YOLO('best.pt')
pygame.mixer.init()

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

classNames = ['Grenade', 'Knife', 'Pistol', 'Rifle', 'Shotgun']

is_detecting = False

# Variables to track the last detected class and confidence
last_detected_class = None
last_confidence = 0.0

def toggle_detection():
    global is_detecting
    is_detecting = not is_detecting
    update_button_state()
    if is_detecting:
        detect_objects()  # Start detection when toggled on

def update_button_state():
    toggle_button.config(
        text="Stop Detection" if is_detecting else "Start Detection",
        bg="#d32f2f" if is_detecting else "#4caf50",
    )

def play_audio(audio_file):
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()

def detect_objects():
    global last_detected_class, last_confidence
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        return

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            # Check if the detection is significant and different from the last one
            if confidence >= 0.75 and (last_detected_class != cls or confidence > last_confidence):
                play_audio("alarm.mp3")
                print("Confidence --->", confidence)
                print("Class name -->", classNames[cls])

                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, classNames[cls] + str(confidence), org, font, fontScale, color, thickness)

                # Update last detected class and confidence
                last_detected_class = cls
                last_confidence = confidence

    update_image(img)  # Update the image in the label
    if is_detecting:
        label.after(100, detect_objects)  # Schedule the next detection

def update_image(img):
    cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    photo = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=photo)
    label.imgtk = imgtk
    label.config(image=imgtk)

app = tk.Tk()
app.title("Weapon Detection")
app.geometry("600x540")
app.configure(bg="#212121")

label = tk.Label(app, bg="#212121")
label.pack(fill=tk.BOTH, expand=True)

toggle_button = tk.Button(
    app,
    text="Start Detection",
    command=toggle_detection,
    bg="#4caf50",
    fg="white",
    borderwidth=0,
)

toggle_button.pack(pady=5)

label.after(10, detect_objects)

app.mainloop()
