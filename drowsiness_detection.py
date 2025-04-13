from math import sqrt
import numpy as np
import sqlite3
from datetime import datetime
from tkinter.ttk import Label
import cv2
import dlib
from threading import Thread
from playsound import playsound
from imutils import face_utils
import tkinter as tk



#Calculate distance between two points
def euclidean(p1, p2):
    return sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

#landmarks on eyes are 0,1,2,3,4,5
#Eye Aspect Ratio = (Distance between 1 and 5 + Distance between 2 and 4) / (2 * Distance between 0 and 3)
#1 and 5 , 2 and 4 are vertical distances and 0 and 3 are horizontal
def eye_aspect_ratio(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

#In class manager we just created database and save them
class Manager:
    def __init__(self):
        self.conn = sqlite3.connect('Records.db', check_same_thread=False)  # Allow access from other threads
        #database Records is created
        self.cursor = self.conn.cursor() #Cursor
        self.cursor.execute(
            "CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, status TEXT)" #table
        )
        self.conn.commit() #save

    def save_event(self, status):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute("INSERT INTO events (timestamp, status) VALUES (?, ?)", (now, status))
        self.conn.commit()


class Alert:
    def __init__(self):
        self.sound = False #set default sound as false

    def play(self):
        try:
            playsound("sound.wav")
        except:
            print("File not found")

    def alert(self):
        if not self.sound:
            self.sound = True #sound on
            Thread(target=self.play).start()

    def reset(self):
        self.sound = False #sound off


EAR_THRESH = 0.28    #threshold = 0.28
EAR_FRAMES = 40         # No. of consecutive frames with low EAR to trigger drowsiness


class Detector:
    def __init__(self, gui):
        self.gui = gui
        self.detector = dlib.get_frontal_face_detector() #Detects face
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks (1).dat") #detects Landmarks on face
        (self.lstart, self.lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] #detects left eye
        (self.rstart, self.rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"] #detects right eye
        self.alert = Alert()
        self.manager = Manager()
        self.count = 0
        self.running = True  # Flag control the detection loop

    def detection(self):
        cap = cv2.VideoCapture(0) # 0 is for default camera of pc , if you are using external camera you can use 1 or 2
        if not cap.isOpened():
            print("Camera couldn't be opened.")
            self.gui.status_label.config(text="Camera Error", foreground="red")
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converting to gray scale for more fast and precise working
            rects = self.detector(gray, 0) # Detect faces in the grayscale frame and return rectangles
            for rect in rects:
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape) # Convert landmarks to NumPy array for easier processing

                left = shape[self.lstart:self.lend] #6 landmarks of left eye
                right = shape[self.rstart:self.rend] #6 landmarks of right eye
                left_ear = eye_aspect_ratio(left)
                right_ear = eye_aspect_ratio(right)
                ear = (left_ear + right_ear) / 2.0 #avaerage of eye aspect ratio

                if ear < EAR_THRESH:
                    self.count += 1
                    if self.count > EAR_FRAMES:
                        self.gui.status_label.config(text="Drowsy", foreground="red")
                        self.alert.alert()
                        self.manager.save_event("Drowsy")
                else:
                    self.count = 0
                    self.gui.status_label.config(text="Awake", foreground="green")
                    self.alert.reset()

            cv2.imshow("Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): #press q to close camera
                break

        cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.running = False  # Stop the detection loop


class Gui:
    def __init__(self):
        self.app = tk.Tk()
        self.app.title("Drowsiness Detector")
        self.app.geometry("600x400")
        self.app.configure(bg="white")

        Label(self.app, text="Drowsiness Detection", font=("Helvetica", 18), background="white").pack(pady=10)
        self.status_label = Label(self.app, text="Status: Not Monitoring", font=("Helvetica", 14), foreground="blue", background="white")
        self.status_label.pack(pady=20)

        tk.Button(self.app, text="Start Monitoring", font=("Helvetica", 12), command=self.start).pack(pady=10)
        tk.Button(self.app, text="Exit", font=("Helvetica", 12), command=self.app.destroy).pack(pady = 10)

    def start(self):
        self.status_label.config(text="Monitoring", foreground="orange")
        detector = Detector(self)
        Thread(target=detector.detection).start()

    def run(self):
        self.app.mainloop()

if __name__ == "__main__":
    app = Gui()
    app.app.mainloop()