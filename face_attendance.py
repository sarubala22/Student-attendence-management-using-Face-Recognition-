import cv2
import face_recognition 
import numpy as np
import os
import pandas as pd
from datetime import datetime

# Path to store student images
path = 'student_images'
images = []
student_names = []

# Load student images and their names
def load_student_images():
    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(f'{path}/{filename}')
            images.append(img)
            student_names.append(os.path.splitext(filename)[0])
    print(f"Loaded {len(images)} student images from {path}.")


# Encode faces from the loaded images
def encode_faces(images):
    encoded_faces = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img)[0]
            encoded_faces.append(encode)
        except IndexError as e:
            print(f"Face not detected in image: {img}")
    return encoded_faces


# Mark attendance in a CSV file
def mark_attendance(name):
    attendance_file = 'attendance.csv'
    current_time = datetime.now().strftime('%H:%M:%S')
    current_date = datetime.now().strftime('%Y-%m-%d')

    # Check if CSV file exists, if not create one
    if not os.path.isfile(attendance_file):
        df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
        df.to_csv(attendance_file, index=False)

    # Read the existing attendance data
    df = pd.read_csv(attendance_file)

    # Check if the student has already been marked present for today
    if not ((df['Name'] == name) & (df['Date'] == current_date)).any():
        new_entry = {'Name': name, 'Date': current_date, 'Time': current_time}
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        df.to_csv(attendance_file, index=False)
        print(f"Marked attendance for {name} at {current_time} on {current_date}.")
    else:
        print(f"{name} is already marked present for today.")


# Main function for face recognition and attendance
def recognize_faces(known_encodings, student_names):
    cap = cv2.VideoCapture(0)  # Open webcam
    print("Starting webcam. Press 'q' to exit.")
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame. Exiting.")
            break

        # Reduce frame size to speed up processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect and encode faces in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare faces with known encodings
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            # Get best match index
            match_index = np.argmin(face_distances) if len(face_distances) > 0 else None
            
            if match_index is not None and matches[match_index]:
                name = student_names[match_index].upper()
                
                # Draw rectangle around the face
                top, right, bottom, left = face_location
                top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                # Mark attendance
                mark_attendance(name)
        
        cv2.imshow('Face Recognition Attendance System', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Loading student images...")
    load_student_images()
    
    print("Encoding student faces...")
    known_face_encodings = encode_faces(images)
    
    print("Starting face recognition...")
    recognize_faces(known_face_encodings, student_names)
