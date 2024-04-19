import cv2
import datetime
from tkinter import simpledialog, messagebox
import os
import subprocess

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Ask for the user's name
user_name = simpledialog.askstring("User Name", "Enter your name:")

# Create a folder based on the user's name if it doesn't exist
folder_path = f"/Users/keshavakarthikeyan/Desktop/ASU/Spring'24/MFG598/final project/dataset/{user_name}"
os.makedirs(folder_path, exist_ok=True)

count = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the frame
    cv2.imshow('Frame', frame)

    # Check for the spacebar key
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Spacebar key
        # Save the frame as a photo
        count += 1
        current_date = datetime.date.today().strftime("%Y-%m-%d")
        file_path = f"{folder_path}/{current_date}_{count}.jpg"
        cv2.imwrite(file_path, frame)
        print(f"Photo {count} taken on {current_date}")

        # Show a pop-up message for 5 seconds
        messagebox.showinfo("Image Taken", "Image has been saved.")

    # Check for the ESC key
    elif key == 27:  # ESC key
        break

#call Face_trainer to train the pictures taken
try:
    subprocess.call(['python', 'Face_trainer.py'])
except Exception as e:
    print(f"Error: {e}")

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
# photo_taker.py
