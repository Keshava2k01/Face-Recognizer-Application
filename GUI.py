from tkinter import Tk, Label, Button, font
from PIL import Image, ImageTk
import subprocess
import tensorflow

def photo_taker():
    subprocess.Popen(['python', 'photo_taker.py'])

def face_compare():
    subprocess.Popen(['python', 'Face_recognizer.py'])

# Main
root = Tk()
root.title("Face Recognizer Application")
root.geometry('800x600')  # Set a fixed window size

# Load the image
image_path = "1.jpg"
img = Image.open(image_path)
photo = ImageTk.PhotoImage(img)

# Display the image in a label
label = Label(root, image=photo)
label.pack()

# Text labels
custom_font = font.Font(size=20)
text_label = Label(root, text="Face Recognizer Application!!", font=custom_font)
text_label.pack(pady=10)

bottom_right_text = Label(root, text="Made by Keshava Karthikeyan", font=custom_font, anchor='se')
bottom_right_text.pack(side='bottom', padx=20, pady=20)

# Buttons for different actions
B = Button(root, text="Take Photo", command=photo_taker, width=20, height=2)
B.pack(pady=10)
B.place(x=100, y=500)

C = Button(root, text="Compare Faces", command=face_compare, width=20, height=2)
C.pack(pady=10)
C.place(x=450, y=500)

root.mainloop()
