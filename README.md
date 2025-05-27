# Face Detection using Haar Cascades with OpenCV and Matplotlib

## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows  
## PROGRAM:
```
import numpy as np
import cv2 
import matplotlib.pyplot as plt
%matplotlib inline

model = cv2.imread('mypic.png',0)
withglass = cv2.imread('person.png',0)
group = cv2.imread('group.jpeg',0)

plt.imshow(model,cmap='gray')
plt.show()

plt.imshow(withglass,cmap='gray')
plt.show()

plt.imshow(group,cmap='gray')
plt.show()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_face(img):
    
  
    face_img = img.copy()
  
    face_rects = face_cascade.detectMultiScale(face_img) 
    
    for (x,y,w,h) in face_rects: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10) 
        
    return face_img


result = detect_face(withglass)
plt.imshow(result,cmap='gray')
plt.show()

result = detect_face(group)
plt.imshow(result,cmap='gray')
plt.show()

# Gets errors!
result = detect_face(group)
plt.imshow(result,cmap='gray')
plt.show()

def adj_detect_face(img):
    
    face_img = img.copy()
  
    face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.2, minNeighbors=5) 
    
    for (x,y,w,h) in face_rects: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10) 
        
    return face_img

# Doesn't detect the side face.
result = adj_detect_face(group)
plt.imshow(result,cmap='gray')
plt.show()

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect_eyes(img):
    
    face_img = img.copy()
  
    eyes = eye_cascade.detectMultiScale(face_img) 
    
    
    for (x,y,w,h) in eyes: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10) 
        
    return face_img


result = detect_eyes(model)
plt.imshow(result,cmap='gray')
plt.show()

eyes = eye_cascade.detectMultiScale(withglass)

# White around the pupils is not distinct enough to detect Denis' eyes here!
result = detect_eyes(withglass)
plt.imshow(result,cmap='gray')
plt.show()

cap = cv2.VideoCapture(0)

# Set up matplotlib
plt.ion()
fig, ax = plt.subplots()

ret, frame = cap.read(0)
frame = detect_face(frame)
im = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title('Video Face Detection')

while True:
    ret, frame = cap.read(0)

    frame = detect_face(frame)

    # Update matplotlib image
    im.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.pause(0.10)

   

cap.release()
plt.close()


```
## OUTPUT:
## I) INPUT IMAGES:

![image](https://github.com/user-attachments/assets/b2290352-cc7f-4674-9664-dfc3a7185e81)

![image](https://github.com/user-attachments/assets/e14f345d-0509-44fb-bd87-a06687881c98)

![image](https://github.com/user-attachments/assets/65eb2c01-17e3-4cc3-a653-23fa3bebf6e4)

## II) FACE DETECTION:
![image](https://github.com/user-attachments/assets/3974bd90-088b-4a41-a2fd-c231be1ca0ed)

![image](https://github.com/user-attachments/assets/6e4eb118-cc07-459b-a7d7-dee05b69e37a)

## III) EYE DETECTION:
![image](https://github.com/user-attachments/assets/711a2938-6a3b-4ada-877a-75c808ccaa44)

![image](https://github.com/user-attachments/assets/75695956-a0af-48ea-92d0-51ecf7ec4367)

## IV) VIDEO FACE DETECTION:
![image](https://github.com/user-attachments/assets/0315ec6e-87fa-4230-82f8-9f71124e83d6)


## RESULT:
thus the given objective of face detection is done sucessfully.
