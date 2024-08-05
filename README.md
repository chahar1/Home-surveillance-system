# Home-surveillance-system
Created by Shoryan Chahar.


Sure! Here is a comprehensive documentation for the provided script:

## Face Detection and Video Creation Script

This script captures video from the webcam, detects faces in real-time, saves images with detected faces, and compiles those images into a video when the user quits the application.

### Prerequisites
- Python 3.x
- OpenCV library
- Haarcascades XML file for face detection (`haarcascade_frontalface_default.xml`)

### Installation
Make sure you have the required packages installed:
```bash
pip install opencv-python
```

### Script Breakdown

#### 1. Import Libraries
```python
import cv2
import time
from datetime import datetime
import argparse
import os
```

#### 2. Load Haar Cascade Classifier
```python
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
```

#### 3. Initialize Video Capture
```python
video = cv2.VideoCapture(0)
```

#### 4. Real-time Face Detection Loop
The script enters a loop to read frames from the webcam:
```python
while True:
    check, frame = video.read()
    if frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
        for x, y, w, h in faces:
            img = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            exact_time = datetime.now().strftime('%Y-%b-%d-%H-%S-%f')
            cv2.imwrite("face detected " + str(exact_time) + ".jpg", img)

        cv2.imshow("home surv", frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break
```

#### 5. Argument Parsing for Video Creation
When the user quits (presses 'q'), the script parses command-line arguments for the video creation process:
```python
ap = argparse.ArgumentParser()
ap.add_argument("-ext", "--extension", required=False, default='jpg')
ap.add_argument("-o", "--output", required=False, default='output.mp4')
args = vars(ap.parse_args())
```

#### 6. Collect Images and Create Video
The script collects images saved during the face detection phase and compiles them into a video:
```python
dir_path = '.'
ext = args['extension']
output = args['output']

images = []
for f in os.listdir(dir_path):
    if f.endswith(ext):
        images.append(f)

image_path = os.path.join(dir_path, images[0])
frame = cv2.imread(image_path)
height, width, channels = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output, fourcc, 5.0, (width, height))

for image in images:
    image_path = os.path.join(dir_path, image)
    frame = cv2.imread(image_path)
    out.write(frame)

out.release()
```

#### 7. Release Video Capture and Destroy Windows
```python
video.release()
cv2.destroyAllWindows()
```

### Usage
Run the script in your terminal or command prompt. The script will start capturing video from your webcam, detect faces, and save the images with detected faces. Press 'q' to stop capturing and create a video from the saved images.

```bash
python script_name.py
```

### Notes
- Ensure the `haarcascade_frontalface_default.xml` file is in the same directory as the script or provide the correct path.
- The default output video format is `.mp4`, but you can change it using the `--output` argument.

### Example
```bash
python script_name.py --extension jpg --output output_video.mp4
```

This documentation should help you understand and use the script effectively.
