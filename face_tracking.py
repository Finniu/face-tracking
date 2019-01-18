import dlib
import cv2
import face_recognition
from sort import Sort
import numpy as np

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

#Create the tracker we will use

#The variable we use to keep track of the fact whether we are
#currently using the dlib tracker
# If we are not tracking a face, then try to detect one

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
tracker = Sort()
color=(0,0,255)
count = 0
ids = []

while True:

    det = []

    # Grab a single frame of video
    ret, frame = video_capture.read()
    if ret:

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1] #rgb_small_frame (240,320,3)

        # Only process every other frame of video to save time

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)

        # face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        # predict=[]

        for face_location in face_locations:
            y1, x2, y2, x1=face_location[0]*4,face_location[1]*4,face_location[2]*4,face_location[3]*4 #上右下左
            det.append((x1, y1, x2, y2,1))

        #左右上下 sort
        predict=tracker.update(np.array(det))
        import random

        for pre in predict:

            x1, y1, x2, y2,id=int(pre[0]),int(pre[1]),int(pre[2]),int(pre[3]),int(pre[4])
            if count>1:
                if id not in ids:
                    ids.append(id)
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


            cv2.rectangle(frame,(x1,y1) ,(x2,y2), color, 2)


    count+=1

    # Display the resulting image
    frame=cv2.resize(frame,(600,600))
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

