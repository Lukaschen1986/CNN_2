# -*- coding: utf-8 -*-
import face_recognition as fr
import cv2
from PIL import Image

# get face in picture
filename = "sss" + ".jpg"
image = fr.load_image_file(filename)
face_locations = fr.face_locations(image, model="hog")
face_location = face_locations[0]
top, right, bottom, left = face_location
face_image = image[top:bottom, left:right]
pil_image = Image.fromarray(face_image)
pil_image.save(filename)

# find face in video
cwd_image = fr.load_image_file("cwd.jpg")
cwd_face = fr.face_encodings(cwd_image)[0]
sss_image = fr.load_image_file("sss.jpg")
sss_face = fr.face_encodings(sss_image)[0]

video_capture = cv2.VideoCapture(0)

face_locs = []
face_objs = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locs = fr.face_locations(small_frame) # 检测视频中的人脸
        face_objs = fr.face_encodings(small_frame, known_face_locations=face_locs) # 编码

        face_names = []
        for face_obj in face_objs:
            # See if the face is a match for the known face(s)
            match = fr.compare_faces([cwd_face], face_obj, tolerance=0.6)
            if match[0]: name = "cwd"
#            if match[1]: name = "史珊珊"
            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locs, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
