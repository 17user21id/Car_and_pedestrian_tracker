import cv2


# add any video file here
video_file = cv2.VideoCapture('')


# PreTrained classifier
car_tracker = cv2.CascadeClassifier('car_detector.xml')
pedestrian_tracker = cv2.CascadeClassifier('haarcascade_fullbody.xml')


while True:
    # Reading Video
    (read_successful, frame) = video_file.read()
    if read_successful:
        # Converting to greyscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Detect Cars & Pedestrian
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    # Draw square around the car
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+1), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Draw square around the pedestrian
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    # Display the image
    cv2.imshow('Car & Pedestrian Detector', frame)

    # it will close when you give some input
    key = cv2.waitKey(1)

    # Stop if Q key is pressed
    if key == 81 or key == 113:
        break

# Release the videoCapture object
video_file.release()

print('Code Completed')