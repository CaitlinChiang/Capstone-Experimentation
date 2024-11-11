import os
import cv2

# Define paths explicitly
prototxt_path = os.path.join(os.path.dirname(__file__), 'deploy.prototxt')
model_path = os.path.join(os.path.dirname(__file__), 'mobilenet_iter_73000.caffemodel')

# Check if files exist
assert os.path.isfile(prototxt_path), f"Prototxt file not found at {prototxt_path}"
assert os.path.isfile(model_path), f"Model file not found at {model_path}"

# Load the model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)


# Define the 'person' class ID in MobileNet-SSD (index 15)
person_class_id = 15

# Set up the camera (use 0 for built-in webcam)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the frame for detection
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Pass the blob through the network and get detections
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter for high-confidence person detections
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            if class_id == person_class_id:
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (startX, startY, endX, endY) = box.astype("int")
                
                # Draw the bounding box with confidence level
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                text = f"Person: {confidence:.2f}"
                cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Real-Time Human Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
