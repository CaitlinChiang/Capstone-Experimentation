import cv2
import numpy as np
import time
import gc

# Load YOLOv4-tiny model
weights_path = 'human-detection/yolov4-tiny.weights'
config_path = 'human-detection/yolov4-tiny.cfg'
names_path = 'human-detection/coco.names'

# Load class names
with open(names_path, 'r') as f:
    class_names = f.read().strip().split('\n')

# Initialize YOLO
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Layer names
layer_names = net.getLayerNames()
unconnected_layers = net.getUnconnectedOutLayers()
output_layers = [layer_names[int(i) - 1] for i in unconnected_layers.flatten()]

# Open camera and configure settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Timer and frame processing setup
start_time = time.time()
duration = 60  # 1 minute
frame_count = 0
confidence_threshold = 0.7
nms_threshold = 0.5
process_every_n_frames = 1  # Process every 5th frame
input_size = (256, 256)  # Reduce input size for YOLO

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Continuing.")
        continue  # Skip to the next loop iteration if frame capture fails

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    if elapsed_time > duration:
        print("Time is up. Closing camera feed.")
        break  # Stop the loop after 1 minute

    frame_count += 1
    if frame_count % process_every_n_frames != 0:
        del frame
        continue

    height, width = frame.shape[:2]
    
    # Prepare the image for YOLO with reduced input size
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, input_size, swapRB=True, crop=False)
    net.setInput(blob)
    
    try:
        outputs = net.forward(output_layers)
    except cv2.error as e:
        print(f"Failed to process frame with error: {e}. Skipping frame.")
        continue  # If YOLO fails on a frame, skip it

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > confidence_threshold and class_names[class_id] == 'person':
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=confidence_threshold, nms_threshold=nms_threshold)
    
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        if w * h < 5000:  # Skip small bounding boxes (likely false positives)
            continue
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("Human Detection", frame)

    # Clear lists after each frame processing to free memory
    boxes.clear()
    confidences.clear()
    class_ids.clear()
    del blob, outputs  # Free intermediate variables

    if frame_count % 10 == 0:
        gc.collect()  # Periodic garbage collection
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
