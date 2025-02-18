import numpy as np
import math
import cv2
import matplotlib.cm as cm
import onnxruntime as ort
# import jetson_camera
import yolo_nms
import time
from collections import deque

session = ort.InferenceSession('yolov8n-pose.onnx', providers=[('TensorrtExecutionProvider', {'trt_engine_cache_enable': True, 'trt_engine_cache_path': '/home/nvidia/TRT_cache/engine_cache', "trt_fp16_enable": True, 'device_id': 0, }), 'CUDAExecutionProvider']) # providers=['CPUExecutionProvider'])#,

input_name = session.get_inputs()[0].name

def model_inference(input=None):
    output = session.run([], {input_name: input})
    return output[0]


sk = [15,13, 13,11, 16,14, 14,12, 11,12, 
            5,11, 6,12, 5,6, 5,7, 6,8, 7,9, 8,10, 
            1,2, 0,1, 0,2, 1,3, 2,4, 3,5, 4,6]
input_shape=(640, 640)

lhStr = "LeftHand"
rhStr = "RightHand"


# Queue to store past positions (length = 30 frames)
queue_len = 18
position_queue = deque(maxlen=queue_len)

# Minimum thresholds for significant movement
MIN_HEIGHT = 150  # Minimum movement for Up-Down detection
MIN_WIDTH = 150   # Minimum movement for Left-Right detection
MAX_WIDTH_THRESHOLD = 50  # Max width allowed for Up-Down / Down-Up
MAX_HEIGHT_THRESHOLD = 50  # Max height allowed for Left-Right / Right-Left

# Store last detected action & bounding box
last_action = None
last_rect = None
action_start_time = None
detected_action = None

def classify_movement(queue):
    """
    Determines the movement direction based on min/max X or Y values in the queue.
    Returns:
      "Up-Down" if moving downward
      "Down-Up" if moving upward
      "Left-Right" if moving rightward
      "Right-Left" if moving leftward
      "No Movement" if movement is insignificant
    """
    global last_action, last_rect, action_start_time, position_queue
    global MIN_HEIGHT, MIN_WIDTH, MAX_WIDTH_THRESHOLD, MAX_HEIGHT_THRESHOLD
    if len(queue) < queue_len:
        return "No Movement"

    # Extract X and Y positions from the queue
    x_values = [pos[0] for pos in queue]
    y_values = [pos[1] for pos in queue]

    # Find the min and max positions
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)

    # Get index positions of min/max values
    min_x_idx, max_x_idx = x_values.index(min_x), x_values.index(max_x)
    min_y_idx, max_y_idx = y_values.index(min_y), y_values.index(max_y)
    # Calculate bounding box size
    width = max_x - min_x
    height = max_y - min_y
    # Check for horizontal movement (Left-Right or Right-Left)
    if width > MIN_WIDTH and max_y <= MAX_HEIGHT_THRESHOLD and min_y <= MAX_HEIGHT_THRESHOLD:
        if min_x_idx < max_x_idx:
            return "Right-Left"
        else:
            return "Left-Right"

    # Check for vertical movement (Up-Down or Down-Up)
    if height > MIN_HEIGHT and width <= MAX_WIDTH_THRESHOLD:
        if min_y_idx < max_y_idx:
            return "Up-Down"
        else:
            return "Down-Up"

    return "No Movement"


def preprocess_img(frame):
    
    img = frame[:, :, ::-1]
    img = cv2.resize(img, input_shape)  # Resize to 640x640
    img = img/255.00
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img,0)
    img = img.transpose(0,3,1,2)
    return img

def single_non_max_suppression(prediction):
    argmax = np.argmax(prediction[4,:])
    x = (prediction.T)[argmax]
    
    box = x[:4] #Cx,Cy,w,h
    conf = x[4]
    keypts = x[5:]

    return box, conf, keypts

def post_process_multi(img, output, h=480, w=640, score_threshold=10):
    boxes, conf_scores, keypt_vectors = yolo_nms.non_max_suppression(output, score_threshold)

    for keypts, conf in zip(keypt_vectors, conf_scores):
        plot_keypoints(img, keypts, h, w, score_threshold)
    return img

def post_process_single(img, output, score_threshold=10):
    box, conf, keypts = single_non_max_suppression(output)
    # keypts = smooth_pred(keypts)
    plot_keypoints(img, keypts, score_threshold)
    return img

def plot_keypoints(img, keypoints, h, w, threshold=10):
    global lhStr, rhStr
    global last_action, last_rect, action_start_time, position_queue
    global MIN_HEIGHT, MIN_WIDTH, MAX_WIDTH_THRESHOLD, MAX_HEIGHT_THRESHOLD, detected_action
    MIN_HEIGHT = int(int((int(keypoints[3*12+1]) - int(keypoints[3*6+1]))*3/4)*h/640)
    MAX_WIDTH_THRESHOLD = int(((int(keypoints[3*5]) - int(keypoints[3*6]))*1/3)*w/640)
    # print(MIN_HEIGHT)
    
    MIN_WIDTH = int(int((int(keypoints[3*5]) - int(keypoints[3*6]))*4/4)*w/640)
    MAX_HEIGHT_THRESHOLD = int((int(keypoints[3*6+1]) + int((int(keypoints[3*12+1]) - int(keypoints[3*6+1]))*2/3))*h/640)
    # print(MIN_HEIGHT)
    
    for i in range(0,len(sk)//2):
        pos1 = (int(keypoints[3*sk[2*i]]), int(keypoints[3*sk[2*i]+1]))
        pos2 = (int(keypoints[3*sk[2*i+1]]), int(keypoints[3*sk[2*i+1]+1]))
        conf1 = keypoints[3*sk[2*i]+2]
        conf2 = keypoints[3*sk[2*i+1]+2]

        color = (cm.jet(i/(len(sk)//2))[:3])
        color = [int(c * 255) for c in color[::-1]]
        # if conf1>threshold and conf2>threshold: # For a limb, both the keypoint confidence must be greater than 0.5
        #     cv2.line(img, (int(pos1[0]*w/640), int(pos1[1]*h/640)), (int(pos2[0]*w/640), int(pos2[1]*h/640)), color, thickness=8)
        
    for i in range(0,len(keypoints)//3):
        x = int(keypoints[3*i])
        y = int(keypoints[3*i+1])
        x = int(x*w/640)
        y = int(y*h/640)
        conf = keypoints[3*i+2]
        pointStr = f"{i}"
        if i == 10:
            rhStr = "RightHand: ("+str(x)+", "+str(y)+")"
            cv2.circle(img, (x,y), 3, (0,0,0), -1)
            
            # Store in queue
            position_queue.append((x, y))

            # Check action classification
            detected_action = classify_movement(position_queue)
            
        # elif i == 10:
        #     pass
        else:
            pointStr = str(i)
        if conf > threshold: # Only draw the circle if confidence is above some threshold
            cv2.circle(img, (x,y), 3, (0,0,0), -1)
            cv2.putText(img, pointStr,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        
        

keypoints_old = None
def smooth_pred(keypoints):
    global keypoints_old
    if keypoints_old is None:
        keypoints_old = keypoints.copy()
        return keypoints
    
    smoothed_keypoints = []
    for i in range(0, len(keypoints), 3):
        x_keypoint = keypoints[i]
        y_keypoint = keypoints[i+1]
        conf = keypoints[i+2]
        x_keypoint_old = keypoints_old[i]
        y_keypoint_old = keypoints_old[i+1]
        conf_old = keypoints_old[i+2]
        x_smoothed = (conf * x_keypoint + conf_old * x_keypoint_old)/(conf+conf_old)
        y_smoothed = (conf * y_keypoint + conf_old * y_keypoint_old)/(conf+conf_old)
        smoothed_keypoints.extend([x_smoothed, y_smoothed, (conf+conf_old)/2])
    keypoints_old = smoothed_keypoints
    return smoothed_keypoints

if __name__== "__main__":
    # cap = jetson_camera.VideoCapture(out_width=736, out_height=480)
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam, change for external cameras
    count = 0
    nfps = 3
    while True:
        rect, frame = cap.read()
        if not rect:  # If frame is None, restart the video
            print("⚠️ Video ended or frame missing. Restarting...")
            cap.release()  # Release the video
            continue  # Skip processing this frame
        count += 1
        h, w, _ = frame.shape
        if count%nfps == 0:
            # print(count)
            input_img = preprocess_img(frame)
            output = model_inference(input_img)
            # frame = post_process_single(frame, output[0], score_threshold=5)
            frame = post_process_multi(frame, output[0], h, w, score_threshold=0.5)
        cv2.putText(frame, lhStr, (int(w/20),int(h/10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.putText(frame, rhStr, (int(w/20),int(h*2/10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        

        if last_action and (time.time() - action_start_time < 1):
            min_x, min_y, max_x, max_y = last_rect
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)  # Draw bounding box
            cv2.putText(frame, f"Action: {last_action}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            last_action = None  # Reset action after 3 seconds
            
        if detected_action != "No Movement" and position_queue:
            last_action = detected_action
            last_rect = (min([p[0] for p in position_queue]), 
                        min([p[1] for p in position_queue]), 
                        max([p[0] for p in position_queue]), 
                        max([p[1] for p in position_queue]))
            action_start_time = time.time()

            # Clear queue after recognition
            position_queue.clear()
        # frame = cv2.resize(frame, input_shape)
        cv2.imshow('out',frame)
        cv2.waitKey(1)