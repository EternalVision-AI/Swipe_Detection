import numpy as np
import math
import cv2
import matplotlib.cm as cm
import onnxruntime as ort
# import jetson_camera
import yolo_nms
import time
from collections import deque
import threading
import asyncio
import websockets
import json

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

PEACE_TIME = 0.7
SIMILAR_THRESHOLD=37
# Queue to store past positions
QUEUE_LEN = 20
right_position_queue = deque(maxlen=QUEUE_LEN)
left_position_queue = deque(maxlen=QUEUE_LEN)

# Minimum thresholds for significant movement
MIN_HEIGHT = 0  # Minimum movement for Up-Down detection
MIN_WIDTH = 0   # Minimum movement for Left-Right detection
MAX_WIDTH_THRESHOLD = 0  # Max width allowed for Up-Down / Down-Up
MAX_HEIGHT_THRESHOLD = 0  # Max height allowed for Left-Right / Right-Left

# Store last detected action & bounding box
rh_last_action = None
lh_last_action = None
rh_last_rect = None
lh_last_rect = None
rh_detected_time = time.time()
lh_detected_time = time.time()
rh_detected_action = None
lh_detected_action = None


def classify_movement(queue):
    """
    Determines the movement direction based on min/max X or Y values in the queue.
    Returns:
      "Up-Down" if moving downward "swiping down!"
      "Down-Up" if moving upward "swiping up!"
      "Left-Right" if moving rightward "swiping right!"
      "Right-Left" if moving leftward "swiping left!"
      "None" if movement is insignificant "idle
    """
    if len(queue) < (QUEUE_LEN//3)*2:
        return None

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
    if (width > MIN_WIDTH and min_y >= MIN_HEIGHT_THRESHOLD) or (width > MIN_WIDTH and height >= (MAX_HEIGHT_THRESHOLD - MIN_HEIGHT_THRESHOLD)//2):
        if min_x_idx < max_x_idx:
            return "swiping left!"
        else:
            return "swiping right!"

    # Check for vertical movement (Up-Down or Down-Up)
    if (height > MIN_HEIGHT//2 and width <= MAX_WIDTH_THRESHOLD):
        if min_y_idx < max_y_idx:
            return "swiping down!"
        else:
            return "swiping up!"

    return None

def is_stop_current_position(queue, point):
    """ Checks if there are more than three points in the queue similar to the given point within a specified SIMILAR_THRESHOLD.
    Args:
        queue (list of tuples): The queue of points.
        point (tuple): The point to check.
        SIMILAR_THRESHOLD (int): The radius within which points are considered similar.
        
    Returns:
        bool: True if there are at least three points similar to the given point, False otherwise.
    """
    similar_count = 0  # Initialize counter for similar points

    # Iterate over each point in the queue
    for existing_point in queue:
        # Calculate the difference between the existing points and the new point
        x_diff = abs(existing_point[0] - point[0])
        y_diff = abs(existing_point[1] - point[1])

        # Check if the differences are within the specified SIMILAR_THRESHOLD
        if x_diff <= SIMILAR_THRESHOLD and y_diff <= SIMILAR_THRESHOLD:
            similar_count += 1  # Increment the counter for similar points

        # If three similar points are found, return True
        if similar_count >= 3:
            return True

    # Append the new point to the queue if no early exit happened
    queue.append(point)

    # If less than three similar points are found, return False
    return False


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
    global rh_last_action, lh_last_action, rh_detected_time, lh_detected_time, right_position_queue, left_position_queue
    global MIN_HEIGHT, MIN_WIDTH, MAX_WIDTH_THRESHOLD, MAX_HEIGHT_THRESHOLD, MIN_HEIGHT_THRESHOLD, rh_detected_action, lh_detected_action
    
    MIN_HEIGHT = int(int((int(keypoints[3*12+1]) - int(keypoints[3*6+1]))*1/3)*h/640)
    MAX_WIDTH_THRESHOLD = int(((int(keypoints[3*5]) - int(keypoints[3*6]))*1/3)*w/640)
    # print(MIN_HEIGHT)
    
    MIN_WIDTH = int(int((int(keypoints[3*5]) - int(keypoints[3*6]))*2/3)*w/640)
    MAX_HEIGHT_THRESHOLD = int(int(keypoints[3*12+1])*h/640)
    MIN_HEIGHT_THRESHOLD = int(int(keypoints[3*6+1])*h/640)
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
        conf_threshold = 0.3
        if i == 10:
            cv2.circle(img, (x,y), 3, (0,0,0), -1)
            # if (time.time() - rh_detected_time > 1):
            # Store in queue
            # right_position_queue.append((x, y))
            if is_stop_current_position(right_position_queue, (x,y)):
                print("New Position")
                right_position_queue.clear()
            else:
                right_position_queue.append((x, y))
                
            # Check action classification
            rh_detected_action = classify_movement(right_position_queue)
            
            if rh_detected_action == None:
                rhStr = f"RightHand: {rh_last_action}"
            else:
                rh_detected_time = time.time()
                rh_last_action = rh_detected_action
                print(rh_detected_action)
                right_position_queue.clear()
                rhStr = f"RightHand: {rh_detected_action}"   
        if keypoints[3*10+2] < conf_threshold and len(right_position_queue) < QUEUE_LEN/2:
            right_position_queue.clear()
        if i == 9:
            cv2.circle(img, (x,y), 3, (0,0,0), -1)
            # if (time.time() - lh_detected_time > 1):
            # Store in queue
            # left_position_queue.append((x, y))
            if is_stop_current_position(left_position_queue, (x,y)):
                print("New Position")
                left_position_queue.clear()
            left_position_queue.append((x, y))
            # Check action classification
            lh_detected_action = classify_movement(left_position_queue)
            if lh_detected_action == None:
                lhStr = f"LeftHand: {lh_last_action}"
            else:
                lh_detected_time = time.time()
                lh_last_action = lh_detected_action
                print(lh_detected_action)
                left_position_queue.clear()
                lhStr = f"LeftHand: {lh_detected_action}"
        if keypoints[3*9+2] < conf_threshold and len(left_position_queue) < QUEUE_LEN/2:
            left_position_queue.clear()
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


# WebSocket server address
SERVER_URI = "ws://127.0.0.1:8765"

async def send_data_to_server(data):
    """Connect to the WebSocket server and send the processed data."""
    async with websockets.connect(SERVER_URI) as websocket:
        message = json.dumps(data)
        await websocket.send(message)
        print(f"Sent to server: {message}")


async def process_camera_feed():
    # cap = jetson_camera.VideoCapture(out_width=736, out_height=480)
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam, change for external cameras
    # cap = cv2.VideoCapture('(2).mp4')  # Use 0 for default webcam, change for external cameras
    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get FPS from the input video

    # Define video writer to save output
    # output_filename = "output.mp4"
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    # out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    count = 0
    nfps = 3
    while True:
        rect, frame = cap.read()
        if not rect:  # If frame is None, restart the video
            print("⚠️ Video ended or frame missing. Restarting...")
            # cap.release()  # Release the video
            break  # Skip processing this frame
        count += 1
        h, w, _ = frame.shape
        if count%nfps == 0:
            # print(count)
            input_img = preprocess_img(frame)
            output = model_inference(input_img)
            # frame = post_process_single(frame, output[0], score_threshold=5)
            frame = post_process_multi(frame, output[0], h, w, score_threshold=0.7)
            if rh_detected_action:
                # Send the processed data to the WebSocket server
                await send_data_to_server(rh_detected_action)
            if lh_detected_action:
                # Send the processed data to the WebSocket server
                await send_data_to_server(lh_detected_action)
        cv2.putText(frame, lhStr, (int(w/20),int(h/10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.putText(frame, rhStr, (int(w/20),int(h*2/10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
            
        if (time.time() - rh_detected_time < PEACE_TIME):
            # Clear queue after recognition
            right_position_queue.clear()
            left_position_queue.clear()
        
        if right_position_queue:
            # Extract X and Y positions from the queue
            x_values = [pos[0] for pos in right_position_queue]
            y_values = [pos[1] for pos in right_position_queue]

            # Find the min and max positions
            min_x, max_x = min(x_values), max(x_values)
            min_y, max_y = min(y_values), max(y_values)

            # Get index positions of min/max values
            min_x_idx, max_x_idx = x_values.index(min_x), x_values.index(max_x)
            min_y_idx, max_y_idx = y_values.index(min_y), y_values.index(max_y)
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)  # Draw bounding box
            cv2.putText(frame, f"MIN_index({min_x_idx})", (min_x, min_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, f"MAX_index({max_x_idx})", (max_x, max_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 5, cv2.LINE_AA)
                
        if (time.time() - lh_detected_time < PEACE_TIME):
            # Clear queue after recognition
            right_position_queue.clear()
            left_position_queue.clear()
            
        if left_position_queue:
            # Extract X and Y positions from the queue
            x_values = [pos[0] for pos in left_position_queue]
            y_values = [pos[1] for pos in left_position_queue]

            # Find the min and max positions
            min_x, max_x = min(x_values), max(x_values)
            min_y, max_y = min(y_values), max(y_values)

            # Get index positions of min/max values
            min_x_idx, max_x_idx = x_values.index(min_x), x_values.index(max_x)
            min_y_idx, max_y_idx = y_values.index(min_y), y_values.index(max_y)
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)  # Draw bounding box
            cv2.putText(frame, f"MIN_index({min_x_idx})", (min_x, min_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, cv2.LINE_AA)
            cv2.putText(frame, f"MAX_index({max_x_idx})", (max_x, max_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, cv2.LINE_AA)

        cv2.imshow('out',cv2.resize(frame, None, fx=0.5, fy=0.5))
        # cv2.imshow('out',frame)
        cv2.waitKey(1)
    # Release resources
    cap.release()
    # out.release()
    cv2.destroyAllWindows()
    
    
    
async def main():
    """Start the camera feed processing and send data."""
    await process_camera_feed()

if __name__ == "__main__":
    asyncio.run(main())