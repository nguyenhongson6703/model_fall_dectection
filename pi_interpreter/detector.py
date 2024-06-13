import argparse
import sys
import time
import datetime
import pytz

import cv2
import mediapipe as mp
import numpy as np


import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

from picamera2 import Picamera2

from concurrent.futures import ThreadPoolExecutor
from moviepy.editor import ImageSequenceClip
from collections import deque

###################
import comms as pblsv
import pi_beeper as beeper

pblsv.login('xxx', 'xxxx')
beeper.beep(0.2)

video_upload_ticket = ''
###################
import subprocess

rtmp_url = 'rtmp://103.68.251.32/live'

# Define the codec and initialize the video writer
command = ['ffmpeg',
           '-y',  # overwrite output file if it exists
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-pix_fmt', 'rgb24',  # format
           '-s', '640x480',
           '-r', '30', 
           '-i', '-',  # The input comes from a pipe
           '-c:v', 'libx264',
           '-g', '30',  
           '-pix_fmt', 'yuv420p',
           '-preset', 'ultrafast',
           '-f', 'flv',
           #'-tag:v', 'h264',
           '-nostats',
           rtmp_url]
           
timezone = pytz.timezone('Etc/GMT-7')
####################
process = subprocess.Popen(command, stdin=subprocess.PIPE)


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

CAMERA_FPS = 30
# Global variables to calculate FPS
COUNTER, FPS = 0, 0
FRAME_COUNTER = 0
START_TIME = time.time()
DETECTION_RESULT = None
executor = None

# detection
lm_list = []
label = "Please wait"
input_details = []
output_details = []

# video buffer
pre_record_time = 10  # Seconds to pre-record
post_record_time = 5  # Seconds to record after key press
frame_buffer = deque(maxlen=CAMERA_FPS * (pre_record_time + post_record_time))  # Holds the last 15 seconds of footage

telegram_bot = None

def make_landmark_timestep(results):
    c_lm = []
    def add_lanmark(index):
        landmark = results.pose_landmarks[0][index]
        c_lm.append(landmark.x)
        c_lm.append(landmark.y)
        c_lm.append(landmark.z)
        c_lm.append(landmark.visibility)
       
    for i in range(33):
        add_lanmark(i)   
    '''
    add_lanmark(0)
    add_lanmark(11)
    add_lanmark(12)
    add_lanmark(13)
    add_lanmark(14)
    add_lanmark(15)
    add_lanmark(16)
    add_lanmark(23)
    add_lanmark(24)
    add_lanmark(25)
    add_lanmark(26)
    add_lanmark(27)
    add_lanmark(28)
    '''

    return c_lm

def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 200)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

so_seq_fall_lien_tiep = 0
gui_thong_bao_sau_te_nga = False
lastFallDetectedTime = None
turn_on_beeper_count = 0
video_record_executor = None
beeper_executor = None

        
def detect(interpreter, lm_list):
    global label, input_details, output_details, so_seq_fall_lien_tiep, gui_thong_bao_sau_te_nga, lastFallDetectedTime, video_record_executor, video_upload_ticket, turn_on_beeper_count
    lm_list = np.array(lm_list, dtype=np.float32)
    lm_list = np.expand_dims(lm_list, axis=0)
    #print(lm_list.shape)

    interpreter.set_tensor(input_details[0]['index'], lm_list)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    if output_data[0][0] > 0.5:
        label = "FALL"
        so_seq_fall_lien_tiep += 1
        if so_seq_fall_lien_tiep >= 3:
            if lastFallDetectedTime is None or (datetime.datetime.now() - lastFallDetectedTime).total_seconds() > 15:
                gui_thong_bao_sau_te_nga = True
                print("flag triggered ", so_seq_fall_lien_tiep)
                lastFallDetectedTime = datetime.datetime.now()
            print("Threshold ", so_seq_fall_lien_tiep)
    else:
        label = "NOT FALL"
        if gui_thong_bao_sau_te_nga == True and so_seq_fall_lien_tiep < 50:
            gui_thong_bao_sau_te_nga = False


            # do alot, like record video and ...
            print("Fall detected. Sending alarm ", so_seq_fall_lien_tiep)
            print("Recording additional 5 seconds")
            
            turn_on_beeper_count, need_video_upload, video_upload_ticket = pblsv.report_fall_detected(so_seq_fall_lien_tiep)
            
            if turn_on_beeper_count > 0:
                beeper_executor.submit(threaded_sos_beeper)
            if need_video_upload:
                video_record_executor.submit(threaded_record_video_and_save)              
        so_seq_fall_lien_tiep = 0
    return label   

def threaded_detect(interpreter, lm_data):
    global label
    label = detect(interpreter, lm_data)
    
def threaded_record_video_and_save():
    global video_upload_ticket
    
    time.sleep(post_record_time)
    frames = list(frame_buffer)
    clip = ImageSequenceClip(frames, fps=CAMERA_FPS)
    clip.write_videofile("output.mp4", codec="libx264")
    pblsv.upload_video('output.mp4', video_upload_ticket)
    
def threaded_sos_beeper():
    for _ in range(turn_on_beeper_count):
        beeper.beep_sos()

def draw_datetime_to_frame(frame):
    current_time = datetime.datetime.now(pytz.utc).astimezone(timezone).strftime('%d-%m-%Y %H:%M:%S')

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)
    font_thickness = 1

    (text_width, text_height), _ = cv2.getTextSize(current_time, font, font_scale, font_thickness)

    top_left_corner_x = 0
    top_left_corner_y = 0

    bottom_right_corner_x = top_left_corner_x + text_width + 4
    bottom_right_corner_y = top_left_corner_y + text_height + 4

    cv2.rectangle(frame, (top_left_corner_x, top_left_corner_y), (bottom_right_corner_x, bottom_right_corner_y), (0, 0, 0), -1)

    text_x = top_left_corner_x + 2
    text_y = bottom_right_corner_y - 2

    cv2.putText(frame, current_time, (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    return frame
    
def run(model: str) -> None:
    global n_time_steps, input_details, output_details, video_record_executor, beeper_executor
    
    n_time_steps = 20
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    
    executor = ThreadPoolExecutor(max_workers=1)
    video_record_executor = ThreadPoolExecutor(max_workers=1)
    beeper_executor = ThreadPoolExecutor(max_workers=1)
    
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)}, controls={'FrameRate': CAMERA_FPS}))
    picam2.start()

    row_size = 50  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 0)  # black
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10
    overlay_alpha = 0.5
    mask_color = (100, 100, 0)  # cyan

    def process_result(result: vision.PoseLandmarkerResult,
                    unused_output_image: mp.Image, timestamp_ms: int):
        global FPS, COUNTER, START_TIME, DETECTION_RESULT

        # Calculate the FPS
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        DETECTION_RESULT = result
        COUNTER += 1
        
        if result.pose_landmarks:
            c_lm = make_landmark_timestep(result)
            lm_list.append(c_lm)
            if len(lm_list) >= n_time_steps:
                lm_data_to_predict = lm_list[-n_time_steps:]
                executor.submit(threaded_detect, interpreter, lm_data_to_predict)
                #label = detect(interpreter, lm_data_to_predict)
                lm_list.pop(0)

    # Initialize the pose landmarker model
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_poses=1,
        min_pose_detection_confidence=0.8,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
        result_callback=process_result)
    detector = vision.PoseLandmarker.create_from_options(options)

    while True:
        frame = picam2.capture_array()                  
        # Run pose landmarker using the model.
        global FRAME_COUNTER
        
        FRAME_COUNTER += 1
        if FRAME_COUNTER % 2 == 0: 
            in_rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            frame_for_detection = cv2.resize(in_rgb_image, (320, 240))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_for_detection)
            detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        camera_feed_frame = draw_datetime_to_frame(frame)
        
        # Show the FPS
        fps_text = 'FPS = {:.1f}'.format(FPS)
        text_location = (left_margin, row_size)
        cv2.putText(camera_feed_frame, fps_text, text_location,
                    cv2.FONT_HERSHEY_DUPLEX,
                    font_size, text_color, font_thickness, cv2.LINE_AA)
                    
        camera_feed_frame = draw_class_on_image(label, camera_feed_frame)
        
        if DETECTION_RESULT:
            # Draw landmarks.
            for pose_landmarks in DETECTION_RESULT.pose_landmarks:
                # Draw the pose landmarks.
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y,
                                                    z=landmark.z) for landmark
                    in pose_landmarks
                ])
                mp_drawing.draw_landmarks(
                    camera_feed_frame,
                    pose_landmarks_proto,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing_styles.get_default_pose_landmarks_style())
        
        rgb_image = cv2.cvtColor(camera_feed_frame, cv2.COLOR_BGR2RGB)              
        frame_buffer.append(rgb_image)
        process.stdin.write(rgb_image.tostring())
        
        # opencv hoat dong tren nen bgr, nen phai conver trc khi call imshow
        #cv2.imshow('pose_landmarker', camera_feed_frame)
        
        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

    detector.close()
    picam2.close()
    process.stdin.close()
    process.wait()
    cv2.destroyAllWindows()


def main():
    global telegram_bot

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Name of the pose landmarker model bundle.',
        required=False,
        default='pose_landmarker_full.task')

    args = parser.parse_args()
    run(args.model)

if __name__ == '__main__':
    main()
