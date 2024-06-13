from picamera2 import Picamera2
from moviepy.editor import ImageSequenceClip
import cv2
import time

def run():
    CAMERA_FPS = 30  
    VIDEO_DURATION = 15 
    VIDEO_FILENAME = "recorded.mp4"


    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)}, controls={'FrameRate': CAMERA_FPS}))
    picam2.start()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(VIDEO_FILENAME, fourcc, CAMERA_FPS, (640, 480))
    
    start_time = time.time()

    try:
        while time.time() - start_time < VIDEO_DURATION:
            frame = picam2.capture_array()    
            #in_rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)             
            out.write(frame)
            
            cv2.imshow('pose_landmarker_test_recorder', frame)
       
            if cv2.waitKey(1) == 27:
                break
            

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        picam2.stop()
        out.release()
   
    print(f"Video saved as {VIDEO_FILENAME}")

if __name__ == "__main__":
    run()
