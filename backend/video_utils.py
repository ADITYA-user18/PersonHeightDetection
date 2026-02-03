from threading import Thread
import cv2
import time

class VideoStream:
    """
    Class to continuously read frames from a file or camera in a separate thread.
    This prevents the main loop from being blocked by I/O operations.
    """
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.total_frames = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            
            # Read the next frame
            grabbed, frame = self.stream.read()
            
            if not grabbed:
                # Video ended, loop it
                self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            self.grabbed = grabbed
            self.frame = frame
            
            # Removed sleep to allow maximum read speed. 
            # Main loop controls playback speed via processing time.
            time.sleep(0.001) # Tiny sleep to yield CPU, but don't force FPS sync here

    def read(self):
        return self.frame

    def running(self):
        return not self.stopped

    def stop(self):
        self.stopped = True
        self.stream.release()
