import numpy as np
import time
import cv2

names = ['with_render.mp4', 'without_render.mp4'];
window_titles = ['with render', 'without render']


cap = [cv2.VideoCapture(i) for i in names]

frames = [None] * len(names);
gray = [None] * len(names);
ret = [None] * len(names);

while True:

    for i,c in enumerate(cap):
        if c is not None:
            ret[i], frames[i] = c.read();


    for i,f in enumerate(frames):
        if ret[i] is True:
            gray[i] = f
            cv2.imshow(window_titles[i], gray[i]);
        time.sleep(0.3)

    if cv2.waitKey(1) & 0xFF == ord('q'):
       break


for c in cap:
    if c is not None:
        c.release();

cv2.destroyAllWindows()
