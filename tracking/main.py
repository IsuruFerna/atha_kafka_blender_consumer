from arm_tracking import armDetectFunction
from hand_tracking import handDetectFunction
import cv2 as cv

video = cv.VideoCapture(0)
quit_key = 'k'

armDetectFunction(video, quit_key)
handDetectFunction(video, quit_key)

