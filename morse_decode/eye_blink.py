from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

def convertMorseToText(s):

    morse_code={'' : 'Check',
    		'.-': 'a',
    		'-...': 'b',
    		'-.-.': 'c',
    		'-..': 'd',
    		'.': 'e',
    		'..-.': 'f',
    		'--.': 'g',
    		'....': 'h',
    		'..': 'i',
    		'.---': 'j',
    		'-.-': 'k',
    		'.-..': 'l',
    		'--': 'm',
    		'-.': 'n',
    		'---': 'o',
    		'.--.': 'p',
    		'--.-': 'q',
    		'.-.': 'r',
    		'...': 's',
    		'-': 't',
    		'..-': 'u',
    		'...-': 'v',
    		'.--': 'w',
    		'-..-': 'x',
    		'-.--': 'y',
    		'--..': 'z',
    		'.-.-': ' '
    		}

    if morse_code.get(s)!='Check':
    	return str(morse_code.get(s))
		


def eyeAspectRatio(eye):

	x = dist.euclidean(eye[1], eye[5])
	y = dist.euclidean(eye[2], eye[4])
	z = dist.euclidean(eye[0], eye[3])

	ear = (x + y) / (2.0 * z)
	return ear
 

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="", help="path to input video file")
args = vars(ap.parse_args())
 

EYE_AR_THRESH = 0.27
EYE_AR_CONSEC_FRAMES =2
EYE_AR_CONSEC_FRAMES2=6
EYE_AR_CONSEC_FRAMES3=11


COUNTER = 0
TOTAL=[]

decodedM=""

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]



fileStream = True
vid = VideoStream(src=0).start()
time.sleep(1.0)

while True:

	stream = vid.read()
	stream = imutils.resize(stream, width=550)
	gray = cv2.cvtColor(stream, cv2.COLOR_BGR2GRAY)
	recta = detector(gray, 0)

	for rect in recta:

		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		rightEye = shape[rStart:rEnd]
		leftEye = shape[lStart:lEnd]

		rightEar = eyeAspectRatio(rightEye)
		leftEar = eyeAspectRatio(leftEye)

		ear = (rightEar+leftEar) / 2.0


		rightHull = cv2.convexHull(rightEye)
		leftHull = cv2.convexHull(leftEye)
		
		cv2.drawContours(stream, [leftHull], -1, (0, 255, 0), 1)
		cv2.drawContours(stream, [rightHull], -1, (0, 255, 0), 1)


		if ear < EYE_AR_THRESH:
			COUNTER += 1

		else:

			if COUNTER >= EYE_AR_CONSEC_FRAMES and COUNTER<=EYE_AR_CONSEC_FRAMES2:
				TOTAL.append(".")
			elif COUNTER >=EYE_AR_CONSEC_FRAMES2 and COUNTER<=EYE_AR_CONSEC_FRAMES3:
				TOTAL.append("-")
			elif COUNTER>=EYE_AR_CONSEC_FRAMES3:
				s=str(convertMorseToText(''.join(TOTAL)))
				if s=="None":
					TOTAL=[]
				else:
					decodedM +=s
					TOTAL=[]

			COUNTER = 0


		cv2.putText(stream,"CODE: {}".format(TOTAL),(20,350),cv2.FONT_HERSHEY_DUPLEX,0.8,(0,0,255),2)
		cv2.putText(stream,"Decoded Msg: {}".format(decodedM),(20,400),cv2.FONT_HERSHEY_TRIPLEX,0.7,(0,255,0),2)
 

	cv2.imshow("Streaming...", stream)
	key = cv2.waitKey(1) & 0xFF
	#print(string)
	file = open("decoded.txt","w")
	file.write(decodedM)
	file.close

	if key == ord("q"):
		break


cv2.destroyAllWindows()
vid.stop()

