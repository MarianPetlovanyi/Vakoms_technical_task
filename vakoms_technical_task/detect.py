from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import yaml


def compute_iou(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	iou = interArea / float(boxAArea + boxBArea - interArea)

	return iou

def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
	if len(boxes) == 0:
		return []

	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	pick = []

	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = y2

	if probs is not None:
		idxs = probs

	idxs = np.argsort(idxs)

	while len(idxs) > 0:
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		overlap = (w * h) / area[idxs[:last]]

		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	return pick
def main(image_path = "demo.jpg"):
   
    INPUT_SIZE = (224,224)
        

    MODEL_PATH = "models/aircraft_detector.h5"
    ENCODER_PATH = "models/label_encoder.pickle"

    MAX_PROPOSALS_INFER = 200

    MIN_PROBA = 0.99

    print("[INFO] loading model and label binarizer...")
    model = load_model(MODEL_PATH)
    lb = pickle.loads(open(ENCODER_PATH, "rb").read())

    MIN_PROBA = 0.0005

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = imutils.resize(image, width=500)

    print("[INFO] running selective search...")
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()


    proposals = []
    boxes = []


    for (x, y, w, h) in rects[:MAX_PROPOSALS_INFER]:
        roi = image[y:y + h, x:x + w]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi, INPUT_SIZE,
            interpolation=cv2.INTER_CUBIC)

        roi = img_to_array(roi)
        roi = preprocess_input(roi)

        proposals.append(roi)
        boxes.append((x, y, x + w, y + h))

    proposals = np.array(proposals, dtype="float32")
    boxes = np.array(boxes, dtype="int32")
    print("[INFO] proposal shape: {}".format(proposals.shape))

    print("[INFO] classifying proposals...")
    proba = model.predict(proposals)

    print("[INFO] applying NMS...")
    labels = lb.classes_[np.argmax(proba, axis=1)]

    idxs = np.where(labels != "no_aircraft")
    labels = labels[np.where(labels != "no_aircraft")]
    boxes = boxes[idxs]
    proba = proba[idxs][:, 1]


    idxs = np.where(proba <= MIN_PROBA)
    boxes = boxes[idxs]
    proba = proba[idxs]

    clone = image.copy()


    for (box, prob, label) in zip(boxes, proba, labels):
        (startX, startY, endX, endY) = box
        cv2.rectangle(clone, (startX, startY), (endX, endY),
            (0, 255, 0), 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        text= "{}: {:.2f}%".format(label,(1-prob) * 100)
        cv2.putText(clone, text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    print("show the output after *before* running NMS")
    cv2.imshow("Detection results", clone)


    boxIdxs = non_max_suppression(boxes, proba)

    for i in boxIdxs:
        (startX, startY, endX, endY) = boxes[i]
        cv2.rectangle(image, (startX, startY), (endX, endY),
            (0, 255, 0), 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        text= "{}: {:.2f}%".format(labels[i],(1- proba[i]) * 100)
        cv2.putText(image, text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    print("show the output image *after* running NMS")
    cv2.imshow("Results after NMS", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description here")
    parser.add_argument("image_path", type=str, help="Path to the image file")

    args = parser.parse_args()
    main(args.image_path)  