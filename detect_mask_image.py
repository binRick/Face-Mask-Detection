#!/usr/bin/env python3
# USAGE
# python detect_mask_image.py --image images/pic1.jpeg

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imgcat import imgcat
import numpy as np
import argparse
import cv2
import os, sys, json, pathlib, time


NO_MASK_DIR = 'results/no_mask'
MASK_DIR = 'results/mask'
OUTPUT_MODE = 'simple'
MANGLE_SOURCE_IMAGE = False

for d in [NO_MASK_DIR, MASK_DIR]:
  if not os.path.exists(d):
    pathlib.Path(d).mkdir(parents=True)


def record_face(startX, startY, endX, endY, confidence, mask_detected, src_image, dst_image, dur_ms):
    if OUTPUT_MODE == 'simple':
     print(f'Mask: {bool(mask_detected)} ({str(confidence)}%)')
    else:
     print(json.dumps({
      'startX': int(startX),
      'startY': int(startY),
      'endX': int(endX),
      'endY': int(endY),
      'src_image': str(src_image),
      'dst_image': str(dst_image),
      'confidence_percent': str(confidence),
      'mask_detected': bool(mask_detected),
      'dur_ms': int(dur_ms),
     }))

def mask_image():
        # construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--image", required=False,
                help="path to input image")
        ap.add_argument("-I", "--images", required=False,
                help="input images seperated by commas")
        ap.add_argument("-f", "--face", type=str,
                default="face_detector",
                help="path to face detector model directory")
        ap.add_argument("-m", "--model", type=str,
                default="mask_detector.model",
                help="path to trained face mask detector model")
        ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
        args = vars(ap.parse_args())

        if args["images"]:
          IMAGES = args["images"].split(',')
        else:
          IMAGES = [args["image"]]

        # load our serialized face detector model from disk
        print("[INFO] loading face detector model...")
        prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
        weightsPath = os.path.sep.join([args["face"],
                "res10_300x300_ssd_iter_140000.caffemodel"])
        net = cv2.dnn.readNet(prototxtPath, weightsPath)

        # load the face mask detector model from disk
        print("[INFO] loading face mask detector model...")
        model = load_model(args["model"])


        ##  for each image:
        #print(f'IMAGES={IMAGES}')
        #args["confidence"] = args["confidence"] / 100
        for img in IMAGES:
            if not os.path.exists(img):
              continue
            start_ms = int(time.time())
            args["image"] = img
            #print(f'img={img}')
            image = cv2.imread(img)
            # load the input image from disk, clone it, and grab the image spatial
            # dimensions
            image = cv2.imread(img)
            orig = image.copy()
            (h, w) = image.shape[:2]

            # construct a blob from the image
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                    (104.0, 177.0, 123.0))

            # pass the blob through the network and obtain the face detections
            #print(f"[INFO] computing face detections :: {img}...")
            net.setInput(blob)
            detections = net.forward()

            # loop over the detections
            for i in range(0, detections.shape[2]):
                    # extract the confidence (i.e., probability) associated with
                    # the detection
                    confidence = detections[0, 0, i, 2]

                    # filter out weak detections by ensuring the confidence is
                    # greater than the minimum confidence
                    if confidence > args["confidence"]:
                            # compute the (x, y)-coordinates of the bounding box for
                            # the object
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")

                            # ensure the bounding boxes fall within the dimensions of
                            # the frame
                            (startX, startY) = (max(0, startX), max(0, startY))
                            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                            # extract the face ROI, convert it from BGR to RGB channel
                            # ordering, resize it to 224x224, and preprocess it
                            face = image[startY:endY, startX:endX]
                            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                            face = cv2.resize(face, (224, 224))
                            face = img_to_array(face)
                            face = preprocess_input(face)
                            face = np.expand_dims(face, axis=0)

                            # pass the face through the model to determine if the face
                            # has a mask or not
                            (mask, withoutMask) = model.predict(face)[0]

                            # determine the class label and color we'll use to draw
                            # the bounding box and text
                            label = "Mask" if mask > withoutMask else "No Mask"
                            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                            # include the probability in the label
                            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                            confidence = "{:.2f}".format(max(mask, withoutMask) * 100)
                            mask_detected = (True if mask > withoutMask else False)
                            dur_ms = int(time.time()) - start_ms
                            img_extension = img.split('.')[-1]
                            RESULTS_DIR = MASK_DIR if mask > withoutMask else NO_MASK_DIR
                            dst_image = '{}/{}_{}_{}.{}'.format(
                                RESULTS_DIR,
                                os.path.basename(img).replace('.{}'.format(img_extension),''),
                                'face',
                                i+1,
                                img_extension,
                            )
                            clone = image.copy()
                            cropped_img = clone[startY:endY, startX:endX]
    #                        cv2.imwrite(dst_image, clone)
                            cv2.imwrite(dst_image, cropped_img)
                            imgcat(open(dst_image))
                            record_face(startX, startY, endX, endY, confidence, mask_detected, img, dst_image, dur_ms)
                            if MANGLE_SOURCE_IMAGE:
                                cv2.putText(clone, label, (startX, startY - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                                cv2.rectangle(clone, (startX, startY), (endX, endY), color, 2)
                                # display the label and bounding box rectangle on the output
                                # frame
                                cv2.putText(image, label, (startX, startY - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                                cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

        # show the output image
        #print(image)
        #filename = 'processed_image.jpg'
        #cv2.imwrite(filename, image)
        sys.exit(0)
        #cv2.imshow("Output", image)
        #cv2.waitKey(0)
        
if __name__ == "__main__":
        mask_image()
