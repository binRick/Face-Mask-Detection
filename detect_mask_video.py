#!/usr/bin/env python3
from loguru import logger

# import the necessary packages
import os, sys, json, pathlib, time, random, traceback, simplejson, cv2, pprint
from threading import Thread
from queue import Queue
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.video.count_frames import count_frames
from lockfile import LockFile
import numpy as np
import argparse, imutils, time, cv2, os, sys, json, p_tqdm, shlex, subprocess
from halo import Halo


DEFAULT_RTSP = f"rtsp://127.0.0.1:8554/mystream"
SHOW_OUTPUT_FRAMES = True
SAVE_OUTPUT_FRAMES = False
SAVE_OUTPUT_FRAMES_DIR = './output_frames'
_SAVE_FRAMES_ANALYSIS_DIR = './analysis_frames'
RESIZE_IMAGE_WIDTH = 400

dat_file = '/tmp/.detect_video.json'
lock = LockFile(f'{dat_file}.lock')

last_stats_interval = 0
now_ts = int(time.time())

new_level = logger.level("STATS", no=38, color="<yellow>", icon="🐍")

L = logger.log

class FileVideoStream:
	def __init__(self, path, queueSize=128):
		self.stream = cv2.VideoCapture(path)
		self.stopped = False
		self.Q = Queue(maxsize=queueSize)
	def update(self):
		# keep looping infinitely
		while True:
			# if the thread indicator variable is set, stop the
			# thread
			if self.stopped:
				return
			# otherwise, ensure the queue has room in it
			if not self.Q.full():
				# read the next frame from the file
				(grabbed, frame) = self.stream.read()
				# if the `grabbed` boolean is `False`, then we have
				# reached the end of the video file
				if not grabbed:
					self.stop()
					return
				# add the frame to the queue
				self.Q.put(frame)
	def read(self):
		# return next frame in the queue
		return self.Q.get()
	def more(self):
		# return True if there are still frames in the queue
		return self.Q.qsize() > 0
	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True

def get_video_info_json(VIDEO_ID):
    if '.' in VIDEO_ID:
      VIDEO_ID = VIDEO_ID.split('.')[0]
    d = f'{VIDEO_ID}.info.json'
    with open(d,'r') as f:
      return json.loads(f.read())


def findVideoMetada(pathToInputVideo):
    cmd = "ffprobe -v quiet -print_format json -show_streams"
    args = shlex.split(cmd)
    args.append(pathToInputVideo)
    # run the ffprobe process, decode stdout into utf-8 & convert to JSON
    ffprobeOutput = subprocess.check_output(args).decode('utf-8')
    ffprobeOutput = json.loads(ffprobeOutput)

    # prints all the metadata available:
    #import pprint
    #pp = pprint.PrettyPrinter(indent=2)
    #pp.pprint(ffprobeOutput)

    # for example, find height and width
    height = ffprobeOutput['streams'][0]['height']
    width = ffprobeOutput['streams'][0]['width']
    
    #print(height, width)
    return ffprobeOutput, height, width

def count_frames(path, override=False):
	# grab a pointer to the video file and initialize the total
	# number of frames read
	video = cv2.VideoCapture(path)
	total = 0
	# if the override flag is passed in, revert to the manual
	# method of counting frames
	if override:
		total = count_frames_manual(video)
	# otherwise, let's try the fast way first
	else:
		# lets try to determine the number of frames in a video
		# via video properties; this method can be very buggy
		# and might throw an error based on your OpenCV version
		# or may fail entirely based on your which video codecs
		# you have installed
		try:
			# check if we are using OpenCV 3
			if True: #is_cv3():
				total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
			# otherwise, we are using OpenCV 2.4
			else:
				total = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
		# uh-oh, we got an error -- revert to counting manually
		except:
			total = count_frames_manual(video)
	# release the video file pointer
	video.release()
	# return the total number of frames in the video
	return total


def trim_dat_file():
  keep_lines = []
  with open(dat_file,'r') as f:
    dat = f.read().splitlines()
  for d in dat:
    d = str(d)
    print(d)
    continue
    if not d or d == '':
      continue
    try:
      j = json.loads(d)
    except Exception as e:
      j = {}
    print(j)
    if (not 'pid' in j.keys()) or (not os.path.exists(f'/proc/{j["pid"]}')):
      if int(j['pid']) == os.getpid():
        keep_lines.append(d)
    else:
      keep_lines.append(d)
  print(f'dat={dat}, {keep_lines}, pid={os.getpid()}, ')
  return
  s = "\n".join(keep_lines)+"\n"
  with open(dat_file,'w') as f:
    f.write('')
  for l in keep_lines:
   with open(dat_file,'a') as f:
    f.write(l+"\n")
  print(f'wrote {len(keep_lines)} lines to {dat_file}')


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

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
            face = frame[startY:endY, startX:endX]
            try:
              face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            except:
              continue
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
    default="face_detector",
    help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
    default="mask_detector.model",
    help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-r", "--rtsp", type=str, default=DEFAULT_RTSP,
    help="RTSP URI")
ap.add_argument("-P", "--fps", type=int, default=1,
    help="Max analysis fps (1 = check 1 image per 30 if the video is 30fps)")
ap.add_argument("-S", "--stats-interval", type=int, default=5,
    help="Stats Interval")
ap.add_argument("-F", "--file", type=str, default=None,
    help="Video File Path")
ap.add_argument("-d","--debug", action='store_true', default=False,
    help="Debug Mode")
args = vars(ap.parse_args())
L("STATS",  args)
# load our serialized face detector model from disk
L("STATS","[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
    "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")

STREAM_NAME = os.path.basename(args['file']).split('.')[0]
SAVE_FRAMES_ANALYSIS_DIR = '{}/{}'.format(_SAVE_FRAMES_ANALYSIS_DIR, STREAM_NAME)
TF = '{}-frames.json'.format(SAVE_FRAMES_ANALYSIS_DIR)
FRAME_ANALYSIS_JSON_FILE = '{}-frame-analysis.json'.format(SAVE_FRAMES_ANALYSIS_DIR)
if os.path.exists(TF):
  os.remove(TF)

frames_qty = None
FILE_MODE = False
fps = None
def get_frame_analysis():
  if not os.path.exists(FRAME_ANALYSIS_JSON_FILE):
    with open(FRAME_ANALYSIS_JSON_FILE,'w') as f:
      f.write('{}')
      return get_frame_analysis()
  else:
    with open(FRAME_ANALYSIS_JSON_FILE,'r') as f:
      try:
        d = json.loads(f.read())
      except:
        return {}
      if type(d) != dict:
        d = {}
      return d

def add_frame_analysis(FRAME_NUMBER, ANALYSIS):
  a = get_frame_analysis()
  a[FRAME_NUMBER] = ANALYSIS
  with open(FRAME_ANALYSIS_JSON_FILE,'w') as f:
    return f.write(json.dumps(a))
  
def get_processed_frames():
 return [int(k) for k in get_frame_analysis().keys() if k]

pf = get_processed_frames()
MAX_PP_FRAME_NUM = None
MIN_PP_FRAME_NUM = None
if len(pf) > 0:
  MAX_PP_FRAME_NUM = max(pf)
  MIN_PP_FRAME_NUM = min(pf)

FRAME_RATE_START_OFFSET = 0
if MAX_PP_FRAME_NUM:
  FRAME_RATE_START_OFFSET = MAX_PP_FRAME_NUM


L('STATS', f'     Starting from frame offset {FRAME_RATE_START_OFFSET}   ')


if args["rtsp"] == "webcam":
    print(f"Binding to webcam")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    STREAM_NAME = 'none'

elif args['file'] != None:
    print(f'opening file')
    vs = cv2.VideoCapture(args["file"])
    vs.set(cv2.CAP_PROP_POS_FRAMES, FRAME_RATE_START_OFFSET)
    video_file_fps = vs.get(cv2.CAP_PROP_FPS)
    L('STATS',    f'      {video_file_fps}xfps      ')

    #vs.set(cv2.CAP_PROP_FPS, 1)
    FH = vs.get(cv2.CAP_PROP_FRAME_HEIGHT)
    FW = vs.get(cv2.CAP_PROP_FRAME_WIDTH)
    FO = vs.get(cv2.CAP_PROP_POS_FRAMES)
    FP = vs.get(cv2.CAP_PROP_POS_MSEC)
    FC = vs.get(cv2.CAP_PROP_FRAME_COUNT)


    video_file_fps = vs.get(cv2.CAP_PROP_FPS)
    L('STATS',    f'      {video_file_fps}xfps     VH={type(FH)}={FH}/{FW} :: FO={FO},   FP={type(FP)}={FP},      FC={FC},         ')


    #(video_meta, height, width) = findVideoMetada(args['file'])
    video_info = get_video_info_json(args['file'])
    #print(f'opened {type(video_meta)},      {height}x{width}   ,       {dict(video_meta).keys()}')
    pp = pprint.PrettyPrinter(indent=2)
    #pp.pprint(video_meta)
    #pp.pprint(video_info)

    VIDEO_OBJECT = {
      'fps': int(video_info['fps']),
      'duration': int(video_info['duration']),
      'format': str(video_info['format']),
      'height': int(video_info['height']),
      'width': int(video_info['width']),
      'upload_date': str(video_info['upload_date']),
      'view_count': int(video_info['view_count']),
      'title': str(video_info['title']),
      'uploader': str(video_info['uploader']),
    }
    VIDEO_OBJECT['frames'] = VIDEO_OBJECT['fps'] * VIDEO_OBJECT['duration']
    L("STATS",VIDEO_OBJECT)
    #sys.exit()
    #fps = FPS().start()
    print(f'started')
    #frames_qty = count_frames(args['rtsp'])
    #print(f'       #{frames_qty}')
    #sys.exit()
    STREAM_NAME = os.path.basename(args['file']).split('.')[0]
    FILE_MODE = True
    frames_qty = count_frames(args["file"])
    msg = f'        frames_qty={frames_qty}'
    L("STATS",msg)
    #sys.exit()
else:
    STREAM_NAME = args['rtsp'].split('/')
    STREAM_NAME = STREAM_NAME[len(STREAM_NAME)-1]
    #if not os.path.exists(SAVE_FRAMES_ANALYSIS_DIR):
    #  pathlib.Path(SAVE_FRAMES_ANALYSIS_DIR).mkdir(parents=True)
    print(f"Binding to {args['rtsp']} => {STREAM_NAME}")
    vs = VideoStream(src=args['rtsp']).start()



msg = f'#{len(pf)} pre-processed frames (max {MAX_PP_FRAME_NUM}, min {MIN_PP_FRAME_NUM},   '
L('STATS', msg)

started = time.time()
FRAME_NUM = 0 + FRAME_RATE_START_OFFSET
SKIPPED_FRAMES = []
fps_since_stats = 0
last_stats_frames = None
last_stats_ts = None
last_stats_frames_delta = None
last_stats_fps = None
# loop over the frames from the video stream
while True:
    if frames_qty and FRAME_NUM > frames_qty:
      break
    #print("Waiting for frame..")
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    if FILE_MODE:
       	(grabbed, frame) = vs.read()
        #if (not grabbed) or (not frame):
        #  L('STATS', 'FRAME NOT GRABBED')
        #  sys.exit(1)
    else:
        frame = vs.read()


    frame_read = time.time()
    frame_delay = frame_read - started
    FRAME_NUM += 1
    fps_since_start = round(FRAME_NUM / frame_delay, 2)
    now_ts = int(time.time())
    dxx = (now_ts - last_stats_interval) 
    too_old = (dxx > args['stats_interval'])
    msg = f'last_stats_interval+args["stats_interval"]={last_stats_interval+args["stats_interval"]}, now_ts={now_ts}, dxx={dxx}, too_old={too_old},  '
    #print(msg)
    PROCESSED_FRAMES = get_processed_frames()
    if (last_stats_interval+args["stats_interval"]) < now_ts:
      if last_stats_frames:
        last_stats_frames_delta = round(FRAME_NUM - last_stats_frames, 2)
        last_stats_fps = round(last_stats_frames_delta / (now_ts - last_stats_interval), 2)
      msg = f'stats..... fps_since_start={fps_since_start}, frame #:{FRAME_NUM},  last_stats_frames_delta={last_stats_frames_delta}@{last_stats_interval} => {last_stats_fps} '
      x = get_frame_analysis()
      STATS_MSG = f' #{FRAME_NUM} ::   Processed:{len(PROCESSED_FRAMES)}/{frames_qty} qty={len(x)},   SKIPPED_FRAMES={len(SKIPPED_FRAMES)}, video framerate={video_info["fps"]},  requested analysis rate={args["fps"]},  previously analyzed frame id={last_stats_frames} '
      L("STATS",  msg)
      L("STATS",  STATS_MSG)
      last_stats_interval = now_ts
      last_stats_frames = FRAME_NUM
      #print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))

    if FRAME_NUM in PROCESSED_FRAMES:
      SKIPPED_FRAMES.append(FRAME_NUM)
      continue

    #if args["debug"]:
    #  print("[INFO] processing Frame # {}".format(FRAME_NUM))
    #print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    #print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    #frame_count = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    #print(f'FRAME_NUM={FRAME_NUM}, fps_since_start={fps_since_start}, ')

    #print(f" read {len(str(frame))} bytes")
    try:
      frame = imutils.resize(frame, width=RESIZE_IMAGE_WIDTH)
      if args["debug"]:
        if False:
          print(f" resized to {len(str(frame))} bytes")
      #L('STATS', f" resized to {len(str(frame))} bytes")
    except Exception as e:
      L('STATS', f'failed to resize frame: {str(e)}')
      continue

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)



    # loop over the detected face locations and their corresponding
    # locations
    BOXES = []
    PREDICTIONS = []
    LABELS = []
    COLORS = []
    msg = f'  predictions qty={len(preds)},    '
    #L('STATS', msg)
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        BOXES.append(box)
        PREDICTIONS.append(pred)
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        COLORS.append(color)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        LABELS.append(label)
        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    df = '{}/{}.json'.format(SAVE_FRAMES_ANALYSIS_DIR, str(FRAME_NUM))
    FA = {'FRAME_NUM':FRAME_NUM,'LABELS':LABELS,'frame_delay':frame_delay,}

    add_frame_analysis(FRAME_NUM, FA, )

    if SAVE_OUTPUT_FRAMES:
        save_path = '{}/{}.png'.format(
          SAVE_OUTPUT_FRAMES_DIR,
          str(int(time.time()*1000)),
        )
        if not os.path.exists(os.path.dirname(save_path)):
          pathlib.Path(os.path.dirname(save_path)).mkdir(parents=True)

        print(f'saving to {save_path}')
        cv2.imwrite(save_path, frame)


    if SHOW_OUTPUT_FRAMES:
        random.seed(STREAM_NAME)
        RAND_X_COL = random.randint(10,100)
        RAND_Y_COL = random.randint(10,100)
        MSG = f'=RAND_X_COL={RAND_X_COL}, RAND_Y_COL={RAND_Y_COL},'
        #print(MSG)
        WINDOW_X = 100 + (RAND_X_COL * 10)
        WINDOW_Y = 100 + (RAND_Y_COL * 10)
        cv2.imshow(STREAM_NAME, frame)
        cv2.moveWindow(STREAM_NAME, WINDOW_X, WINDOW_Y);

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
if not FILE_MODE:
  vs.stop()
