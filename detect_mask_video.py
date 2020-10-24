#!/usr/bin/env python3
from loguru import logger

# import the necessary packages
import os, sys, json, pathlib, time, random, traceback, simplejson, cv2, pprint, glob, humanize, datetime, psutil
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
os.environ['VIDEO_PROCESSOR'] = '1'
pp = pprint.PrettyPrinter(indent=2)

DEFAULT_ANALYSIS_FPS = 1
SPINNER_STREAM = sys.stderr
DEFAULT_RTSP = f"rtsp://127.0.0.1:8554/mystream"
DEFAULT_SHOW_OUTPUT_FRAMES = False
SAVE_OUTPUT_FRAMES = False
SAVE_OUTPUT_FRAMES_DIR = './output_frames'
VIDEOS_DIR = './videos'
_SAVE_FRAMES_ANALYSIS_DIR = './analysis_frames'

TOTALS_PARSE_TIME = {'started':time.time(),}
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


def get_pids():
    for pid in psutil.pids():
    
      try:
        p = psutil.Process(pid)
      except:
        continue
      #print(p)
      my_env_val = os.environ.get('VIDEO_PROCESSOR')
      try:
        penv = dict(p.environ())
      except:
        continue
      ENV_MATCH = (  'VIDEO_PROCESSOR' in penv.keys() and penv['VIDEO_PROCESSOR'] == my_env_val )
      MATCHES = []
      if ENV_MATCH:
        MATCHES.append(p.pid)
      msg = f' my env={my_env_val}, pid {p.pid} env={len(penv.keys())} keys match={ENV_MATCH},  name={p.name()}'
      #L('STATS',msg)
    msg = f' matches={len(MATCHES)},  {MATCHES}'
    L('STATS',msg)



def get_video_info_json(VIDEO_ID):
    oVIDEO_ID = VIDEO_ID
    if '/' in VIDEO_ID and '.' in VIDEO_ID:
      VIDEO_ID = os.path.basename(VIDEO_ID).split('.')[0]
    if '.' in VIDEO_ID:
      VIDEO_ID = VIDEO_ID.split('.')[0]
    d = f'./videos/{VIDEO_ID}.info.json'
    if len(VIDEO_ID) < 8:
      m = f'Invalid ID ({VIDEO_ID}). orig VIDEO_ID={oVIDEO_ID}'
      raise Exception(m)
      sys.exit(1)
    if not os.path.exists(d):
      print(f'path {d} does not exist (get_video_info_json for {VIDEO_ID})')
      sys.exit(1)
    with open(d,'r') as f:
      return json.loads(f.read())

def get_videos_total_bytes():
  SUM = 0
  for v in get_videos():
    SUM += os.path.getsize(v)
  return SUM

def get_videos():
  V =  glob.glob(f'./videos/*.mkv')
  return V

def get_jsons():
  JSONS =  glob.glob(f'./videos/*.info.json')
  return JSONS

get_totals_spinner = Halo(text='Getting Totals JSON', text_color= 'cyan', color='green', spinner='dots', stream=SPINNER_STREAM)
def get_totals():
  get_totals_spinner.start()
  TB = get_videos_total_bytes()
  T = {
   'ts': int(time.time()),
   'duration': 0,
   'total_videos_bytes': int(TB),
   'total_videos_bytes_human': str(humanize.naturalsize(TB)),
   'frames': 0,
   'view_count': 0,
   'pixels': 0,
   'detections': 0,
   'analyzed_frames': 0,
   'analyzed_videos': 0,
   'size_bytes': 0,
   'analyzed_seconds': 0,
   'detection_objects': {
      'min_dur': {'10':[],},
   }
  }
  L('STATS',f'get_videos_total_bytes: {get_videos_total_bytes()}')
  videos_qty = 0
  started_ts = time.time()
  JSONS = get_jsons()
  for j in JSONS:
    get_totals_spinner.color = 'magenta'
    get_totals_spinner.text_color = 'green'
    with open(j,'r') as f:
      J = json.loads(f.read())
      videos_qty += 1
      ID = os.path.basename(j).split('.')[0]
      get_totals_spinner.text = f'Processed {videos_qty}/{len(JSONS)} videos [{ID}] :: '
      VIDEO_FILE = './videos/{}.mkv'.format(ID)
      #video_info = get_video_info_json(VIDEO_FILE)
      #(video_meta, height, width) = findVideoMetada(VIDEO_FILE)
      #L('STATS',video_meta)
      #pp.pprint(video_meta)
      #L('STATS',ID+' ' + VIDEO_FILE)
      #L('STATS',video_info.keys())
      FRAME_ANALYSIS_JSON_FILE = './analysis_frames/{}-frame-analysis-x{}.json'.format(
            ID,
            args['resize_width'],
      )
      if not os.path.exists(FRAME_ANALYSIS_JSON_FILE):
        if args["debug"] and False:
          L('STATS',f' skipping {ID}')
        videos_qty -= 1
        continue

      with open(FRAME_ANALYSIS_JSON_FILE,'r') as f:
        try:
          dat = json.loads(f.read())
        except:
          L('STATS',f' failed to parse json for {ID}')
          os.remove(FRAME_ANALYSIS_JSON_FILE)
          L('STATS',f' removed {FRAME_ANALYSIS_JSON_FILE}')
          videos_qty -= 1
          continue

      T['analyzed_frames'] += len(dat.keys())
      analyzed_seconds = int(len(dat.keys()) / J["fps"])
      T['analyzed_seconds'] += analyzed_seconds
      T['analyzed_videos'] += 1
      J['detections']  = 0
      T['size_bytes']  += os.path.getsize(VIDEO_FILE)
      
      if(analyzed_seconds > 3099999999):
        pp.pprint(J)
        pp.pprint(dat)

      for dk in dat.keys():
        K = dat[dk]
        J['detections'] += len(K['LABELS'])

      T['detections'] += J['detections']

      #for min_duration in T['detection_objects']['min_dur'].keys():
      #  for dk in dat.keys():
      #    K = dat[dk]
        
    for tk in T.keys():
     if tk in J.keys() and tk in ['duration','frames','view_count']:
       T[tk] += J[tk]
     if 'fps' in J.keys():
       frames = int(J['duration'] * J['duration'])
       T['frames'] += frames
    T['pixels'] += int(J['width']*J['height'])
    if args["debug"]:
      get_totals_spinner.succeed(f'OK- {ID}')


  ended_ts = time.time()
  T['videos_qty'] = videos_qty
  T['duration_human'] = humanize.naturaldelta(datetime.timedelta(seconds=T['duration']))
  T['analyzed_time_human'] = humanize.naturaldelta(datetime.timedelta(seconds=T['analyzed_seconds']))
  T['pixels'] = T['pixels'] * T['frames']
  T['started_ts'] = int(TOTALS_PARSE_TIME['started'])
  T['ended_ts'] = int(time.time())
  T['dur'] = T['ended_ts'] - T['started_ts']
  T['dur_human'] = humanize.naturaldelta(datetime.timedelta(seconds=T['dur']))
  T['size_human'] = humanize.naturalsize(T['size_bytes'])
  PERCENTAGE = round((T['analyzed_seconds'] / T['duration'])*100, 2)
  ANALYZED_PERCENTAGE = round((int(T['analyzed_videos']) / len(JSONS))*100, 2)
  T['msg'] = f"Detected {T['detections']} Frames with mask/nomask using {T['analyzed_time_human']} of Video from {int(T['analyzed_videos'])} ({T['size_human']}) Videos Files of the {len(JSONS)} ({T['duration_human']}, {T['total_videos_bytes_human']}, and {int(T['frames'])} Frames) Local Videos in {T['dur_human']}"

  if args["debug"]:
    L('STATS', T)
  L('STATS', T['msg'])
  get_totals_spinner.stop()
  sys.stderr.write("\n{}\n".format(T['msg']))
  print(json.dumps(T)+"\n")
  get_pids()
  print('exiting.................')
  sys.exit(0)



@Halo(text='Converting Video to Desired Frame Rate', text_color= 'yellow', color='green', spinner='dots', stream=SPINNER_STREAM)
def convert_video_to_fps(VIDEO_FILE, FPS):
    FPS = int(round(FPS,2))
    OUT_FILE = f'./videos_fps/{FPS}/{os.path.basename(VIDEO_FILE)}'
    print(f'convert_video_to_fps.....................                 ')
    if os.path.exists(OUT_FILE):
        try:
          (video_meta, height, width) = findVideoMetada(OUT_FILE)
          #L('STATS', video_meta)
          #pp.pprint(video_meta)
          avg_frame_rate = video_meta['streams'][0]['avg_frame_rate']
          L('STATS', f'avg_frame_rate={avg_frame_rate}')
          avg_frame_rate_split = avg_frame_rate.split('/')
          fps = int(round(int(avg_frame_rate_split[0]) / int(avg_frame_rate_split[1]), 2))
          L('STATS', f'fps={fps}')
        except:
          fps = 0
        if int(fps) != int(FPS):
            msg = f'        REMOVING invalid source file {OUT_FILE}         ({fps}!={FPS})'
            L('STATS', msg)
            L('STATS', video_meta)
            #sys.exit()
            os.remove(OUT_FILE)
            msg = f'        REMOVED invalid source file {OUT_FILE}'
            L('STATS', msg)

    cmd = ["command","ffmpeg","-v","quiet","-i",VIDEO_FILE,"-an", "-r", str(int(FPS)),"-vf","scale=400:-2", "-y", OUT_FILE]
    L('STATS',f'cmd={" ".join(cmd)}')
    if not os.path.exists(os.path.dirname(OUT_FILE)):
      pathlib.Path(os.path.dirname(OUT_FILE)).mkdir(parents=True)

    if not os.path.exists(OUT_FILE):
      result = subprocess.run(cmd, shell=False)
    L('STATS', f'output file exists = {os.path.exists(OUT_FILE)}')
    if not os.path.exists(OUT_FILE):
        sys.exit(9)
    return OUT_FILE

def findVideoMetada(pathToInputVideo):
    cmd = f"command ffprobe -v quiet -print_format json -show_streams"
    args = shlex.split(cmd)
    args.append(pathToInputVideo)
    ffprobeOutput = subprocess.check_output(args).decode('utf-8')
    ffprobeOutput = json.loads(ffprobeOutput)
    height = ffprobeOutput['streams'][0]['height']
    width = ffprobeOutput['streams'][0]['width']
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
ap.add_argument("-X", "--x-offset", type=int, default=0,
    help="X offset")
ap.add_argument("-Y", "--y-offset", type=int, default=0,
    help="Y offset")
ap.add_argument("-W", "--resize-width", type=int, default=400,
    help="Resize Image Width. Default 400")
ap.add_argument("-P", "--fps", type=int, default=int(DEFAULT_ANALYSIS_FPS),
    help=f"Max analysis fps (1 = check 1 image per 30 if the video is 30fps).  Default={DEFAULT_ANALYSIS_FPS}")
ap.add_argument("-S", "--stats-interval", type=int, default=5,
    help="Stats Interval")
ap.add_argument("-F", "--file", type=str, default=None,
    help="Video File Path")
ap.add_argument("-s","--show", action='store_true', default=DEFAULT_SHOW_OUTPUT_FRAMES,
    help=f"Show Mode (default {DEFAULT_SHOW_OUTPUT_FRAMES}")
ap.add_argument("-H","--hide", action='store_true', default=False,
    help=f"Hide Mode (default {False}")
ap.add_argument("-d","--debug", action='store_true', default=False,
    help="Debug Mode")
ap.add_argument("-t","--get-totals", action='store_true', default=False,
    help="Get Totals")
args = vars(ap.parse_args())
#L("STATS",  args)

if args['get_totals']:
  get_totals()
  sys.exit()

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

FRAME_ANALYSIS_JSON_FILE = '{}-frame-analysis-{}x-@{}fps.json'.format(
    SAVE_FRAMES_ANALYSIS_DIR,
    args['resize_width'],
    args['fps'],
)
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
    return f.write(simplejson.dumps(a))
  
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
    SRC_FILE_BYTES = os.path.getsize(args["file"])
    requested_fps_ok = False
    print(f'      before conversion working with {video_file_fps}')
    if (  int(round(video_file_fps,2)) == int(round(args['fps'],2)) ):
        requested_fps_ok = True
    L('STATS',    f'      {video_file_fps}xfps   requested={args["fps"]}       requested_fps_ok={requested_fps_ok},    src_file_bytes={SRC_FILE_BYTES}  ')
    VIDEO_FILE_PATH = convert_video_to_fps(args["file"], int(round(args['fps'])))
    args['file'] = VIDEO_FILE_PATH
    print(f'kkkkkkk                  VIDEO_FILE_PATH={VIDEO_FILE_PATH}')

    print(f'opening file')
    vs = cv2.VideoCapture(args["file"])
    vs.set(cv2.CAP_PROP_POS_FRAMES, FRAME_RATE_START_OFFSET)
    video_file_fps = vs.get(cv2.CAP_PROP_FPS)
    print(f'       after conversion working with {video_file_fps}')

    #sys.exit()

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
face_since_last_stats = []
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
      STATS_MSG = f' #{FRAME_NUM} ::   Processed:{len(PROCESSED_FRAMES)}/{frames_qty} qty={len(x)},   SKIPPED_FRAMES={len(SKIPPED_FRAMES)}, video framerate={video_info["fps"]},  requested analysis rate={args["fps"]},  previously analyzed frame id={last_stats_frames} ,  file={args["file"]},   face_since_last_stats={face_since_last_stats},     '
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
      frame = imutils.resize(frame, width=args['resize_width'])
      if args["debug"]:
        if False:
          print(f" resized to {len(str(frame))} bytes")
      #L('STATS', f" resized to {len(str(frame))} bytes")
    except Exception as e:
      print(f'. {e}                    file={args["file"]},           ')
      if " object has no attribute 'shape'" in str(e):
        break
      L('STATS', f'failed to resize frame: {str(e)}')
      break

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
        BOXES.append([startX, startY, endX, endY])
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
    FA = {'FRAME_NUM':FRAME_NUM,'LABELS':LABELS,}           #'BOXES':BOXES,}
    #print(FA)
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


    if args['show']:
        random.seed(STREAM_NAME)
        RAND_X_COL = args['x_offset']
        RAND_Y_COL = args['y_offset']
        #RAND_X_COL = random.randint(10,100)
        #RAND_Y_COL = random.randint(10,100)
        MSG = f'=RAND_X_COL={RAND_X_COL}, RAND_Y_COL={RAND_Y_COL},'
        #print(MSG)
        WINDOW_X = 100 + (RAND_X_COL * 10)
        WINDOW_Y = 100 + (RAND_Y_COL * 10)

        BG_RECTANGLE = {'start_x':20,'start_y':7,'end_x':190,'end_y':40}
        cv2.rectangle(frame, (BG_RECTANGLE['start_x'], BG_RECTANGLE['start_x']), (BG_RECTANGLE['end_x'], BG_RECTANGLE['end_y']),  (255, 255, 255), -1)
        #cv2.rectangle(frame, (20, 30), (150, 80),  (255, 255, 0), 2)
        TEXT_TITLE = "{ID}".format(
            ID=os.path.basename(args['file']).split('.')[0],
        )
        cv2.putText(frame, TEXT_TITLE, (BG_RECTANGLE['start_x'], 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)
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
