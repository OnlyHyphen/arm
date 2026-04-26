import cv2
import time
import board
import busio
import threading
import numpy as np
from flask import Flask, Response, render_template_string, request, jsonify
from adafruit_pca9685 import PCA9685
from rknnlite.api import RKNNLite

# --- CONFIGURATION ---
MODEL_PATH = '/home/radxa/detect/best_rknn_model_v2/best-rk3588.rknn'
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CENTER_X = CAMERA_WIDTH / 2

HOST_IP = '0.0.0.0'
HOST_PORT = 5000

# --- TUNING (live-adjustable) ---
params = {
    'Kp':                  0.0006,
    'Kd':                  0.0008,
    'BASE_SPEED':          0.08,
    'MAX_STEER':           0.5,
    'ROI_VERTICAL_CUTOFF': 0.65,
    'LANE_WIDTH_PIXELS':   450,
    'STOP_DURATION':       2.0,
    'STOP_COOLDOWN':       5.0,
    'CONF_THRESH':         0.35,
}
params_lock = threading.Lock()

# --- FIXED CONFIG ---
IMG_SIZE       = 640
NMS_THRESH     = 0.45
MIN_MOTOR_POWER = 0.07
STOP_THRESHOLD_Y = CAMERA_HEIGHT * 0.5

CLASS_NAMES = {
    0: 'duck',
    1: 'stopsign',
    2: 'rat',
    3: 'redline',
    4: 'l_intersection',
    5: 't_intersection',
    6: 'whiteline',
    7: 'yellowline'
}

CLASS_COLORS = {
    0: (0,   255, 0  ),
    1: (0,   0,   255),
    2: (128, 0,   255),
    3: (0,   0,   200),
    4: (255, 128, 0  ),
    5: (255, 0,   255),
    6: (255, 255, 255),
    7: (0,   255, 255),
}

# --- GLOBALS ---
output_frame = None
lock         = threading.Lock()
app          = Flask(__name__)


# --- RKNN Helpers ---
def preprocess(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.expand_dims(img, axis=0)

def nms(boxes, scores, iou_thresh):
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2-xx1); h = np.maximum(0, yy2-yy1)
        iou = (w*h) / (areas[i] + areas[order[1:]] - w*h)
        order = order[np.where(iou <= iou_thresh)[0]+1]
    return keep

def postprocess(outputs, orig_h, orig_w, conf_thresh):
    pred        = outputs[0][0]
    boxes_raw   = pred[:4, :].T
    class_scores= pred[4:, :].T
    class_ids   = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)
    mask        = confidences > conf_thresh
    boxes_raw, confidences, class_ids = boxes_raw[mask], confidences[mask], class_ids[mask]
    if len(boxes_raw) == 0:
        return []
    x1 = np.clip((boxes_raw[:,0]-boxes_raw[:,2]/2)*orig_w/IMG_SIZE, 0, orig_w)
    y1 = np.clip((boxes_raw[:,1]-boxes_raw[:,3]/2)*orig_h/IMG_SIZE, 0, orig_h)
    x2 = np.clip((boxes_raw[:,0]+boxes_raw[:,2]/2)*orig_w/IMG_SIZE, 0, orig_w)
    y2 = np.clip((boxes_raw[:,1]+boxes_raw[:,3]/2)*orig_h/IMG_SIZE, 0, orig_h)
    boxes_xyxy = np.stack([x1,y1,x2,y2], axis=1)
    results = []
    for cls in np.unique(class_ids):
        m = class_ids==cls
        keep = nms(boxes_xyxy[m], confidences[m], NMS_THRESH)
        for k in keep:
            b = boxes_xyxy[m][k]
            results.append((int(b[0]),int(b[1]),int(b[2]),int(b[3]),float(confidences[m][k]),int(cls)))
    return results

def draw_detections(frame, detections):
    for (x1,y1,x2,y2,conf,cls) in detections:
        color = CLASS_COLORS.get(cls, (0,255,0))
        name  = CLASS_NAMES.get(cls, str(cls))
        label = f"{name}: {conf:.2f}"
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        (lw,lh),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, max(y1-lh-10,0)), (x1+lw, y1), color, -1)
        cv2.putText(frame, label, (x1, max(y1-4,10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
    return frame


# --- Motor Class ---
class Motor:
    def __init__(self, pca, in1, in2):
        self.in1 = pca.channels[in1]
        self.in2 = pca.channels[in2]

    def set_speed(self, speed):
        if abs(speed) < 0.01:
            self.stop(); return
        mapped = MIN_MOTOR_POWER + (abs(speed)*(1.0-MIN_MOTOR_POWER))
        pwm    = int(min(mapped,1.0)*65535)
        if speed > 0: self.in1.duty_cycle=pwm;  self.in2.duty_cycle=0
        else:         self.in1.duty_cycle=0;    self.in2.duty_cycle=pwm

    def stop(self):
        self.in1.duty_cycle = 0
        self.in2.duty_cycle = 0


# --- Robot Thread ---
def robot_control_loop():
    global output_frame, lock

    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c)
        pca.frequency = 100
        left_motors  = [Motor(pca, 0, 1), Motor(pca, 2, 3)]
        right_motors = [Motor(pca, 6, 7), Motor(pca, 4, 5)]
    except Exception as e:
        print(f"Hardware Init Error: {e}"); return

    def set_drive(fwd, steer):
        with params_lock:
            max_steer = params['MAX_STEER']
        steer = max(min(steer, max_steer), -max_steer)
        left  = fwd + steer
        right = fwd - steer
        mv = max(abs(left), abs(right))
        if mv > 1.0: left /= mv; right /= mv
        for m in left_motors:  m.set_speed(left)
        for m in right_motors: m.set_speed(right)

    def stop_all():
        for m in left_motors + right_motors: m.stop()

    print("Loading RKNN Model...")
    rknn = RKNNLite()
    if rknn.load_rknn(MODEL_PATH) != 0:
        print("Failed to load RKNN model"); return
    if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
        print("Failed to init runtime"); return
    print("RKNN Model loaded.")

    prev_error     = 0
    last_stop_time = 0

    print("--- ROBOT STARTED ---")

    try:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera")

        while True:
            ok, frame = cap.read()
            if not ok or frame is None: continue

            frame = cv2.flip(frame, 1)
            orig_h, orig_w = frame.shape[:2]

            # Read current params
            with params_lock:
                Kp          = params['Kp']
                Kd          = params['Kd']
                BASE_SPEED  = params['BASE_SPEED']
                ROI_CUTOFF  = params['ROI_VERTICAL_CUTOFF']
                LANE_WIDTH  = params['LANE_WIDTH_PIXELS']
                STOP_DUR    = params['STOP_DURATION']
                STOP_COOL   = params['STOP_COOLDOWN']
                CONF_THRESH = params['CONF_THRESH']

            inp     = preprocess(frame)
            outputs = rknn.inference(inputs=[inp])
            detections = postprocess(outputs, orig_h, orig_w, CONF_THRESH)

            best_y_x = None; best_w_x = None
            max_y_area = 0;  max_w_area = 0
            stop_requested = False
            current_time   = time.time()

            for (x1,y1,x2,y2,conf,cls) in detections:
                name = CLASS_NAMES.get(cls, '')
                cx   = (x1+x2)/2; cy = (y1+y2)/2
                area = (x2-x1)*(y2-y1)

                if name == 'redline' and cy > STOP_THRESHOLD_Y:
                    if (current_time - last_stop_time) > STOP_COOL:
                        stop_requested = True

                if cy < CAMERA_HEIGHT * ROI_CUTOFF:
                    continue

                if name == 'yellowline' and area > max_y_area:
                    max_y_area = area; best_y_x = cx
                elif name == 'whiteline' and area > max_w_area:
                    max_w_area = area; best_w_x = cx

            annotated = draw_detections(frame.copy(), detections)

            if stop_requested:
                print("!!! STOPPING FOR RED LINE !!!")
                stop_all()
                cv2.putText(annotated, "STOPPING FOR LINE", (50,240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                with lock: output_frame = annotated.copy()
                time.sleep(STOP_DUR)
                last_stop_time = time.time()
                continue

            if best_y_x is not None and best_w_x is not None:
                target_x = (best_y_x + best_w_x) / 2
            elif best_y_x is not None:
                target_x = best_y_x + (LANE_WIDTH / 2)
            elif best_w_x is not None:
                target_x = best_w_x - (LANE_WIDTH / 2)
            else:
                target_x = CENTER_X

            error      = target_x - CENTER_X
            derivative = error - prev_error
            prev_error = error
            steering   = (error * Kp) + (derivative * Kd)

            debug_lines = [
                f"Kp:{Kp:.4f}  Kd:{Kd:.4f}",
                f"best_w_x: {best_w_x}",
                f"best_y_x: {best_y_x}",
                f"target_x: {target_x:.1f}",
                f"error:    {error:.1f}",
                f"steering: {steering:.4f}",
                f"speed:    {BASE_SPEED:.2f}",
            ]
            for i, line in enumerate(debug_lines):
                y = 25 + i*22
                cv2.putText(annotated, line, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0),     3, cv2.LINE_AA)
                cv2.putText(annotated, line, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1, cv2.LINE_AA)

            debug_y = int(CAMERA_HEIGHT * ROI_CUTOFF) + 20
            cv2.circle(annotated, (int(target_x), debug_y), 10, (0,255,0), -1)
            cv2.line(annotated, (int(CENTER_X),0), (int(CENTER_X),CAMERA_HEIGHT), (255,255,255), 1)

            set_drive(BASE_SPEED, steering)

            with lock: output_frame = annotated.copy()

    except Exception as e:
        print(f"Robot Loop Error: {e}")
    finally:
        stop_all()
        rknn.release()
        print("Robot Loop Ended")


# --- Flask Routes ---
def generate_frames():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None: continue
            flag, enc = cv2.imencode('.jpg', output_frame)
            if not flag: continue
            data = bytearray(enc)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
        time.sleep(0.03)


@app.route('/')
def index():
    with params_lock:
        p = dict(params)
    return render_template_string("""
    <html>
    <head>
        <title>GooseBot Tuner</title>
        <style>
            body { background:#111; color:#eee; font-family:monospace; text-align:center; }
            img  { border:2px solid #555; margin-top:10px; display:block; margin:10px auto; }
            .panel { display:inline-block; background:#222; padding:20px; border-radius:8px;
                     margin:10px; text-align:left; vertical-align:top; }
            label { display:block; margin:8px 0 2px; font-size:0.85em; color:#aaa; }
            input[type=range] { width:200px; }
            input[type=number]{ width:80px; background:#333; color:#eee; border:1px solid #555;
                                padding:3px; border-radius:4px; }
            .val  { display:inline-block; width:60px; text-align:right; color:#4cf; }
            h1    { margin-bottom:4px; }
        </style>
    </head>
    <body>
        <h1>GooseBot Live Tuner</h1>
        <img src="{{ url_for('video_feed') }}" width="640" height="480">

        <div class="panel">
            <h3>PID / Speed</h3>

            <label>Kp <span class="val" id="Kp_val">{{ p.Kp }}</span></label>
            <input type="range" id="Kp" min="0.0001" max="0.003" step="0.0001"
                   value="{{ p.Kp }}" oninput="update('Kp', this.value)">

            <label>Kd <span class="val" id="Kd_val">{{ p.Kd }}</span></label>
            <input type="range" id="Kd" min="0.0001" max="0.003" step="0.0001"
                   value="{{ p.Kd }}" oninput="update('Kd', this.value)">

            <label>BASE_SPEED <span class="val" id="BASE_SPEED_val">{{ p.BASE_SPEED }}</span></label>
            <input type="range" id="BASE_SPEED" min="0.0" max="0.3" step="0.005"
                   value="{{ p.BASE_SPEED }}" oninput="update('BASE_SPEED', this.value)">

            <label>MAX_STEER <span class="val" id="MAX_STEER_val">{{ p.MAX_STEER }}</span></label>
            <input type="range" id="MAX_STEER" min="0.1" max="1.0" step="0.05"
                   value="{{ p.MAX_STEER }}" oninput="update('MAX_STEER', this.value)">
        </div>

        <div class="panel">
            <h3>Lane Detection</h3>

            <label>ROI_VERTICAL_CUTOFF <span class="val" id="ROI_VERTICAL_CUTOFF_val">{{ p.ROI_VERTICAL_CUTOFF }}</span></label>
            <input type="range" id="ROI_VERTICAL_CUTOFF" min="0.3" max="0.9" step="0.05"
                   value="{{ p.ROI_VERTICAL_CUTOFF }}" oninput="update('ROI_VERTICAL_CUTOFF', this.value)">

            <label>LANE_WIDTH_PIXELS <span class="val" id="LANE_WIDTH_PIXELS_val">{{ p.LANE_WIDTH_PIXELS }}</span></label>
            <input type="range" id="LANE_WIDTH_PIXELS" min="100" max="640" step="10"
                   value="{{ p.LANE_WIDTH_PIXELS }}" oninput="update('LANE_WIDTH_PIXELS', this.value)">

            <label>CONF_THRESH <span class="val" id="CONF_THRESH_val">{{ p.CONF_THRESH }}</span></label>
            <input type="range" id="CONF_THRESH" min="0.1" max="0.9" step="0.05"
                   value="{{ p.CONF_THRESH }}" oninput="update('CONF_THRESH', this.value)">
        </div>

        <div class="panel">
            <h3>Stop Line</h3>

            <label>STOP_DURATION <span class="val" id="STOP_DURATION_val">{{ p.STOP_DURATION }}</span></label>
            <input type="range" id="STOP_DURATION" min="0.5" max="10.0" step="0.5"
                   value="{{ p.STOP_DURATION }}" oninput="update('STOP_DURATION', this.value)">

            <label>STOP_COOLDOWN <span class="val" id="STOP_COOLDOWN_val">{{ p.STOP_COOLDOWN }}</span></label>
            <input type="range" id="STOP_COOLDOWN" min="1.0" max="20.0" step="0.5"
                   value="{{ p.STOP_COOLDOWN }}" oninput="update('STOP_COOLDOWN', this.value)">
        </div>

        <script>
            function update(key, val) {
                val = parseFloat(val);
                document.getElementById(key + '_val').innerText = val.toFixed(4);
                fetch('/set_param', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({key: key, value: val})
                });
            }
        </script>
    </body>
    </html>
    """, p=p)


@app.route('/set_param', methods=['POST'])
def set_param():
    data = request.get_json()
    key, value = data.get('key'), data.get('value')
    if key in params:
        with params_lock:
            params[key] = float(value)
        print(f"Updated {key} = {value}")
        return jsonify(success=True)
    return jsonify(success=False), 400


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# --- Main ---
if __name__ == "__main__":
    t = threading.Thread(target=robot_control_loop, daemon=True)
    t.start()
    print(f"Starting Web Server at http://{HOST_IP}:{HOST_PORT}")
    try:
        app.run(host=HOST_IP, port=HOST_PORT, debug=False,
                threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("Stopping...")
