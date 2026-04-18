import time 
import os 
import cv2 
import numpy as np 
import RPi.GPIO as GPIO 
from picamera2 import Picamera2 
import threading
from collections import deque

# --- GPIO SETUP --- 
GPIO.setmode(GPIO.BCM) 
ENA, IN1, IN2 = 12, 23, 24 
ENB, IN3, IN4 = 13, 17, 27 
GPIO.setup([IN1, IN2, ENA, IN3, IN4, ENB], GPIO.OUT) 
pwmA = GPIO.PWM(ENA, 1000); pwmB = GPIO.PWM(ENB, 1000) 
pwmA.start(0); pwmB.start(0) 

# --- SETTINGS & GLOBALS --- 
last_error = 0 
color_entry_side = None  # Memory for Left-In/Left-Out recovery
SAVE_DIR = "templates" 
os.makedirs(SAVE_DIR, exist_ok=True) 

GOOD_MATCH_DIST = 50 
MIN_MATCH_COUNT = 10 
ROI_START = 0.55         
REQUIRED_FRAMES = 5    
detection_frames = 0   
COOLDOWN_UNTIL = 0     
stop_until = 0     

# Thresholds for switching logic
COLOR_THRESHOLD = 800  # Min pixels to prioritize color following
BLACK_THRESHOLD = 500  # Min pixels to consider black line valid

# Threading Globals
frame_buffer = deque(maxlen=1)
running = True

# --- HSV THRESHOLDS ---
HSV_THRESHOLDS = {
    "black":  {"low": np.array([0, 0, 0]),     "high": np.array([180, 255, 60])},
    "yellow": {"low": np.array([20, 100, 100]), "high": np.array([35, 255, 255])},
    "red1":   {"low": np.array([0, 100, 100]),  "high": np.array([10, 255, 255])},
    "red2":   {"low": np.array([160, 100, 100]), "high": np.array([180, 255, 255])}
}

# States 
STATE_FOLLOWING = 0 
STATE_STOPPED = 1 
STATE_FORCED_TURN = 2 
STATE_RECYCLING = 3 

current_state = STATE_FOLLOWING 
forced_turn_side = None 
forced_turn_until = 0 
recycle_until = 0 
RECYCLE_DURATION = 1.8 

# Initialize ORB 
orb = cv2.ORB_create(nfeatures=500) 
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) 

# --- MOTOR FUNCTIONS --- 
def stop_motors(): 
    GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW) 
    pwmA.ChangeDutyCycle(0); pwmB.ChangeDutyCycle(0) 

def move_robot(error, pixel_count): 
    global last_error 
    BASE_SPEED, PIVOT_SPEED, MAX_STEERING = 30, 50, 35 
    Kp, Kd = 4.5, 3.0 
     
    if error is not None: 
        steering = np.clip((error * Kp) + ((error - last_error) * Kd), -MAX_STEERING, MAX_STEERING) 
        last_error = error 
        l_pwr, r_pwr = BASE_SPEED + steering, BASE_SPEED - steering 
        GPIO.output([IN1, IN3], GPIO.LOW); GPIO.output([IN2, IN4], GPIO.HIGH) 
    else: 
        l_pwr = PIVOT_SPEED if last_error > 0 else -PIVOT_SPEED 
        r_pwr = -PIVOT_SPEED if last_error > 0 else PIVOT_SPEED 
        GPIO.output(IN1, GPIO.LOW if l_pwr > 0 else GPIO.HIGH) 
        GPIO.output(IN2, GPIO.HIGH if l_pwr > 0 else GPIO.LOW) 
        GPIO.output(IN3, GPIO.LOW if r_pwr > 0 else GPIO.HIGH) 
        GPIO.output(IN4, GPIO.HIGH if r_pwr > 0 else GPIO.LOW) 
         
    pwmA.ChangeDutyCycle(max(0, min(100, abs(l_pwr)))) 
    pwmB.ChangeDutyCycle(max(0, min(100, abs(r_pwr)))) 

# --- VISION FUNCTIONS --- 
def get_skeleton(img): 
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) 
    skel = np.zeros(binary.shape, np.uint8) 
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)) 
    while True: 
        open_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, element) 
        temp = cv2.subtract(binary, open_img) 
        eroded = cv2.erode(binary, element) 
        skel = cv2.bitwise_or(skel, temp) 
        binary = eroded.copy() 
        if cv2.countNonZero(binary) == 0: break 
    return skel 

def load_templates(): 
    tpls = {} 
    if not os.path.exists(SAVE_DIR): return tpls 
    for f in os.listdir(SAVE_DIR): 
        if f.lower().endswith(".png"): 
            img = cv2.imread(os.path.join(SAVE_DIR, f), cv2.IMREAD_GRAYSCALE) 
            if img is not None: 
                skel = get_skeleton(cv2.resize(img, (120, 120))) 
                kp, des = orb.detectAndCompute(skel, None) 
                if des is not None: tpls[f.replace(".png", "")] = des 
    return tpls 

def detect_and_crop_symbol(frame_rgb): 
    H, W, _ = frame_rgb.shape 
    roi = frame_rgb[0:int(H * ROI_START), :]  
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV) 
    
    color_mask = cv2.medianBlur(cv2.inRange(hsv, HSV_THRESHOLDS["black"]["low"], HSV_THRESHOLDS["black"]["high"]), 5)  
    
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) 
    bin_inv = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5, 5), 0), 255,   
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5) 
    bin_clean = cv2.bitwise_and(bin_inv, color_mask) 
    bin_clean = cv2.dilate(bin_clean, np.ones((3, 3), np.uint8), iterations=1) 
    weld = cv2.morphologyEx(bin_clean, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8)) 
    contours, _ = cv2.findContours(weld, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    if not contours: return None, None, bin_clean, None 
    c = max(contours, key=cv2.contourArea) 
    if cv2.contourArea(c) < 600: return None, None, bin_clean, None 
    x, y, w, h = cv2.boundingRect(c) 
    pad = 10 
    crop = bin_clean[max(0, y-pad):min(int(H*ROI_START), y+h+pad), max(0, x-pad):min(W, x+w+pad)] 
    return crop, (x, y, x+w, y+h), bin_clean, c 

def get_line_error(frame_rgb): 
    small = cv2.resize(frame_rgb, (160, 120)) 
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV) 
    
    m1 = cv2.inRange(hsv, HSV_THRESHOLDS["red1"]["low"], HSV_THRESHOLDS["red1"]["high"]) 
    m2 = cv2.inRange(hsv, HSV_THRESHOLDS["red2"]["low"], HSV_THRESHOLDS["red2"]["high"]) 
    m3 = cv2.inRange(hsv, HSV_THRESHOLDS["yellow"]["low"], HSV_THRESHOLDS["yellow"]["high"]) 
    color_mask = cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3) 
    
    black_mask = cv2.inRange(hsv, HSV_THRESHOLDS["black"]["low"], HSV_THRESHOLDS["black"]["high"])
    
    color_roi = color_mask[70:120, 0:160] 
    black_roi = black_mask[70:120, 0:160]
    
    c_px = cv2.countNonZero(color_roi)
    b_px = cv2.countNonZero(black_roi)
    
    if c_px > COLOR_THRESHOLD:
        active_roi = color_roi
    elif b_px > BLACK_THRESHOLD:
        active_roi = black_roi
    else:
        return None, 0, color_roi 

    M = cv2.moments(active_roi) 
    if M['m00'] > 0: 
        error = int(M['m10'] / M['m00']) - 80 
        
        left_count = cv2.countNonZero(active_roi[:, 0:80]) 
        right_count = cv2.countNonZero(active_roi[:, 80:160]) 
        if (left_count + right_count) > 2500: 
            error = -40 if left_count > right_count else 40 
            
        return error, M['m00']/255, active_roi 
    return None, 0, active_roi 

def capture_thread(picam2):
    global running
    while running:
        frame = picam2.capture_array()
        frame_buffer.append(frame)

# --- MAIN LOOP --- 
templates = load_templates() 
picam2 = Picamera2() 
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}) 
config['buffer_count'] = 1 
picam2.configure(config) 
picam2.start() 

cam_t = threading.Thread(target=capture_thread, args=(picam2,), daemon=True)
cam_t.start()

print(f"Ready. Templates: {list(templates.keys())}") 
input(">>> Press ENTER to start") 

try: 
    while True: 
        if not frame_buffer:
            continue
            
        frame = frame_buffer[0]
        now = time.time() 
        crop_mask, symbol_box, bin_clean, best_contour = detect_and_crop_symbol(frame) 

        # --- PRE-PROCESS COLOR FOR PIXEL CENSUS ---
        small = cv2.resize(frame, (160, 120))
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        m_r1 = cv2.inRange(hsv, HSV_THRESHOLDS["red1"]["low"], HSV_THRESHOLDS["red1"]["high"])
        m_r2 = cv2.inRange(hsv, HSV_THRESHOLDS["red2"]["low"], HSV_THRESHOLDS["red2"]["high"])
        m_y = cv2.inRange(hsv, HSV_THRESHOLDS["yellow"]["low"], HSV_THRESHOLDS["yellow"]["high"])
        color_mask = cv2.bitwise_or(cv2.bitwise_or(m_r1, m_r2), m_y)
        
        left_px = cv2.countNonZero(color_mask[70:120, 0:80])
        right_px = cv2.countNonZero(color_mask[70:120, 80:160])
        total_color = left_px + right_px

        if current_state == STATE_FOLLOWING: 
            # 1. CHECK FOR TASK A: ARROWS
            if best_contour is not None and now > COOLDOWN_UNTIL: 
                detection_frames += 1 
                if detection_frames >= REQUIRED_FRAMES: 
                    best_name, max_matches = "Unknown", 0 
                    if crop_mask is not None and len(templates) > 0: 
                        live_skel = get_skeleton(cv2.resize(crop_mask, (120, 120))) 
                        kp_live, des_live = orb.detectAndCompute(live_skel, None) 
                        if des_live is not None: 
                            for name, des_template in templates.items(): 
                                matches = bf.match(des_template, des_live) 
                                good = [m for m in matches if m.distance < GOOD_MATCH_DIST] 
                                if len(good) > max_matches: 
                                    max_matches = len(good); best_name = name 

                    if max_matches > MIN_MATCH_COUNT: 
                        name_low = best_name.lower()
                        print(f"TASK A MATCH: {best_name}")
                        if "left" in name_low: 
                            forced_turn_side = "left"; stop_until = now + 1.2; current_state = STATE_STOPPED
                        elif "right" in name_low: 
                            forced_turn_side = "right"; stop_until = now + 1.2; current_state = STATE_STOPPED
                        elif "recycle" in name_low:
                            recycle_until = now + RECYCLE_DURATION; current_state = STATE_RECYCLING
                        elif "danger" in name_low or "button" in name_low:
                            stop_until = now + 5.0; current_state = STATE_STOPPED
                        else: 
                            stop_until = now + 2.0; current_state = STATE_STOPPED
                    detection_frames = 0 
            
            # 2. CHECK FOR TASK B: COLOR LINE CENSUS (Integrated Memory Logic)
            elif total_color > 600 and now > COOLDOWN_UNTIL:
                # Capture entry side memory (Left-In/Right-In)
                if color_entry_side is None:
                    color_entry_side = "left" if left_px > right_px else "right"
                    print(f"Locked color entry side: {color_entry_side}")
                
                err, count, _ = get_line_error(frame) 
                move_robot(err, count) 

            # Handle "Left-Out/Right-Out" when color is lost but memory is active
            elif color_entry_side is not None and total_color < 100:
                print(f"Color lost! Memory Pivot: {color_entry_side}")
                last_error = -40 if color_entry_side == "left" else 40
                move_robot(None, 0) # Force pivot in entry direction

            else: 
                detection_frames = 0 
                err, count, _ = get_line_error(frame) 
                move_robot(err, count) 
                
                # Reset memory after 2 seconds of clean black line following
                if now > COOLDOWN_UNTIL + 2:
                    color_entry_side = None

        elif current_state == STATE_STOPPED: 
            stop_motors() 
            if now >= stop_until: 
                if forced_turn_side: 
                    current_state = STATE_FORCED_TURN; forced_turn_until = now + 5.0 
                else: 
                    COOLDOWN_UNTIL = now + 3.0; current_state = STATE_FOLLOWING 

        elif current_state == STATE_FORCED_TURN: 
            black_mask = cv2.inRange(hsv, HSV_THRESHOLDS["black"]["low"], HSV_THRESHOLDS["black"]["high"])
            combined_mask = cv2.bitwise_or(color_mask, black_mask)
            roi = combined_mask[70:120, 0:80] if forced_turn_side == "left" else combined_mask[70:120, 80:160] 
            
            if cv2.countNonZero(roi) > 500: 
                print(f"Line acquired on {forced_turn_side}"); forced_turn_side = None 
                color_entry_side = None # Clear memory after success
                COOLDOWN_UNTIL = now + 2.0; current_state = STATE_FOLLOWING 
            else: 
                last_error = -40 if forced_turn_side == "left" else 40; move_robot(None, 0) 
            
            if now > forced_turn_until: 
                forced_turn_side = None; current_state = STATE_FOLLOWING

        elif current_state == STATE_RECYCLING:
            last_error = 40; move_robot(None, 0)
            if now >= recycle_until:
                COOLDOWN_UNTIL = now + 2.0; current_state = STATE_FOLLOWING

        cv2.imshow("View", frame) 
        if cv2.waitKey(1) & 0xFF == ord('q'): break 

finally: 
    running = False
    stop_motors(); picam2.stop(); GPIO.cleanup(); cv2.destroyAllWindows()
