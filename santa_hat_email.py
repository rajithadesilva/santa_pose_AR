"""
santa_hat.py

Flow:
- Capture mode:
    - SPACE: take a photo and go to email entry
    - q: quit
    - ESC: toggle fullscreen on/off
- Email entry mode:
    - Type email address and press ENTER to "send"
    - Email is validated with a regex
    - If invalid -> "Incorrect email" message for 2s, then return to email entry
      WITHOUT clearing what you typed
    - After 3 invalid attempts -> warning and return to capture mode (input cleared)
    - Press ESC to cancel and return to capture mode
- If email is valid:
    - Calls email stub
    - Shows 4s confirmation:
      "Photo emailed to: X. Please check your inbox and junk/spam folder."
    - Then return to capture mode

Fullscreen:
- Starts fullscreen on Ubuntu
- ESC toggles fullscreen on/off (except in email mode, where it cancels email)
- q always quits
"""

import cv2
import numpy as np
import os
import re
import time
import mediapipe as mp

# ========== DEBUG / visualisation ==========
DEBUG = True  # Set True to draw full landmark mesh on faces

# ========== Email validation ==========
EMAIL_REGEX = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"


def is_valid_email(email: str) -> bool:
    return re.match(EMAIL_REGEX, email) is not None


# ========== MediaPipe setup ==========
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=10,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Landmark indices (MediaPipe Face Mesh)
FOREHEAD_LM = 10
LEFT_CHEEK_LM = 234
RIGHT_CHEEK_LM = 454

# ========== Hat / appearance settings ==========
HAT_SCALE_FACTOR = 1.6      # Overall scale relative to cheek distance
HAT_VERTICAL_FACTOR = 0.55  # How much of hat height sits above forehead
HAT_ROTATION_FACTOR = 0.2   # -1.0: -45°, 0: no extra offset, +1.0: +45°

# Screen resolution (adjust to your screen if needed)
screen_width = 1920
screen_height = 1080

# ========== Camera setup ==========
cap = cv2.VideoCapture(0)  # Adjust this index if you have multiple webcams
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# ========== Load Santa hat ==========
santa_hat = cv2.imread("santa_hat.png", cv2.IMREAD_UNCHANGED)  # Must have alpha
if santa_hat is None:
    print("Error: Could not load santa_hat.png")
    cap.release()
    exit()

if santa_hat.shape[2] < 4:
    print("Error: santa_hat.png has no alpha channel. Use a PNG with transparency.")
    cap.release()
    exit()

# ========== Output directory & naming ==========
output_dir = "photos"
os.makedirs(output_dir, exist_ok=True)


def get_starting_photo_count(directory):
    existing_files = os.listdir(directory)
    numbers = [
        int(re.search(r"photo_(\d+)", f).group(1))
        for f in existing_files
        if re.search(r"photo_(\d+)", f)
    ]
    return max(numbers) + 1 if numbers else 0


photo_count = get_starting_photo_count(output_dir)

# ========== Utility: alpha blending with clipping ==========
def overlay_image_alpha(background, overlay, x, y):
    """
    Overlay RGBA `overlay` onto BGR `background` at position (x, y) (top-left),
    handling alpha and clipping to image borders.
    """
    h, w = overlay.shape[:2]

    # Completely outside
    if x >= background.shape[1] or y >= background.shape[0] or x + w <= 0 or y + h <= 0:
        return background

    # Clip to valid ROI in background
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, background.shape[1])
    y2 = min(y + h, background.shape[0])

    # Corresponding region in overlay
    overlay_x1 = x1 - x
    overlay_y1 = y1 - y
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    overlay_cropped = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

    if overlay_cropped.shape[2] < 4:
        return background

    alpha = overlay_cropped[:, :, 3] / 255.0
    alpha = alpha[..., np.newaxis]

    bg_roi = background[y1:y2, x1:x2, :3]

    # Blend
    background[y1:y2, x1:x2, :3] = alpha * overlay_cropped[:, :, :3] + (1 - alpha) * bg_roi

    return background


# ========== Utility: draw rotated hat per face ==========
def add_hat_to_face(frame, face_landmarks):
    """
    Given a frame (BGR) and MediaPipe face_landmarks,
    compute head tilt & scale from cheek landmarks and overlay a rotated hat.
    """

    h, w, _ = frame.shape

    def lm_xy(idx):
        lm = face_landmarks.landmark[idx]
        return int(lm.x * w), int(lm.y * h)

    # Get key points
    fx, fy = lm_xy(FOREHEAD_LM)
    lx, ly = lm_xy(LEFT_CHEEK_LM)
    rx, ry = lm_xy(RIGHT_CHEEK_LM)

    # Distance between cheeks -> hat size
    dx = rx - lx
    dy = ry - ly
    cheek_dist = np.sqrt(dx * dx + dy * dy)
    if cheek_dist < 1:
        return frame  # too small / degenerate

    # Compute rotation angle (roll) based on cheek line
    base_angle_deg = -np.degrees(np.arctan2(dy, dx))

    # Extra correction: map HAT_ROTATION_FACTOR [-1,1] -> [-45°, +45°]
    extra_angle_deg = float(HAT_ROTATION_FACTOR) * 45.0
    angle_deg = base_angle_deg + extra_angle_deg

    # Scale hat relative to cheek distance
    base_hat_h, base_hat_w = santa_hat.shape[:2]
    scale = (cheek_dist * HAT_SCALE_FACTOR) / base_hat_w

    if scale <= 0:
        return frame

    # Resize hat
    resized_hat = cv2.resize(
        santa_hat, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )
    hat_h, hat_w = resized_hat.shape[:2]

    # Rotate hat around its centre
    center = (hat_w // 2, hat_h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated_hat = cv2.warpAffine(
        resized_hat,
        rot_mat,
        (hat_w, hat_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    # Position: centre hat horizontally on forehead, mostly above head
    hat_x = int(fx - hat_w / 2)
    hat_y = int(fy - hat_h * HAT_VERTICAL_FACTOR)

    frame = overlay_image_alpha(frame, rotated_hat, hat_x, hat_y)
    return frame


# ========== Simple placeholder for email sending ==========
def send_photo_via_email_stub(photo_path, email_address):
    """
    Placeholder for actual email sending logic.
    Replace this with your real emailing code (SMTP, API, etc.).
    """
    print(f"[EMAIL PLACEHOLDER] Would send '{photo_path}' to '{email_address}'")


# ========== OpenCV window + fullscreen control ==========
window_name = "MediaPipe Face Mesh Santa Hat"

# Start as normal window; fullscreen is applied after first imshow
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
FULLSCREEN = True
fullscreen_applied = False  # apply fullscreen once the window actually exists


def apply_fullscreen():
    global fullscreen_applied
    if FULLSCREEN:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    fullscreen_applied = True


# ========== State variables ==========
MODE_CAPTURE = "capture"
MODE_EMAIL = "email"
MODE_CONFIRM = "confirm"
MODE_EMAIL_ERROR = "email_error"
MODE_EMAIL_FAILOUT = "email_failout"

mode = MODE_CAPTURE

save_frame = None               # Last frame to save as photo
last_captured_frame = None      # For showing on email & confirm screens
last_photo_path = None          # File path of last captured photo

email_input = ""                # Current email text input
failed_attempts = 0             # Invalid email attempts

# Confirmation state
CONFIRM_DURATION = 4.0          # seconds
confirm_start_time = None
last_confirmation_email = None  # Email shown on confirmation screen

# Error / failout state
EMAIL_ERROR_DURATION = 2.0      # seconds
email_error_start_time = None

FAILOUT_DURATION = 3.0          # seconds
failout_start_time = None


# ========== UI drawing helpers ==========
def draw_capture_ui(frame):
    """Draw capture mode instructions."""
    text = "Press SPACE to take a photo | Press 'q' to quit | ESC: toggle fullscreen"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    text_x = (screen_width - text_size[0]) // 2
    text_y = screen_height - 20
    cv2.putText(
        frame,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    return frame


def draw_email_ui(base_frame, email_text):
    """
    Draw the email input overlay on top of the last captured frame.
    """
    frame = base_frame.copy()

    # Semi-transparent dark overlay at the bottom
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, int(screen_height * 0.65)),
        (screen_width, screen_height),
        (0, 0, 0),
        -1,
    )
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Instructions
    instr1 = "Type email address and press ENTER to send"
    instr2 = "Press ESC to cancel and go back to camera"
    email_label = "Email: " + email_text

    y_base = int(screen_height * 0.72)
    for i, text in enumerate([instr1, instr2]):
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        text_x = (screen_width - text_size[0]) // 2
        text_y = y_base + i * 35
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )

    # Email input line
    text_size, _ = cv2.getTextSize(email_label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
    text_x = (screen_width - text_size[0]) // 2
    text_y = int(screen_height * 0.90)
    cv2.putText(
        frame,
        email_label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 255, 0),
        2,
    )

    return frame


def draw_confirm_ui(base_frame, email_text):
    """
    Draw a confirmation message after "sending" the email.
    """
    frame = base_frame.copy()

    # Semi-transparent full overlay
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (int(screen_width * 0.1), int(screen_height * 0.3)),
        (int(screen_width * 0.9), int(screen_height * 0.7)),
        (0, 0, 0),
        -1,
    )
    alpha = 0.7
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Messages
    lines = [
        f"Photo emailed to: {email_text}",
        "Please check your inbox and junk/spam folder.",
        "Returning to camera..."
    ]

    y_start = int(screen_height * 0.4)
    for i, text in enumerate(lines):
        font_scale = 1.0 if i == 0 else 0.9
        thickness = 2
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = (screen_width - text_size[0]) // 2
        text_y = y_start + i * 50
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 255),
            thickness,
        )

    return frame


def draw_email_error_ui(base_frame):
    """
    Draw an 'incorrect email' warning for a short time.
    """
    frame = base_frame.copy()
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (int(screen_width * 0.15), int(screen_height * 0.4)),
        (int(screen_width * 0.85), int(screen_height * 0.6)),
        (0, 0, 0),
        -1,
    )
    alpha = 0.8
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    msg = "Incorrect email. Please re-enter."
    text_size, _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    text_x = (screen_width - text_size[0]) // 2
    text_y = int(screen_height * 0.52)
    cv2.putText(
        frame,
        msg,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )

    return frame


def draw_failout_ui(base_frame):
    """
    Draw a '3 failed attempts' warning before returning to capture mode.
    """
    frame = base_frame.copy()
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (int(screen_width * 0.12), int(screen_height * 0.4)),
        (int(screen_width * 0.88), int(screen_height * 0.6)),
        (0, 0, 0),
        -1,
    )
    alpha = 0.8
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    msg = "3 failed attempts. Returning to capture mode..."
    text_size, _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    text_x = (screen_width - text_size[0]) // 2
    text_y = int(screen_height * 0.52)
    cv2.putText(
        frame,
        msg,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )

    return frame


# ========== Main loop ==========
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    frame = cv2.flip(frame, 1)

    # Resize + letterbox
    orig_h, orig_w = frame.shape[:2]
    scale = min(screen_width / orig_w, screen_height / orig_h)
    resized_w = int(orig_w * scale)
    resized_h = int(orig_h * scale)

    frame = cv2.resize(frame, (resized_w, resized_h))

    pad_left = (screen_width - resized_w) // 2
    pad_right = screen_width - resized_w - pad_left
    pad_top = (screen_height - resized_h) // 2
    pad_bottom = screen_height - resized_h - pad_top

    frame = cv2.copyMakeBorder(
        frame,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    display = frame.copy()

    # ===== RENDERING PER MODE =====
    if mode == MODE_CAPTURE:
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                display = add_hat_to_face(display, face_landmarks)

                if DEBUG:
                    mp_drawing.draw_landmarks(
                        image=display,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                    )

        save_frame = display.copy()
        display = draw_capture_ui(display)

    elif mode == MODE_EMAIL:
        base_frame = last_captured_frame if last_captured_frame is not None else display
        display = draw_email_ui(base_frame, email_input)

    elif mode == MODE_CONFIRM:
        base_frame = last_captured_frame if last_captured_frame is not None else display
        display = draw_confirm_ui(base_frame, last_confirmation_email or "")
        if confirm_start_time is not None:
            elapsed = time.time() - confirm_start_time
            if elapsed >= CONFIRM_DURATION:
                mode = MODE_CAPTURE
                confirm_start_time = None
                last_confirmation_email = None

    elif mode == MODE_EMAIL_ERROR:
        base_frame = last_captured_frame if last_captured_frame is not None else display
        display = draw_email_error_ui(base_frame)
        if email_error_start_time is not None:
            elapsed = time.time() - email_error_start_time
            if elapsed >= EMAIL_ERROR_DURATION:
                mode = MODE_EMAIL
                email_error_start_time = None

    elif mode == MODE_EMAIL_FAILOUT:
        base_frame = last_captured_frame if last_captured_frame is not None else display
        display = draw_failout_ui(base_frame)
        if failout_start_time is not None:
            elapsed = time.time() - failout_start_time
            if elapsed >= FAILOUT_DURATION:
                mode = MODE_CAPTURE
                failout_start_time = None
                failed_attempts = 0
                email_input = ""

    cv2.imshow(window_name, display)

    # Apply fullscreen once the window exists
    if not fullscreen_applied:
        apply_fullscreen()

    key_raw = cv2.waitKey(1)
    if key_raw == -1:
        key_char = None
    else:
        key_char = key_raw & 0xFF

    # Global quit
    if key_char == ord("q"):
        break

    # ESC behaviour:
    # - In EMAIL mode: cancel email & go back to capture (no fullscreen toggle)
    # - In all other modes: toggle fullscreen
    if key_char == 27:  # ESC
        if mode == MODE_EMAIL:
            mode = MODE_CAPTURE
            email_input = ""
            failed_attempts = 0
        else:
            FULLSCREEN = not FULLSCREEN
            fullscreen_applied = False
        continue

    # ===== LOGIC PER MODE =====
    if mode == MODE_CAPTURE:
        if key_char == ord(" "):
            if save_frame is not None:
                photo_path = os.path.join(output_dir, f"photo_{photo_count}.png")
                cv2.imwrite(photo_path, save_frame)
                print(f"Photo saved: {photo_path}")
                photo_count += 1

                last_captured_frame = save_frame.copy()
                last_photo_path = photo_path
                email_input = ""
                failed_attempts = 0
                mode = MODE_EMAIL

    elif mode == MODE_EMAIL:
        if key_char in (13, 10):  # ENTER
            if last_photo_path is not None and email_input.strip():
                entered_email = email_input.strip()

                if not is_valid_email(entered_email):
                    failed_attempts += 1
                    print(f"Invalid email entered: '{entered_email}' (attempt {failed_attempts})")

                    if failed_attempts >= 3:
                        failout_start_time = time.time()
                        mode = MODE_EMAIL_FAILOUT
                    else:
                        email_error_start_time = time.time()
                        mode = MODE_EMAIL_ERROR

                    # DO NOT clear email_input: allow user to fix it
                else:
                    print(f"Valid email: {entered_email}")
                    print(f"Email entered: {entered_email}")
                    send_photo_via_email_stub(last_photo_path, entered_email)

                    last_confirmation_email = entered_email
                    confirm_start_time = time.time()
                    mode = MODE_CONFIRM
                    email_input = ""
                    failed_attempts = 0
            else:
                print("Empty email or no photo; nothing to send.")

        elif key_char in (8, 127):  # Backspace
            email_input = email_input[:-1]

        elif key_char is not None and 32 <= key_char <= 126:
            email_input += chr(key_char)

    # Other modes ignore keys except q and ESC (already handled)

# Cleanup
cap.release()
face_mesh.close()
cv2.destroyAllWindows()

