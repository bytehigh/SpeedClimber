import cv2
import mediapipe as mp
import time # Potentially useful

# --- Constants and Configuration ---
VIDEO_PATH = 'speed.mp4'
WINDOW_NAME = 'Pose and Hand Estimation'
SKIP_DURATION_SEC = 5
PROGRESS_BAR_HEIGHT = 10
PROGRESS_BAR_MARGIN_BOTTOM = 10
BUTTON_HEIGHT = 40
BUTTON_WIDTH = 120
BUTTON_MARGIN = 20
BUTTON_Y = 30
# Colors (B, G, R)
COLOR_ACTIVE_BUTTON = (0, 255, 0)
COLOR_INACTIVE_BUTTON = (200, 200, 200)
COLOR_TEXT = (0, 0, 0)
COLOR_PROGRESS_BAR_BG = (50, 50, 50)
COLOR_PROGRESS_BAR_FG = (0, 255, 0)
COLOR_MARKER_START = (255, 0, 0) # Blue
COLOR_MARKER_END = (0, 0, 255)   # Red
# Key codes
KEY_QUIT = ord('q')
KEY_SKIP_FORWARD = ord('d')
KEY_SKIP_BACKWARD = ord('a')
KEY_PAUSE = ord(' ') # Spacebar
KEY_MARK_START = ord('s')
KEY_MARK_END = ord('e')
KEY_JUMP_START = ord('r')

# --- MediaPipe Initialization ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# --- Helper Functions ---

def initialize_video(video_path):
    """Opens video, gets properties, returns cap and properties dict."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return None, None
    props = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': cap.get(cv2.CAP_PROP_FRAME_COUNT)
    }
    props['duration_sec'] = props['frame_count'] / props['fps'] if props['fps'] > 0 else 0
    props['duration_ms'] = props['duration_sec'] * 1000
    return cap, props

def initialize_mediapipe_models():
    """Initializes and returns pose and hand models."""
    pose_model = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands_model = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return pose_model, hands_model

def initialize_state(video_props):
    """Initializes the application state dictionary."""
    state = {
        'paused': False,
        'is_dragging': False,
        'crop_mode': 'full', # 'full', 'left', 'right'
        'start_time_ms': None,
        'end_time_ms': None,
        'current_frame': None,
        'current_time_ms': 0,
        'video_width': video_props['width'],
        'video_height': video_props['height'],
        'video_duration_ms': video_props['duration_ms'],
        'frame_count': video_props['frame_count'],
        'fps': video_props['fps'],
        # UI positions (can be updated dynamically if needed)
        'button_left_x': BUTTON_MARGIN,
        'button_right_x': video_props['width'] - BUTTON_WIDTH - BUTTON_MARGIN,
        'button_full_x': (video_props['width'] - BUTTON_WIDTH) // 2,
        'progress_bar_y_start': video_props['height'] - PROGRESS_BAR_HEIGHT - PROGRESS_BAR_MARGIN_BOTTOM,
        'progress_bar_y_end': video_props['height'] - PROGRESS_BAR_MARGIN_BOTTOM
    }
    return state

def get_current_frame_width(state):
    """Calculates frame width based on crop mode."""
    if state['crop_mode'] == 'left' or state['crop_mode'] == 'right':
        return state['video_width'] // 2
    else:
        return state['video_width']

def update_video_position_from_coord(x, frame_width, cap, state):
    """Sets video position based on x-coordinate relative to frame width."""
    if frame_width > 0 and state['frame_count'] > 0:
        new_pos_ratio = x / frame_width
        new_pos_frames = int(new_pos_ratio * state['frame_count'])
        new_pos_frames = max(0, min(new_pos_frames, state['frame_count'] - 1)) # Clamp
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos_frames)
        state['current_time_ms'] = cap.get(cv2.CAP_PROP_POS_MSEC) # Update state

def mouse_callback(event, x, y, flags, state):
    """Handles mouse events using the state dictionary."""
    frame_width = get_current_frame_width(state)
    cap = state['cap'] # Get cap from state

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check button clicks
        if BUTTON_Y <= y <= BUTTON_Y + BUTTON_HEIGHT:
            # Adjust button positions based on current frame width for click detection
            left_x = BUTTON_MARGIN
            right_x = frame_width - BUTTON_WIDTH - BUTTON_MARGIN
            full_x = (frame_width - BUTTON_WIDTH) // 2
            if left_x <= x <= left_x + BUTTON_WIDTH:
                state['crop_mode'] = 'left'
                return
            elif right_x <= x <= right_x + BUTTON_WIDTH:
                state['crop_mode'] = 'right'
                return
            elif full_x <= x <= full_x + BUTTON_WIDTH:
                state['crop_mode'] = 'full'
                return

        # Check progress bar click
        # Use original video height for progress bar y-coords
        pb_y_start = state['video_height'] - PROGRESS_BAR_HEIGHT - PROGRESS_BAR_MARGIN_BOTTOM
        pb_y_end = state['video_height'] - PROGRESS_BAR_MARGIN_BOTTOM
        if pb_y_start <= y <= pb_y_end:
            state['is_dragging'] = True
            update_video_position_from_coord(x, frame_width, cap, state)

    elif event == cv2.EVENT_MOUSEMOVE and state['is_dragging']:
        update_video_position_from_coord(x, frame_width, cap, state)

    elif event == cv2.EVENT_LBUTTONUP:
        state['is_dragging'] = False

def setup_ui(window_name, callback, state):
    """Creates window and sets mouse callback."""
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, callback, state) # Pass state dict as param

def draw_buttons(frame, state):
    """Draws crop mode buttons."""
    frame_width = frame.shape[1]
    crop_mode = state['crop_mode']
    # Calculate positions based on the *current* frame being drawn on
    left_x = BUTTON_MARGIN
    right_x = frame_width - BUTTON_WIDTH - BUTTON_MARGIN
    full_x = (frame_width - BUTTON_WIDTH) // 2

    cv2.rectangle(frame, (left_x, BUTTON_Y), (left_x + BUTTON_WIDTH, BUTTON_Y + BUTTON_HEIGHT),
                  COLOR_ACTIVE_BUTTON if crop_mode == 'left' else COLOR_INACTIVE_BUTTON, -1)
    cv2.putText(frame, 'Left', (left_x + 25, BUTTON_Y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)
    cv2.rectangle(frame, (right_x, BUTTON_Y), (right_x + BUTTON_WIDTH, BUTTON_Y + BUTTON_HEIGHT),
                  COLOR_ACTIVE_BUTTON if crop_mode == 'right' else COLOR_INACTIVE_BUTTON, -1)
    cv2.putText(frame, 'Right', (right_x + 20, BUTTON_Y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)
    cv2.rectangle(frame, (full_x, BUTTON_Y), (full_x + BUTTON_WIDTH, BUTTON_Y + BUTTON_HEIGHT),
                  COLOR_ACTIVE_BUTTON if crop_mode == 'full' else COLOR_INACTIVE_BUTTON, -1)
    cv2.putText(frame, 'Full', (full_x + 30, BUTTON_Y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)

def draw_progress_bar(frame, state):
    """Draws the progress bar and markers."""
    frame_height, frame_width = frame.shape[:2]
    pb_y_start = frame_height - PROGRESS_BAR_HEIGHT - PROGRESS_BAR_MARGIN_BOTTOM
    pb_y_end = frame_height - PROGRESS_BAR_MARGIN_BOTTOM
    duration_ms = state['video_duration_ms']
    current_time_ms = state['current_time_ms']
    start_ms = state['start_time_ms']
    end_ms = state['end_time_ms']

    # Background
    cv2.rectangle(frame, (0, pb_y_start), (frame_width, pb_y_end), COLOR_PROGRESS_BAR_BG, -1)
    # Foreground
    if duration_ms > 0:
         progress_ratio = current_time_ms / duration_ms
         progress_x = int(progress_ratio * frame_width)
         cv2.rectangle(frame, (0, pb_y_start), (progress_x, pb_y_end), COLOR_PROGRESS_BAR_FG, -1)

    # Markers
    if start_ms is not None and duration_ms > 0:
        start_x = int((start_ms / duration_ms) * frame_width)
        cv2.rectangle(frame, (start_x - 1, pb_y_start), (start_x + 1, pb_y_end), COLOR_MARKER_START, -1)
    if end_ms is not None and duration_ms > 0:
        end_x = int((end_ms / duration_ms) * frame_width)
        cv2.rectangle(frame, (end_x - 1, pb_y_start), (end_x + 1, pb_y_end), COLOR_MARKER_END, -1)

def handle_keyboard_input(key, cap, state):
    """Handles keyboard input and updates state. Returns True if quit."""
    if key == KEY_QUIT:
        return True # Signal to exit

    elif key == KEY_PAUSE:
        state['paused'] = not state['paused']

    elif key == KEY_SKIP_FORWARD:
        target_ms = state['current_time_ms'] + SKIP_DURATION_SEC * 1000
        if state['end_time_ms'] is not None:
            target_ms = min(target_ms, state['end_time_ms'])
        # Ensure target is within bounds
        target_ms = min(target_ms, state['video_duration_ms'])
        cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)
        state['current_time_ms'] = cap.get(cv2.CAP_PROP_POS_MSEC) # Update state

    elif key == KEY_SKIP_BACKWARD:
        target_ms = state['current_time_ms'] - SKIP_DURATION_SEC * 1000
        if state['start_time_ms'] is not None:
            target_ms = max(target_ms, state['start_time_ms'])
        target_ms = max(0, target_ms) # Ensure non-negative
        cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)
        state['current_time_ms'] = cap.get(cv2.CAP_PROP_POS_MSEC) # Update state

    elif key == KEY_MARK_START:
        state['start_time_ms'] = cap.get(cv2.CAP_PROP_POS_MSEC)
        print(f"Start point set to: {state['start_time_ms'] / 1000:.2f}s")

    elif key == KEY_MARK_END:
        state['end_time_ms'] = cap.get(cv2.CAP_PROP_POS_MSEC)
        print(f"End point set to: {state['end_time_ms'] / 1000:.2f}s")

    elif key == KEY_JUMP_START:
        if state['start_time_ms'] is not None:
            cap.set(cv2.CAP_PROP_POS_MSEC, state['start_time_ms'])
            state['current_time_ms'] = cap.get(cv2.CAP_PROP_POS_MSEC) # Update state
        else:
            print("Start point not set.")

    return False # Continue loop

def handle_looping(cap, state):
    """Checks if the video needs to loop based on markers or end."""
    looped = False
    loop_back_to_ms = state['start_time_ms'] if state['start_time_ms'] is not None else 0

    # Loop if end marker is set and reached
    if state['end_time_ms'] is not None and state['current_time_ms'] >= state['end_time_ms']:
        cap.set(cv2.CAP_PROP_POS_MSEC, loop_back_to_ms)
        state['current_time_ms'] = cap.get(cv2.CAP_PROP_POS_MSEC)
        looped = True

    # Check if read failed (potentially end of video without end marker)
    elif state['current_frame'] is None and not state['paused']:
         cap.set(cv2.CAP_PROP_POS_MSEC, loop_back_to_ms)
         state['current_time_ms'] = cap.get(cv2.CAP_PROP_POS_MSEC)
         looped = True

    return looped

def read_frame(cap, state):
    """Reads a frame if not paused, updates state."""
    if not state['paused']:
        ret, frame = cap.read()
        if ret:
            state['current_frame'] = frame
            # Update time only if not dragging (dragging sets time directly)
            if not state['is_dragging']:
                 state['current_time_ms'] = cap.get(cv2.CAP_PROP_POS_MSEC)
        else:
            state['current_frame'] = None # Indicate end or error
    # If paused, state['current_frame'] remains the last valid frame

def process_mediapipe(frame, pose_model, hands_model):
    """Runs MediaPipe pose and hand detection on the frame."""
    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False # Performance optimization

    # Process
    pose_results = pose_model.process(frame_rgb)
    hand_results = hands_model.process(frame_rgb)

    frame_rgb.flags.writeable = True # Allow drawing
    return pose_results, hand_results

def draw_landmarks(frame, pose_results, hand_results):
    """Draws pose and hand landmarks on the frame."""
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

def crop_frame(frame, state):
    """Crops the frame based on the state's crop_mode."""
    if frame is None:
        return None
    if state['crop_mode'] == 'left':
        return frame[:, :state['video_width'] // 2]
    elif state['crop_mode'] == 'right':
        return frame[:, state['video_width'] // 2:]
    else:
        return frame # Return original frame (or copy if modification is intended later)

def cleanup(cap, pose_model, hands_model):
    """Releases resources."""
    print("Releasing resources...")
    if cap:
        cap.release()
    if pose_model:
        pose_model.close()
    if hands_model:
        hands_model.close()
    cv2.destroyAllWindows()
    print("Cleanup finished.")

# --- Main Execution ---
def main():
    cap = None
    pose_model = None
    hands_model = None
    try:
        # Initialization
        cap, video_props = initialize_video(VIDEO_PATH)
        if not cap:
            return # Exit if video failed to open

        pose_model, hands_model = initialize_mediapipe_models()
        app_state = initialize_state(video_props)
        app_state['cap'] = cap # Add cap to state for mouse callback access

        setup_ui(WINDOW_NAME, mouse_callback, app_state)

        # Main Loop
        while True:
            read_frame(cap, app_state)

            # Handle looping or end of video
            if handle_looping(cap, app_state):
                 read_frame(cap, app_state) # Read again after seeking

            # If still no frame after potential loop (e.g., video error), break
            if app_state['current_frame'] is None and not app_state['paused']:
                 print("Error reading frame or end of video reached unexpectedly.")
                 break

            # Get the frame to display (last valid one if paused)
            display_frame_orig = app_state['current_frame']
            if display_frame_orig is None:
                # Should only happen if paused on the very first frame read failure
                time.sleep(0.01) # Prevent busy-waiting if paused at end
                key = cv2.waitKey(1) & 0xFF # Still need to process keys when paused at end
                if handle_keyboard_input(key, cap, app_state): break
                continue

            # --- Processing and Drawing ---
            # 1. Crop the original frame
            cropped_frame = crop_frame(display_frame_orig, app_state)
            if cropped_frame is None or cropped_frame.shape[1] == 0: # Check if crop resulted in empty frame
                # Show black screen or skip display? Let's skip.
                key = cv2.waitKey(1) & 0xFF
                if handle_keyboard_input(key, cap, app_state): break
                continue

            # 2. Process the *cropped* frame with MediaPipe
            pose_results, hand_results = process_mediapipe(cropped_frame, pose_model, hands_model)

            # 3. Draw landmarks on the *cropped* frame
            # Make a copy to draw on, preserving the cropped_frame if needed elsewhere
            annotated_frame = cropped_frame.copy()
            draw_landmarks(annotated_frame, pose_results, hand_results)

            # 4. Draw UI elements on the *annotated cropped* frame
            draw_buttons(annotated_frame, app_state)
            draw_progress_bar(annotated_frame, app_state)

            # 5. Display the final frame
            cv2.imshow(WINDOW_NAME, annotated_frame)

            # --- Input Handling ---
            key = cv2.waitKey(1) & 0xFF # Crucial for display updates
            if handle_keyboard_input(key, cap, app_state):
                break # Exit loop if quit requested

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Cleanup
        cleanup(cap, pose_model, hands_model)

if __name__ == "__main__":
    main()