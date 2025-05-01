import cv2
import mediapipe as mp

# Initialize MediaPipe Pose and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

# Open video file
video_path = 'speed.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
progress_bar_height = 10

# Define skip duration in seconds
SKIP_DURATION = 5

# Variables to track mouse dragging
is_dragging = False
last_mouse_x = None

# Crop mode: 'full', 'left', or 'right'
crop_mode = 'full'

# Button overlay positions and sizes
button_height = 40
button_width = 120
button_margin = 20
button_y = 30
left_button_x = button_margin
right_button_x = video_width - button_width - button_margin
full_button_x = (video_width - button_width) // 2

# Add a variable to track pause state
paused = False

# Variables for start/end points (in milliseconds)
start_time_ms = None
end_time_ms = None

# Helper to draw buttons
def draw_buttons(frame, crop_mode):
    frame_width = frame.shape[1]
    # Recalculate button positions based on current frame width
    left_button_x = 20
    right_button_x = frame_width - button_width - 20
    full_button_x = (frame_width - button_width) // 2
    # Colors
    active_color = (0, 255, 0)
    inactive_color = (200, 200, 200)
    text_color = (0, 0, 0)
    # Left button
    cv2.rectangle(frame, (left_button_x, button_y), (left_button_x + button_width, button_y + button_height),
                  active_color if crop_mode == 'left' else inactive_color, -1)
    cv2.putText(frame, 'Left', (left_button_x + 25, button_y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
    # Right button
    cv2.rectangle(frame, (right_button_x, button_y), (right_button_x + button_width, button_y + button_height),
                  active_color if crop_mode == 'right' else inactive_color, -1)
    cv2.putText(frame, 'Right', (right_button_x + 20, button_y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
    # Full button
    cv2.rectangle(frame, (full_button_x, button_y), (full_button_x + button_width, button_y + button_height),
                  active_color if crop_mode == 'full' else inactive_color, -1)
    cv2.putText(frame, 'Full', (full_button_x + 30, button_y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

# Helper to draw start/end markers on progress bar
def draw_markers(frame, start_ms, end_ms, duration_ms, frame_width, pb_y_start, pb_y_end):
    if start_ms is not None:
        start_x = int((start_ms / duration_ms) * frame_width)
        cv2.rectangle(frame, (start_x - 1, pb_y_start), (start_x + 1, pb_y_end), (255, 0, 0), -1) # Blue marker
    if end_ms is not None:
        end_x = int((end_ms / duration_ms) * frame_width)
        cv2.rectangle(frame, (end_x - 1, pb_y_start), (end_x + 1, pb_y_end), (0, 0, 255), -1) # Red marker

# Function to handle mouse events
def mouse_callback(event, x, y, flags, param):
    global is_dragging, crop_mode
    # Get the current frame width (after cropping)
    if crop_mode == 'left' or crop_mode == 'right':
        frame_width = video_width // 2
    else:
        frame_width = video_width
    left_button_x = 20
    right_button_x = frame_width - button_width - 20
    full_button_x = (frame_width - button_width) // 2
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is on a button (using current frame width)
        if button_y <= y <= button_y + button_height:
            if left_button_x <= x <= left_button_x + button_width:
                crop_mode = 'left'
                return
            elif right_button_x <= x <= right_button_x + button_width:
                crop_mode = 'right'
                return
            elif full_button_x <= x <= full_button_x + button_width:
                crop_mode = 'full'
                return
        # Handle progress bar scrubbing
        if progress_bar_y_start <= y <= progress_bar_y_end:
            is_dragging = True
            # Set video position based on x position
            new_pos = (x / frame_width) * video_duration
            new_pos = max(0, min(new_pos, video_duration))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(new_pos * cap.get(cv2.CAP_PROP_FPS)))
    elif event == cv2.EVENT_MOUSEMOVE and is_dragging:
        # While dragging, update video position based on x
        new_pos = (x / frame_width) * video_duration
        new_pos = max(0, min(new_pos, video_duration))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(new_pos * cap.get(cv2.CAP_PROP_FPS)))
    elif event == cv2.EVENT_LBUTTONUP:
        is_dragging = False

# Adjust the progress bar to overlay on the video frame
progress_bar_y_start = video_height - progress_bar_height - 10  # 10 pixels above the bottom of the frame
progress_bar_y_end = video_height - 10  # 10 pixels above the bottom edge

# Set the mouse callback
cv2.namedWindow('Pose and Hand Estimation')
cv2.setMouseCallback('Pose and Hand Estimation', mouse_callback)

# Initialize Pose and Hand models
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        # If end point is set and current time exceeds it, loop back to start
        if end_time_ms is not None and current_time_ms >= end_time_ms:
            if start_time_ms is not None:
                cap.set(cv2.CAP_PROP_POS_MSEC, start_time_ms)
            else:
                cap.set(cv2.CAP_PROP_POS_MSEC, 0) # Loop back to beginning if no start time
            current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC) # Update current time after seeking

        if not paused:
            ret, frame = cap.read()
            if not ret:
                # If end of video reached and no end marker set, loop to start marker or beginning
                if start_time_ms is not None:
                    cap.set(cv2.CAP_PROP_POS_MSEC, start_time_ms)
                else:
                    cap.set(cv2.CAP_PROP_POS_MSEC, 0)
                continue # Read the frame again after seeking

            # Crop the frame *before* processing based on crop_mode
            if crop_mode == 'left':
                process_frame = frame[:, :video_width // 2]
            elif crop_mode == 'right':
                process_frame = frame[:, video_width // 2:]
            else:
                process_frame = frame

            # Convert the *cropped* frame to RGB
            frame_rgb = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)

            # Process the *cropped* frame for pose and hand landmarks
            pose_results = pose.process(frame_rgb)
            hand_results = hands.process(frame_rgb)

            # Draw pose landmarks on the *cropped* frame
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    process_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Draw hand landmarks on the *cropped* frame
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        process_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Draw the button overlays on the *cropped* frame
            draw_buttons(process_frame, crop_mode)

            # Draw the progress bar (use *cropped* frame width)
            current_pos_sec = current_time_ms / 1000
            progress = int((current_pos_sec / video_duration) * process_frame.shape[1])
            pb_y_start = process_frame.shape[0] - progress_bar_height - 10
            pb_y_end = process_frame.shape[0] - 10
            cv2.rectangle(process_frame, (0, pb_y_start), (process_frame.shape[1], pb_y_end), (50, 50, 50), -1)
            cv2.rectangle(process_frame, (0, pb_y_start), (progress, pb_y_end), (0, 255, 0), -1)

            # Draw start/end markers
            draw_markers(process_frame, start_time_ms, end_time_ms, video_duration * 1000, process_frame.shape[1], pb_y_start, pb_y_end)

            # Display the *processed and annotated cropped* frame
            cv2.imshow('Pose and Hand Estimation', process_frame)

        # Handle keyboard input for video controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):  # Skip forward
            current_pos = cap.get(cv2.CAP_PROP_POS_MSEC)
            new_pos = current_pos + SKIP_DURATION * 1000
            if end_time_ms is not None:
                new_pos = min(new_pos, end_time_ms)
            cap.set(cv2.CAP_PROP_POS_MSEC, new_pos)
        elif key == ord('a'):  # Skip backward
            current_pos = cap.get(cv2.CAP_PROP_POS_MSEC)
            new_pos = max(0, current_pos - SKIP_DURATION * 1000)
            if start_time_ms is not None:
                new_pos = max(new_pos, start_time_ms)
            cap.set(cv2.CAP_PROP_POS_MSEC, new_pos)
        elif key == 32:  # Spacebar to pause/resume
            paused = not paused
        elif key == ord('s'): # Mark start point
            start_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            print(f"Start point set to: {start_time_ms / 1000:.2f}s")
        elif key == ord('e'): # Mark end point
            end_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            print(f"End point set to: {end_time_ms / 1000:.2f}s")
        elif key == ord('r'): # Jump to start point
            if start_time_ms is not None:
                cap.set(cv2.CAP_PROP_POS_MSEC, start_time_ms)
            else:
                print("Start point not set.")

# Release resources
cap.release()
cv2.destroyAllWindows()