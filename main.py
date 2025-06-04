import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# ==== Setup ====
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# State
prev_mouse = np.array([0, 0])
last_move_time = time.time()
last_click_time = 0
click_feedback_pos = None
click_feedback_timer = 0

# === Config ===
alpha = 0.2
move_delay = 0.05
click_delay = 1.0
gaze_overlay_radius = 10
click_feedback_duration = 0.5  # seconds

print("🚀 Eye-controlled mouse system started. Press 'q' to quit. Press 's' to take screenshot.")

try:
    while True:
        ret, frame = cam.read()
        if not ret:
            print("❌ Camera frame failed.")
            break

        frame = cv2.flip(frame, 1)
        frame_small = cv2.resize(frame, (320, 240))
        rgb_frame = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        frame_h, frame_w, _ = frame_small.shape

        output = face_mesh.process(rgb_frame)

        if output.multi_face_landmarks:
            landmarks = output.multi_face_landmarks[0].landmark
            if len(landmarks) > 475:
                iris = landmarks[475]
                screen_x = screen_w * iris.x
                screen_y = screen_h * iris.y

                new_mouse = np.array([screen_x, screen_y])
                smooth_mouse = prev_mouse * (1 - alpha) + new_mouse * alpha

                if time.time() - last_move_time > move_delay:
                    pyautogui.moveTo(smooth_mouse[0], smooth_mouse[1])
                    last_move_time = time.time()
                prev_mouse = smooth_mouse

                # Overlay position (back to preview space)
                overlay_x = int(smooth_mouse[0] / screen_w * frame_w)
                overlay_y = int(smooth_mouse[1] / screen_h * frame_h)

                # Draw gaze overlay
                cv2.circle(frame_small, (overlay_x, overlay_y), gaze_overlay_radius, (0, 255, 255), 2)
                cv2.putText(frame_small, "👁️ Gaze", (overlay_x + 12, overlay_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                # Blink click detection (Left Eye)
                if len(landmarks) > 159:
                    eye_top = landmarks[159]
                    eye_bottom = landmarks[145]
                    ear = eye_bottom.y - eye_top.y
                    if ear < 0.018 and time.time() - last_click_time > click_delay:
                        pyautogui.click()
                        last_click_time = time.time()
                        click_feedback_pos = (overlay_x, overlay_y)
                        click_feedback_timer = time.time()
                        print("🖱️ Click!")

            # Click Feedback Animation
            if click_feedback_pos and time.time() - click_feedback_timer < click_feedback_duration:
                radius = int(20 * (1 - (time.time() - click_feedback_timer) / click_feedback_duration))
                cv2.circle(frame_small, click_feedback_pos, radius, (0, 150, 255), 2)

        else:
            cv2.putText(frame_small, "⚠️ Face Not Detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Show Preview Window
        cv2.imshow('👁️ Eye Mouse Pro', frame_small)

        # Handle Keys
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            print("👋 Quit triggered.")
            break
        elif key & 0xFF == ord('s'):
            ts = int(time.time())
            filename = f"screenshot_{ts}.png"
            cv2.imwrite(filename, frame_small)
            print(f"📸 Screenshot saved as {filename}")

finally:
    cam.release()
    cv2.destroyAllWindows()
    print("✅ Shutdown complete.")
