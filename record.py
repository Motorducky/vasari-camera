#!/usr/bin/env python3
"""
Vasari Camera Recorder
Records all 3 OAK-D camera feeds (RGB + stereo pair) at full resolution.
"""

import cv2
import depthai as dai
import os
from datetime import datetime

# Configuration
SAVE_DIR = "/Users/corbettgriffith/Downloads/Varari-Desk-Cam-Recordings"

# Full resolutions for OAK-D cameras
RGB_W, RGB_H = 1920, 1080  # RGB camera (can go up to 4K: 3840x2160)
MONO_W, MONO_H = 1280, 800  # Mono/stereo cameras
FPS = 30

def main():
    # Create save directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Generate filenames with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rgb_filename = os.path.join(SAVE_DIR, f"vasari_{timestamp}_rgb.mp4")
    left_filename = os.path.join(SAVE_DIR, f"vasari_{timestamp}_left.mp4")
    right_filename = os.path.join(SAVE_DIR, f"vasari_{timestamp}_right.mp4")

    # Set up video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_rgb = cv2.VideoWriter(rgb_filename, fourcc, FPS, (RGB_W, RGB_H))
    out_left = cv2.VideoWriter(left_filename, fourcc, FPS, (MONO_W, MONO_H))
    out_right = cv2.VideoWriter(right_filename, fourcc, FPS, (MONO_W, MONO_H))

    print(f"Recording to: {SAVE_DIR}")
    print(f"  RGB:   {rgb_filename} ({RGB_W}x{RGB_H})")
    print(f"  Left:  {left_filename} ({MONO_W}x{MONO_H})")
    print(f"  Right: {right_filename} ({MONO_W}x{MONO_H})")
    print("\nPress 'q' to stop recording")
    print("-" * 50)

    # Create DepthAI pipeline
    pipeline = dai.Pipeline()

    # RGB Camera (CAM_A)
    cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    rgb_out = cam_rgb.requestOutput((RGB_W, RGB_H), dai.ImgFrame.Type.BGR888p)
    q_rgb = rgb_out.createOutputQueue(maxSize=2, blocking=False)

    # Left Mono Camera (CAM_B)
    cam_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    left_out = cam_left.requestOutput((MONO_W, MONO_H), dai.ImgFrame.Type.GRAY8)
    q_left = left_out.createOutputQueue(maxSize=2, blocking=False)

    # Right Mono Camera (CAM_C)
    cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    right_out = cam_right.requestOutput((MONO_W, MONO_H), dai.ImgFrame.Type.GRAY8)
    q_right = right_out.createOutputQueue(maxSize=2, blocking=False)

    frame_count = 0

    try:
        with pipeline:
            print("Recording started...")

            while pipeline.isRunning():
                # Get frames from all cameras
                rgb_msg = q_rgb.tryGet()
                left_msg = q_left.tryGet()
                right_msg = q_right.tryGet()

                # Write RGB frame
                if rgb_msg is not None:
                    rgb_frame = rgb_msg.getCvFrame()
                    out_rgb.write(rgb_frame)

                # Write Left frame (convert grayscale to BGR for video)
                if left_msg is not None:
                    left_frame = left_msg.getCvFrame()
                    left_bgr = cv2.cvtColor(left_frame, cv2.COLOR_GRAY2BGR)
                    out_left.write(left_bgr)

                # Write Right frame (convert grayscale to BGR for video)
                if right_msg is not None:
                    right_frame = right_msg.getCvFrame()
                    right_bgr = cv2.cvtColor(right_frame, cv2.COLOR_GRAY2BGR)
                    out_right.write(right_bgr)
                    frame_count += 1  # Count based on right camera

                # Show preview (scaled down RGB)
                if rgb_msg is not None:
                    preview = cv2.resize(rgb_frame, (640, 360))
                    cv2.imshow("Recording (press 'q' to stop)", preview)

                # Print progress every 30 frames (1 second)
                if frame_count > 0 and frame_count % 30 == 0:
                    elapsed = frame_count / FPS
                    print(f"\rRecorded: {elapsed:.1f}s ({frame_count} frames)", end="", flush=True)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nRecording interrupted")
    finally:
        out_rgb.release()
        out_left.release()
        out_right.release()
        cv2.destroyAllWindows()

        duration = frame_count / FPS
        print(f"\n\nRecording complete!")
        print(f"Duration: {duration:.1f} seconds ({frame_count} frames)")
        print(f"\nSaved files:")
        print(f"  {rgb_filename}")
        print(f"  {left_filename}")
        print(f"  {right_filename}")

if __name__ == "__main__":
    main()
