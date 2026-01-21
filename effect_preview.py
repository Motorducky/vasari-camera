"""
50 Visual Effects Preview Script
Cycles through effects on camera feed
Press SPACE to advance, 's' to save frame, 'q' to quit
"""

import cv2
import numpy as np
from datetime import datetime
import os

# RTSP stream URL - change to your camera
STREAM_URL = "rtsp://192.168.0.XXX:8554/camera_name"  # Update this
SAVE_DIR = "effect_samples"
os.makedirs(SAVE_DIR, exist_ok=True)

def apply_effect(frame, effect_num, depth_map=None):
    """Apply effect based on number. depth_map optional for stereo effects."""
    h, w = frame.shape[:2]
    
    # Generate fake depth map if none provided (will be grayscale gradient for demo)
    if depth_map is None:
        depth_map = np.linspace(0, 255, h).reshape(h, 1).repeat(w, axis=1).astype(np.uint8)
    
    if effect_num == 0:
        return frame, "Original"
    
    # INVERT VARIATIONS (Softer approaches)
    elif effect_num == 1:
        inv = cv2.bitwise_not(frame)
        result = cv2.addWeighted(frame, 0.5, inv, 0.5, 0)
        return result, "50% Invert Blend"
    
    elif effect_num == 2:
        inv = cv2.bitwise_not(frame)
        result = cv2.addWeighted(frame, 0.7, inv, 0.3, 0)
        return result, "30% Invert Blend"
    
    elif effect_num == 3:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:,:,0] = (hsv[:,:,0] + 90) % 180  # Hue rotate 180°
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), "Hue Rotate 180"
    
    elif effect_num == 4:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:,:,0] = (hsv[:,:,0] + 45) % 180
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), "Hue Rotate 90"
    
    elif effect_num == 5:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inv_gray = cv2.bitwise_not(gray)
        return cv2.cvtColor(inv_gray, cv2.COLOR_GRAY2BGR), "Luminance Invert Only"
    
    elif effect_num == 6:
        threshold = 128
        mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > threshold
        result = frame.copy()
        result[mask] = cv2.bitwise_not(frame)[mask]
        return result, "Solarization (Highlights Inverted)"
    
    elif effect_num == 7:
        b, g, r = cv2.split(frame)
        r_inv = cv2.bitwise_not(r)
        return cv2.merge([b, g, r_inv]), "Red Channel Inverted"
    
    elif effect_num == 8:
        b, g, r = cv2.split(frame)
        b_inv = cv2.bitwise_not(b)
        return cv2.merge([b_inv, g, r]), "Blue Channel Inverted"
    
    elif effect_num == 9:
        gradient = np.linspace(0, 1, h).reshape(h, 1, 1)
        inv = cv2.bitwise_not(frame)
        result = (frame * (1 - gradient) + inv * gradient).astype(np.uint8)
        return result, "Gradient Invert (Top to Bottom)"
    
    elif effect_num == 10:
        depth_norm = depth_map.astype(float) / 255.0
        depth_norm = depth_norm.reshape(h, w, 1)
        inv = cv2.bitwise_not(frame)
        result = (frame * (1 - depth_norm) + inv * depth_norm).astype(np.uint8)
        return result, "Depth-Based Invert"
    
    # DIGITIZATION EFFECTS
    elif effect_num == 11:
        block = 16
        small = cv2.resize(frame, (w//block, h//block), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST), "Pixelation 16x"
    
    elif effect_num == 12:
        block = 32
        small = cv2.resize(frame, (w//block, h//block), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST), "Pixelation 32x"
    
    elif effect_num == 13:
        block = 8
        small = cv2.resize(frame, (w//block, h//block), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST), "Pixelation 8x"
    
    elif effect_num == 14:
        # Depth-adaptive pixelation
        result = frame.copy()
        for i in range(0, h, 4):
            depth_row = int(depth_map[min(i, h-1), w//2] / 255.0 * 28) + 4
            small = cv2.resize(frame[i:i+depth_row, :], (w//depth_row, 1), interpolation=cv2.INTER_LINEAR)
            result[i:i+depth_row, :] = cv2.resize(small, (w, depth_row), interpolation=cv2.INTER_NEAREST)
        return result, "Depth-Adaptive Pixelation"
    
    elif effect_num == 15:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), "Binary Threshold Dither"
    
    elif effect_num == 16:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dither = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return cv2.cvtColor(dither, cv2.COLOR_GRAY2BGR), "Adaptive Dither"
    
    elif effect_num == 17:
        result = frame.copy()
        for i in range(0, h, 2):
            result[i:i+1, :] = 0
        return result, "Scanlines"
    
    elif effect_num == 18:
        block = 16
        small = cv2.resize(frame, (w//block, h//block))
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        result = pixelated.copy()
        for i in range(0, h, 2):
            result[i:i+1, :] = 0
        return result, "Pixelation + Scanlines"
    
    # DEPTH/STEREO EFFECTS
    elif effect_num == 19:
        depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 0.6, depth_color, 0.4, 0), "Depth Map Overlay"
    
    elif effect_num == 20:
        depth_norm = depth_map.astype(float) / 255.0
        blurred = cv2.GaussianBlur(frame, (21, 21), 0)
        depth_3ch = depth_norm.reshape(h, w, 1)
        result = (frame * (1 - depth_3ch) + blurred * depth_3ch).astype(np.uint8)
        return result, "Depth-Based Blur"
    
    elif effect_num == 21:
        depth_3ch = (depth_map / 255.0).reshape(h, w, 1)
        fog_color = np.array([200, 200, 255], dtype=np.uint8)
        result = (frame * (1 - depth_3ch) + fog_color * depth_3ch).astype(np.uint8)
        return result, "Depth Fog"
    
    elif effect_num == 22:
        edges = cv2.Canny(depth_map, 50, 150)
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(frame, 0.7, edges_color, 0.3, 0), "Depth Edge Detection"
    
    elif effect_num == 23:
        mask = (depth_map > 80) & (depth_map < 180)
        result = frame.copy()
        result[~mask] = 0
        return result, "Depth Slice (Mid-range Only)"
    
    elif effect_num == 24:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        depth_hue = (depth_map.astype(float) / 255.0 * 179).astype(np.uint8)
        hsv[:,:,0] = depth_hue
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), "Depth-Based Color Grading"
    
    elif effect_num == 25:
        # Anaglyph effect (fake stereo)
        shift = 5
        left = frame.copy()
        right = np.roll(frame, shift, axis=1)
        result = frame.copy()
        result[:,:,2] = left[:,:,2]  # Red from left
        result[:,:,0] = right[:,:,0]  # Blue from right (shifted)
        return result, "Anaglyph 3D (Red/Cyan)"
    
    # EDGE DETECTION VARIANTS
    elif effect_num == 26:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), "Canny Edge Detection"
    
    elif effect_num == 27:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_inv = cv2.bitwise_not(edges)
        return cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR), "Inverted Edges"
    
    elif effect_num == 28:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = np.clip(sobel, 0, 255).astype(np.uint8)
        return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR), "Sobel Edge Detection"
    
    elif effect_num == 29:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_color[:,:,1] = 0  # Remove green, keep red/blue edges
        return cv2.addWeighted(frame, 0.7, edges_color, 0.3, 0), "Colored Edge Overlay"
    
    # CHROMATIC EFFECTS
    elif effect_num == 30:
        b, g, r = cv2.split(frame)
        b_shift = np.roll(b, -3, axis=1)
        r_shift = np.roll(r, 3, axis=1)
        return cv2.merge([b_shift, g, r_shift]), "Chromatic Aberration"
    
    elif effect_num == 31:
        depth_shift = (depth_map / 255.0 * 10).astype(int)
        b, g, r = cv2.split(frame)
        result = frame.copy()
        for i in range(h):
            shift = depth_shift[i, w//2]
            b[i] = np.roll(b[i], -shift)
            r[i] = np.roll(r[i], shift)
        return cv2.merge([b, g, r]), "Depth-Based Chromatic Aberration"
    
    elif effect_num == 32:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = 255  # Max saturation
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), "Saturation Boost"
    
    elif effect_num == 33:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = hsv[:,:,1] // 2  # Half saturation
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), "Desaturated"
    
    elif effect_num == 34:
        b, g, r = cv2.split(frame)
        return cv2.merge([g, b, r]), "Channel Swap (BGR to GBR)"
    
    # COMBINATION EFFECTS
    elif effect_num == 35:
        # Pixelate + Invert blend
        block = 16
        small = cv2.resize(frame, (w//block, h//block))
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        inv = cv2.bitwise_not(pixelated)
        return cv2.addWeighted(pixelated, 0.6, inv, 0.4, 0), "Pixelation + 40% Invert"
    
    elif effect_num == 36:
        # Depth fog + digitization
        block = 12
        small = cv2.resize(frame, (w//block, h//block))
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        depth_3ch = (depth_map / 255.0).reshape(h, w, 1)
        fog = np.array([180, 180, 200], dtype=np.uint8)
        result = (pixelated * (1 - depth_3ch) + fog * depth_3ch).astype(np.uint8)
        return result, "Depth Fog + Pixelation"
    
    elif effect_num == 37:
        # Edges + Solarization
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        threshold = 128
        mask = gray > threshold
        result = frame.copy()
        result[mask] = cv2.bitwise_not(frame)[mask]
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(result, 0.8, edges_color, 0.2, 0), "Solarization + Edges"
    
    elif effect_num == 38:
        # Hue rotate + pixelation
        block = 20
        small = cv2.resize(frame, (w//block, h//block))
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        hsv = cv2.cvtColor(pixelated, cv2.COLOR_BGR2HSV)
        hsv[:,:,0] = (hsv[:,:,0] + 90) % 180
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), "Pixelation + Hue Shift"
    
    elif effect_num == 39:
        # Depth slicing + color
        result = np.zeros_like(frame)
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
        for i, (low, high) in enumerate([(0,64), (64,128), (128,192), (192,255)]):
            mask = (depth_map >= low) & (depth_map < high)
            result[mask] = colors[i % len(colors)]
        return cv2.addWeighted(frame, 0.5, result, 0.5, 0), "Depth Slices (Colored Layers)"
    
    elif effect_num == 40:
        # Scanlines + chromatic aberration
        b, g, r = cv2.split(frame)
        r_shift = np.roll(r, 2, axis=1)
        b_shift = np.roll(b, -2, axis=1)
        result = cv2.merge([b_shift, g, r_shift])
        for i in range(0, h, 2):
            result[i:i+1, :] = result[i:i+1, :] // 2
        return result, "Scanlines + Chromatic"
    
    # GLITCH/DISPLACEMENT EFFECTS
    elif effect_num == 41:
        result = frame.copy()
        for i in range(0, h, 20):
            shift = np.random.randint(-10, 10)
            result[i:i+20, :] = np.roll(result[i:i+20, :], shift, axis=1)
        return result, "Horizontal Glitch Bands"
    
    elif effect_num == 42:
        depth_norm = depth_map.astype(float) / 255.0
        displacement = (depth_norm * 20).astype(int)
        result = frame.copy()
        for i in range(h):
            shift = displacement[i, w//2]
            result[i] = np.roll(frame[i], shift, axis=0)
        return result, "Depth Displacement"
    
    elif effect_num == 43:
        # RGB split glitch
        b, g, r = cv2.split(frame)
        result = frame.copy()
        result[::3, :, 0] = 255
        result[1::3, :, 1] = 255
        result[2::3, :, 2] = 255
        return result, "RGB Split Glitch"
    
    elif effect_num == 44:
        # Mirror effect
        left_half = frame[:, :w//2]
        mirrored = cv2.flip(left_half, 1)
        result = np.hstack([left_half, mirrored])
        return result, "Mirror Effect"
    
    # POSTERIZATION/COLOR REDUCTION
    elif effect_num == 45:
        levels = 4
        result = (frame // (256 // levels)) * (256 // levels)
        return result, "Posterization (4 levels)"
    
    elif effect_num == 46:
        levels = 8
        result = (frame // (256 // levels)) * (256 // levels)
        return result, "Posterization (8 levels)"
    
    elif effect_num == 47:
        # Depth-based posterization
        depth_levels = depth_map // 64 * 64
        depth_3ch = (depth_levels / 255.0).reshape(h, w, 1)
        color_reduced = (frame // 32) * 32
        result = (frame * (1 - depth_3ch*0.5) + color_reduced * depth_3ch*0.5).astype(np.uint8)
        return result, "Depth-Based Posterization"
    
    # THERMAL/FALSE COLOR
    elif effect_num == 48:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET), "Thermal (Jet)"
    
    elif effect_num == 49:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_HOT), "Thermal (Hot)"
    
    elif effect_num == 50:
        # Depth + thermal
        depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_RAINBOW)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        return cv2.addWeighted(depth_color, 0.5, thermal, 0.5, 0), "Depth + Thermal Fusion"
    
    return frame, "Unknown Effect"


def main():
    cap = cv2.VideoCapture(STREAM_URL)
    
    if not cap.isOpened():
        print(f"ERROR: Cannot open stream {STREAM_URL}")
        print("Update STREAM_URL at top of script with your camera's RTSP URL")
        return
    
    effect_num = 0
    max_effects = 50
    
    print("Controls:")
    print("  SPACE - Next effect")
    print("  's' - Save current frame")
    print("  'q' - Quit")
    print(f"\nCycling through {max_effects} effects...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Apply effect
        processed, effect_name = apply_effect(frame, effect_num)
        
        # Add label
        label = f"Effect {effect_num}: {effect_name}"
        cv2.putText(processed, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Effect Preview', processed)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space - next effect
            effect_num = (effect_num + 1) % (max_effects + 1)
            print(f"Effect {effect_num}: {effect_name}")
        
        elif key == ord('s'):  # Save frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{SAVE_DIR}/effect_{effect_num:02d}_{timestamp}.jpg"
            cv2.imwrite(filename, processed)
            print(f"Saved: {filename}")
        
        elif key == ord('q'):  # Quit
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
