#!/usr/bin/env python3
"""
VASARI RIG V2 — Refined Effects

RGB MODES:
  1 = Normal RGB
  2 = Edge ghost (↑↓ adjusts line weight)
  5 = Invert
  6 = Thermal (RGB)
  7 = VASARI (invert + breathing corruption)
  B = VASARI BREAKUP (no invert, just corruption on normal image)

3D DEPTH MODES:
  9 = Point cloud (base)
  W = Point cloud - SPARSE (bigger gaps)
  E = Point cloud - DENSE (tight)
  R = Point cloud - COLOR SHIFT (depth = hue)
  T = Point cloud - STRIPES (horizontal only)
  Y = Point cloud - RAIN (vertical streaks)
  U = Point cloud - SCATTER (randomized positions)

  ] = Depth displacement (base)
  A = Displacement - WAVE (sinusoidal warp)
  S = Displacement - SHATTER (fragmented)
  D = Displacement - RIPPLE (circular from center)
  F = Displacement - STRETCH (vertical pull)
  G = Displacement - TWIST (spiral distortion)

  - = Depth contours (base)
  Z = Contours - THICK (heavy lines, slow)
  X = Contours - FILLED (solid bands)

  + = Volumetric slices (keep as-is)
  0 = Portal (FIXED - less flashy)

  Q = Quit

ADJUSTMENTS:
  ↑↓ = Line weight (mode 2)
  ←→ = Blend original (mode 7/B)
"""

import cv2
import depthai as dai
import numpy as np
import pyvirtualcam
import random
import math

W, H, FPS = 640, 480, 30

# === GLOBAL STATE ===
frame_count = 0
line_weight = 1
blend_amount = 0.0  # 0 = full effect, 1 = full original
contour_phase = 0.0

# Separate wave state for each VASARI variant
vasari_wave_start = -999
vasari_wave_length = 200
vasari_wave_intensity = 0.7

breakup_wave_start = -999
breakup_wave_length = 200
breakup_wave_intensity = 0.7


def crop_center(img):
    """Crop to center 1/3 and resize back"""
    h, w = img.shape[:2]
    if h < 3 or w < 3:
        return img
    y1, y2 = h // 3, 2 * h // 3
    x1, x2 = w // 3, 2 * w // 3
    cropped = img[y1:y2, x1:x2]
    if cropped.size == 0:
        return img
    return cv2.resize(cropped, (w, h))


# === RGB EFFECTS ===

def effect_normal(rgb, depth):
    return rgb


def effect_edge(rgb, depth):
    global line_weight
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    # Adjust thresholds based on line weight
    low = max(20, 50 - line_weight * 10)
    high = max(50, 150 - line_weight * 20)
    edges = cv2.Canny(gray, low, high)

    # Dilate for thicker lines
    if line_weight > 1:
        kernel = np.ones((line_weight, line_weight), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

    out = np.zeros_like(rgb)
    out[:, :, 1] = edges  # green
    out[:, :, 0] = edges // 2  # slight blue
    return out


def effect_invert(rgb, depth):
    return cv2.bitwise_not(rgb)


def effect_thermal_rgb(rgb, depth):
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def effect_vasari(rgb, depth):
    """Mode 7 - inverted + breathing corruption"""
    global vasari_wave_start, vasari_wave_length, vasari_wave_intensity, blend_amount, frame_count

    out = cv2.bitwise_not(rgb)

    # Random wave trigger
    if frame_count - vasari_wave_start > vasari_wave_length + random.randint(120, 300) and random.random() > 0.997:
        vasari_wave_start = frame_count
        vasari_wave_length = random.randint(150, 400)
        vasari_wave_intensity = random.uniform(0.5, 1.0)

    wave_progress = frame_count - vasari_wave_start
    if wave_progress < vasari_wave_length:
        breath = math.sin(math.pi * wave_progress / vasari_wave_length)
        breath = breath * breath * breath
        intensity = breath * vasari_wave_intensity

        # Corruption effects
        if intensity > 0.1:
            num_bands = int(intensity * 8) + 1
            for _ in range(num_bands):
                y = random.randint(0, max(0, H - 50))
                band_h = random.randint(10, int(40 * intensity) + 10)
                band_h = min(band_h, H - y)  # Clamp to bounds
                shift = int(random.uniform(-30, 30) * intensity)
                out[y:y + band_h] = np.roll(out[y:y + band_h], shift, axis=1)

            if intensity > 0.4 and random.random() > 0.7:
                # Fixed channel swap - pick one permutation and apply it
                perm = random.choice([[2, 1, 0], [1, 0, 2], [0, 2, 1]])
                out = out[:, :, perm]

    # Blend with original if slider > 0
    if blend_amount > 0:
        out = cv2.addWeighted(out, 1 - blend_amount, rgb, blend_amount, 0)

    return out


def effect_vasari_breakup(rgb, depth):
    """Mode B - corruption WITHOUT invert (normal image breaks up)"""
    global breakup_wave_start, breakup_wave_length, breakup_wave_intensity, blend_amount, frame_count

    out = rgb.copy()  # Start with normal, not inverted

    # Random wave trigger
    if frame_count - breakup_wave_start > breakup_wave_length + random.randint(120, 300) and random.random() > 0.997:
        breakup_wave_start = frame_count
        breakup_wave_length = random.randint(150, 400)
        breakup_wave_intensity = random.uniform(0.5, 1.0)

    wave_progress = frame_count - breakup_wave_start
    if wave_progress < breakup_wave_length:
        breath = math.sin(math.pi * wave_progress / breakup_wave_length)
        breath = breath * breath * breath
        intensity = breath * breakup_wave_intensity

        if intensity > 0.1:
            num_bands = int(intensity * 8) + 1
            for _ in range(num_bands):
                y = random.randint(0, max(0, H - 50))
                band_h = random.randint(10, int(40 * intensity) + 10)
                band_h = min(band_h, H - y)  # Clamp to bounds
                shift = int(random.uniform(-30, 30) * intensity)
                out[y:y + band_h] = np.roll(out[y:y + band_h], shift, axis=1)

            # Subtle color shifts instead of full channel swap
            if intensity > 0.4 and random.random() > 0.7:
                # Slight hue rotation - use int16 to avoid overflow
                hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
                hsv_h = hsv[:, :, 0].astype(np.int16)
                hsv_h = (hsv_h + int(intensity * 30)) % 180
                hsv[:, :, 0] = hsv_h.astype(np.uint8)
                out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if blend_amount > 0:
        out = cv2.addWeighted(out, 1 - blend_amount, rgb, blend_amount, 0)

    return out


# === POINT CLOUD VARIANTS (Mode 9 family) ===

def effect_pointcloud_base(rgb, depth):
    """9 - Original point cloud"""
    if depth is None:
        return rgb
    d = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    out = np.zeros_like(rgb)
    for y in range(0, H - 1, 4):
        for x in range(0, W - 1, 4):
            if d[y, x] > 50:
                r = max(1, int((255 - d[y, x]) / 50))
                cv2.circle(out, (x, y), r, rgb[y, x].tolist(), -1)
    return out


def effect_pointcloud_sparse(rgb, depth):
    """W - Bigger gaps between points"""
    if depth is None:
        return rgb
    d = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    out = np.zeros_like(rgb)
    for y in range(0, H - 1, 8):  # Bigger step
        for x in range(0, W - 1, 8):
            if d[y, x] > 50:
                r = max(2, int((255 - d[y, x]) / 40))
                cv2.circle(out, (x, y), r, rgb[y, x].tolist(), -1)
    return out


def effect_pointcloud_dense(rgb, depth):
    """E - Tight packed points"""
    if depth is None:
        return rgb
    d = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    out = np.zeros_like(rgb)
    for y in range(0, H - 1, 2):  # Smaller step
        for x in range(0, W - 1, 2):
            if d[y, x] > 40:
                r = max(1, int((255 - d[y, x]) / 80))
                cv2.circle(out, (x, y), r, rgb[y, x].tolist(), -1)
    return out


def effect_pointcloud_colorshift(rgb, depth):
    """R - Depth mapped to hue"""
    if depth is None:
        return rgb
    d = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    out = np.zeros_like(rgb)
    for y in range(0, H - 1, 4):
        for x in range(0, W - 1, 4):
            if d[y, x] > 50:
                r = max(1, int((255 - d[y, x]) / 50))
                # Map depth to hue
                hue = int(d[y, x] * 0.7)  # 0-180 range
                color = cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0].tolist()
                cv2.circle(out, (x, y), r, color, -1)
    return out


def effect_pointcloud_stripes(rgb, depth):
    """T - Horizontal lines only"""
    if depth is None:
        return rgb
    d = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    out = np.zeros_like(rgb)
    for y in range(0, H - 2, 6):
        for x in range(0, W):
            if d[y, x] > 50:
                out[y:min(y + 2, H), x] = rgb[y, x]
    return out


def effect_pointcloud_rain(rgb, depth):
    """Y - Vertical streaks (sparse sampling for distinct rain drops)"""
    if depth is None:
        return rgb
    d = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    out = np.zeros_like(rgb)
    for y in range(0, H - 1, 8):  # Sparse vertical sampling for distinct streaks
        for x in range(0, W - 2, 6):
            if d[y, x] > 50:
                streak_len = max(1, int((255 - d[y, x]) / 20))
                y2 = min(H, y + streak_len)
                x2 = min(W, x + 2)
                out[y:y2, x:x2] = rgb[y, x]
    return out


def effect_pointcloud_scatter(rgb, depth):
    """U - Randomized positions (deterministic per-pixel for stability)"""
    if depth is None:
        return rgb
    d = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    out = np.zeros_like(rgb)
    for y in range(0, H - 1, 5):
        for x in range(0, W - 1, 5):
            if d[y, x] > 50:
                # Deterministic offset based on position for stable scatter
                np.random.seed((y * W + x) % 2147483647)
                ox = x + np.random.randint(-3, 4)
                oy = y + np.random.randint(-3, 4)
                ox = max(0, min(W - 1, ox))
                oy = max(0, min(H - 1, oy))
                r = max(1, int((255 - d[y, x]) / 50))
                cv2.circle(out, (ox, oy), r, rgb[y, x].tolist(), -1)
    return out


# === DISPLACEMENT VARIANTS (Mode ] family) ===

def effect_displace_base(rgb, depth):
    """] - Base displacement using backward mapping"""
    if depth is None:
        return rgb
    d = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

    # Create mapping using backward mapping (sample from source)
    map_x = np.zeros((H, W), dtype=np.float32)
    map_y = np.zeros((H, W), dtype=np.float32)

    for y in range(H):
        for x in range(W):
            offset = int((d[y, x] - 0.5) * 30)
            map_x[y, x] = max(0, min(W - 1, x - offset))  # Backward mapping
            map_y[y, x] = y

    return cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR)


def effect_displace_wave(rgb, depth):
    """A - Sinusoidal warp"""
    global frame_count
    if depth is None:
        return rgb
    d = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

    # Create mapping
    map_x = np.zeros((H, W), dtype=np.float32)
    map_y = np.zeros((H, W), dtype=np.float32)

    for y in range(H):
        for x in range(W):
            wave = math.sin(y / 20 + frame_count / 10) * d[y, x] * 20
            map_x[y, x] = x + wave
            map_y[y, x] = y

    return cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR)


def effect_displace_shatter(rgb, depth):
    """S - Fragmented blocks with randomized offsets"""
    if depth is None:
        return rgb
    d = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    out = rgb.copy()

    block = 20
    for by in range(0, H, block):
        for bx in range(0, W, block):
            # Calculate actual block size (handle edges)
            bh = min(block, H - by)
            bw = min(block, W - bx)

            # Get average depth of block
            block_d = d[by:by + bh, bx:bx + bw]
            if block_d.size > 0:
                avg_d = np.mean(block_d)
                # Add randomization based on block position for true shatter effect
                np.random.seed(int(by * W + bx) % 2147483647)
                rand_offset = np.random.randint(-15, 16)
                offset_x = int((avg_d / 255 - 0.5) * 40) + rand_offset
                offset_y = int((avg_d / 255 - 0.5) * 20) + np.random.randint(-10, 11)

                src_x = max(0, min(W - bw, bx + offset_x))
                src_y = max(0, min(H - bh, by + offset_y))
                out[by:by + bh, bx:bx + bw] = rgb[src_y:src_y + bh, src_x:src_x + bw]

    return out


def effect_displace_ripple(rgb, depth):
    """D - Circular ripple from center"""
    global frame_count
    if depth is None:
        return rgb
    d = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

    cx, cy = W // 2, H // 2
    map_x = np.zeros((H, W), dtype=np.float32)
    map_y = np.zeros((H, W), dtype=np.float32)

    for y in range(H):
        for x in range(W):
            dx, dy = x - cx, y - cy
            dist = math.sqrt(dx * dx + dy * dy)
            angle = math.atan2(dy, dx)
            ripple = math.sin(dist / 15 - frame_count / 5) * d[y, x] * 10

            map_x[y, x] = x + math.cos(angle) * ripple
            map_y[y, x] = y + math.sin(angle) * ripple

    return cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR)


def effect_displace_stretch(rgb, depth):
    """F - Vertical pull based on depth"""
    if depth is None:
        return rgb
    d = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

    map_x = np.zeros((H, W), dtype=np.float32)
    map_y = np.zeros((H, W), dtype=np.float32)

    for y in range(H):
        for x in range(W):
            stretch = (d[y, x] - 0.5) * 40
            map_x[y, x] = x
            map_y[y, x] = y + stretch

    return cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR)


def effect_displace_twist(rgb, depth):
    """G - Spiral distortion"""
    global frame_count
    if depth is None:
        return rgb
    d = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

    cx, cy = W // 2, H // 2
    max_dist = math.sqrt(cx * cx + cy * cy)  # Use actual max distance
    map_x = np.zeros((H, W), dtype=np.float32)
    map_y = np.zeros((H, W), dtype=np.float32)

    for y in range(H):
        for x in range(W):
            dx, dy = x - cx, y - cy
            dist = math.sqrt(dx * dx + dy * dy)
            angle = math.atan2(dy, dx)

            # Twist amount based on depth, clamped to prevent reversal
            falloff = max(0, 1 - dist / max_dist)
            twist = d[y, x] * 0.5 * falloff
            new_angle = angle + twist

            map_x[y, x] = cx + dist * math.cos(new_angle)
            map_y[y, x] = cy + dist * math.sin(new_angle)

    return cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR)


# === CONTOUR VARIANTS (Mode - family) ===

def effect_contours_base(rgb, depth):
    """- Base contours"""
    if depth is None:
        return rgb
    d = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.Canny(d, 30, 100)
    out = np.zeros_like(rgb)
    out[:, :, 1] = edges
    out[:, :, 2] = edges // 2
    return out


def effect_contours_thick(rgb, depth):
    """Z - Heavy lines, slower animation"""
    global contour_phase
    contour_phase = (contour_phase + 0.05) % (2 * math.pi)  # Wrap to prevent overflow

    if depth is None:
        return rgb
    d = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Multiple edge thresholds for thick lines
    edges1 = cv2.Canny(d, 20, 80)
    edges2 = cv2.Canny(d, 40, 120)
    edges3 = cv2.Canny(d, 60, 160)

    edges = cv2.bitwise_or(edges1, edges2)
    edges = cv2.bitwise_or(edges, edges3)

    # Dilate for thickness
    kernel = np.ones((4, 4), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    out = np.zeros_like(rgb)
    # Animated color - clamp hue to valid range
    hue = int((math.sin(contour_phase) + 1) * 45) + 30  # Range: 30-120
    hue = max(0, min(90, hue))  # Ensure non-negative for subtraction

    out[:, :, 0] = (edges.astype(np.int32) * hue // 255).astype(np.uint8)
    out[:, :, 1] = edges
    out[:, :, 2] = (edges.astype(np.int32) * max(0, 90 - hue) // 90).astype(np.uint8)

    return out


def effect_contours_filled(rgb, depth):
    """X - Solid bands (filled shades)"""
    if depth is None:
        return rgb
    d = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Quantize depth into bands
    num_bands = 8
    quantized = (d // (256 // num_bands)) * (256 // num_bands)

    # Apply colormap to quantized depth
    colored = cv2.applyColorMap(quantized, cv2.COLORMAP_VIRIDIS)

    # Add contour lines on top
    edges = cv2.Canny(d, 30, 100)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # White edges on colored bands
    colored[edges > 0] = [255, 255, 255]

    return colored


# === OTHER 3D EFFECTS ===

def effect_portal_fixed(rgb, depth):
    """0 - Portal (fixed: dark blue, not flashy)"""
    if depth is None:
        return rgb
    d = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Create mask for mid-depth (the "portal")
    lower = 80
    upper = 160
    mask = cv2.inRange(d, lower, upper)

    out = rgb.copy()
    # Portal area becomes dark blue (BGR: high B, low G, low R)
    portal_color = np.zeros_like(rgb)
    portal_color[:, :] = [150, 40, 20]  # Dark blue in BGR

    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    out = (rgb.astype(np.float32) * (1 - mask_3ch) + portal_color.astype(np.float32) * mask_3ch).astype(np.uint8)

    # Subtle edge glow
    edges = cv2.Canny(mask, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    out[edges > 0] = [255, 150, 50]  # Bright blue edge glow

    return out


def effect_slices(rgb, depth):
    """+ - Volumetric slices"""
    if depth is None:
        return rgb
    d = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    out = np.zeros_like(rgb)
    for y in range(0, H - 6, 8):
        offset = int((d[y, W // 2] / 255) * 20 - 10)
        y2 = max(0, min(H - 6, y + offset))
        # Ensure source and destination slices are same size
        slice_h = min(6, H - y, H - y2)
        out[y:y + slice_h] = rgb[y2:y2 + slice_h]

    return out


# === EFFECTS MAP (includes both lowercase and uppercase) ===
effects = {
    # RGB modes
    ord('1'): ("Normal", effect_normal),
    ord('2'): ("Edge (↑↓=weight)", effect_edge),
    ord('5'): ("Invert", effect_invert),
    ord('6'): ("Thermal", effect_thermal_rgb),
    ord('7'): ("VASARI", effect_vasari),
    ord('b'): ("Breakup (no invert)", effect_vasari_breakup),
    ord('B'): ("Breakup (no invert)", effect_vasari_breakup),

    # Point cloud family (9)
    ord('9'): ("PointCloud", effect_pointcloud_base),
    ord('w'): ("PC-Sparse", effect_pointcloud_sparse),
    ord('W'): ("PC-Sparse", effect_pointcloud_sparse),
    ord('e'): ("PC-Dense", effect_pointcloud_dense),
    ord('E'): ("PC-Dense", effect_pointcloud_dense),
    ord('r'): ("PC-ColorShift", effect_pointcloud_colorshift),
    ord('R'): ("PC-ColorShift", effect_pointcloud_colorshift),
    ord('t'): ("PC-Stripes", effect_pointcloud_stripes),
    ord('T'): ("PC-Stripes", effect_pointcloud_stripes),
    ord('y'): ("PC-Rain", effect_pointcloud_rain),
    ord('Y'): ("PC-Rain", effect_pointcloud_rain),
    ord('u'): ("PC-Scatter", effect_pointcloud_scatter),
    ord('U'): ("PC-Scatter", effect_pointcloud_scatter),

    # Displacement family (])
    ord(']'): ("Displace", effect_displace_base),
    ord('a'): ("Disp-Wave", effect_displace_wave),
    ord('A'): ("Disp-Wave", effect_displace_wave),
    ord('s'): ("Disp-Shatter", effect_displace_shatter),
    ord('S'): ("Disp-Shatter", effect_displace_shatter),
    ord('d'): ("Disp-Ripple", effect_displace_ripple),
    ord('D'): ("Disp-Ripple", effect_displace_ripple),
    ord('f'): ("Disp-Stretch", effect_displace_stretch),
    ord('F'): ("Disp-Stretch", effect_displace_stretch),
    ord('g'): ("Disp-Twist", effect_displace_twist),
    ord('G'): ("Disp-Twist", effect_displace_twist),

    # Contour family (-)
    ord('-'): ("Contours", effect_contours_base),
    ord('z'): ("Cont-Thick", effect_contours_thick),
    ord('Z'): ("Cont-Thick", effect_contours_thick),
    ord('x'): ("Cont-Filled", effect_contours_filled),
    ord('X'): ("Cont-Filled", effect_contours_filled),

    # Other 3D
    ord('0'): ("Portal (fixed)", effect_portal_fixed),
    ord('+'): ("Slices", effect_slices),
    ord('='): ("Slices", effect_slices),  # Also = key (no shift needed)
}


def main():
    global line_weight, blend_amount, frame_count

    print(__doc__)

    pipeline = dai.Pipeline()

    # RGB Camera - standard DepthAI API
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setPreviewSize(W, H)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    # Mono Cameras for Stereo
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

    # Stereo Depth
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(W, H)

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    mode = ord('7')  # Start in VASARI mode
    print("Starting in VASARI mode (7)")

    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue(name="rgb", maxSize=2, blocking=False)
        q_depth = device.getOutputQueue(name="depth", maxSize=2, blocking=False)

        with pyvirtualcam.Camera(width=W, height=H, fps=FPS) as vcam:
            print(f"Vcam: {vcam.device}")

            while True:
                frame_count += 1

                rgb_msg = q_rgb.tryGet()
                if rgb_msg is None:
                    continue
                rgb = rgb_msg.getCvFrame()
                rgb = crop_center(rgb)

                depth = None
                depth_msg = q_depth.tryGet()
                if depth_msg is not None:
                    depth = depth_msg.getFrame()
                    depth = crop_center(depth)
                    if depth.shape[:2] != (H, W):
                        depth = cv2.resize(depth, (W, H))

                name, fx = effects.get(mode, ("Normal", effect_normal))
                out = fx(rgb, depth)

                # HUD
                info = f"{name}"
                if mode == ord('2'):
                    info += f" | Weight: {line_weight}"
                if mode in [ord('7'), ord('b'), ord('B')]:
                    info += f" | Blend: {int(blend_amount * 100)}%"

                cv2.putText(out, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                vcam.send(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
                cv2.imshow("Vasari V2", out)

                k = cv2.waitKey(1) & 0xFF
                if k == ord('q') or k == ord('Q'):
                    break
                elif k in effects:
                    mode = k
                    print(f"Mode: {effects[k][0]}")
                # Cross-platform arrow key handling
                # Use bracket keys as alternative: [ ] for weight, , . for blend
                elif k == 0 or k == 82 or k == ord('['):  # Up arrow (macOS=0, Linux=82) or [
                    if mode == ord('2'):
                        line_weight = min(10, line_weight + 1)
                        print(f"Line weight: {line_weight}")
                elif k == 1 or k == 84 or k == ord(']'):  # Down arrow (macOS=1, Linux=84) or ]
                    if mode == ord('2'):
                        line_weight = max(1, line_weight - 1)
                        print(f"Line weight: {line_weight}")
                elif k == 3 or k == 83 or k == ord('.'):  # Right arrow (macOS=3, Linux=83) or .
                    if mode in [ord('7'), ord('b'), ord('B')]:
                        blend_amount = min(1.0, blend_amount + 0.1)
                        print(f"Blend: {int(blend_amount * 100)}%")
                elif k == 2 or k == 81 or k == ord(','):  # Left arrow (macOS=2, Linux=81) or ,
                    if mode in [ord('7'), ord('b'), ord('B')]:
                        blend_amount = max(0.0, blend_amount - 0.1)
                        print(f"Blend: {int(blend_amount * 100)}%")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
