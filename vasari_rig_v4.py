#!/usr/bin/env python3
"""
VASARI RIG V4 — New Key Layout

2D MODES:
  1 = Normal
  2 = Edge ghost
  I = Invert
  6 = Thermal
  V = VASARI (invert + breathing corruption)
  B = VASARI BREAKUP (no invert, just corruption)

UNCANNY 3D (subtle/strange - uses depth):
  3 = DepthFocus - mid-range sharp, near/far blurry (broken eyes)
  4 = Atmosphere - far things desaturated and blue
  5 = DepthInvert - background inverted, face normal
  8 = DepthLag - far things ghost/lag behind (CREEPY)
  C = DepthGrain - film grain increases with distance
  O = DepthGlow - subtle warm glow on depth edges

VASARI + 3D COMBOS (home row):
  D = VASARI + Ripple
  F = VASARI + Stretch
  G = VASARI + Twist
  H = VASARI + DepthPixel
  J = VASARI + Parallax
  K = VASARI + SelectiveReal
  L = VASARI + DepthShadow

OBVIOUS 3D:
  9/W/E/R/T/Y/U = Point cloud variants
  A/S = Displacement (Wave, Shatter)
  7/Z/X = Contour variants
  0 = Portal

CONTROLS:
  - = Zoom out, = Zoom in (no shift)
  SPACE = Toggle puppet mode (VASARI modes)
  Close window or Ctrl+C to quit

DEPTH INVERT TUNING (mode 5):
  < > = Depth threshold (where cut happens)
  , . = Depth softness (edge sharpness)

VASARI TUNING (V/B and combos):
  [ ] = Speed down/up
  { } = Intensity down/up
  _ + = Face pop down/up (shift - =)
  N M = Blend with original down/up
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
zoom_level = 1.0  # 1.0 = no zoom, 2.0 = 2x zoom, 3.0 = 3x zoom
lag_intensity = 0.95  # For depth lag effect (0.5 = mild, 0.99 = extreme ghosting)
_lag_buffer = None  # For depth lag effect
vasari_force_corrupt = False  # SPACE triggers corruption
puppet_mode = False  # Hold SPACE for continuous corruption

# TUNING CONTROLS (adjustable with keys)
corrupt_intensity = 1.0  # 0.3 = mild, 1.0 = normal, 2.0 = extreme ({ } to adjust)
corrupt_speed = 1.0      # 0.5 = slow waves, 1.0 = normal, 2.0 = fast ([ ] to adjust)
face_pop_amount = 0.3    # How often face pops through (0 = never, 1 = always) (; ' to adjust)

# DEPTH INVERT TUNING (mode 5)
depth_threshold = 0.35   # Where the invert cut happens (< > to adjust)
depth_softness = 8.0     # Edge sharpness, higher = sharper (, . to adjust)

# Separate wave state for each VASARI variant
vasari_wave_start = -999
vasari_wave_length = 200
vasari_wave_intensity = 0.7

breakup_wave_start = -999
breakup_wave_length = 200
breakup_wave_intensity = 0.7


def crop_center(img, zoom=1.0):
    """Crop to center and resize back. zoom=1.0 means no crop, zoom=3.0 means center 1/3"""
    if zoom <= 1.0:
        return img
    h, w = img.shape[:2]
    if h < 3 or w < 3:
        return img
    # Calculate crop region based on zoom
    crop_h, crop_w = int(h / zoom), int(w / zoom)
    y1 = (h - crop_h) // 2
    x1 = (w - crop_w) // 2
    cropped = img[y1:y1 + crop_h, x1:x1 + crop_w]
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
    """Mode 7 - inverted + AGGRESSIVE corruption + PUPPET MODE + FACE POP"""
    global vasari_wave_start, vasari_wave_length, vasari_wave_intensity, blend_amount, frame_count, vasari_force_corrupt, puppet_mode
    global corrupt_intensity, corrupt_speed, face_pop_amount

    out = cv2.bitwise_not(rgb)
    h, w = out.shape[:2]

    # Wave timing adjusted by speed
    wave_min = max(10, int(15 / corrupt_speed))
    wave_max = max(20, int(40 / corrupt_speed))
    cooldown_min = max(10, int(20 / corrupt_speed))
    cooldown_max = max(30, int(60 / corrupt_speed))

    # PUPPET MODE: continuous corruption while SPACE held
    if puppet_mode:
        if frame_count - vasari_wave_start >= vasari_wave_length:
            vasari_wave_start = frame_count
            vasari_wave_length = random.randint(wave_min, wave_max)
            vasari_wave_intensity = 1.0
    elif 'vasari_force_corrupt' in globals() and vasari_force_corrupt:
        globals()['vasari_force_corrupt'] = False
        vasari_wave_start = frame_count
        vasari_wave_length = random.randint(wave_max, wave_max * 3)
        vasari_wave_intensity = 1.0
    elif frame_count - vasari_wave_start > vasari_wave_length + random.randint(cooldown_min, cooldown_max) and random.random() > 0.95:
        vasari_wave_start = frame_count
        vasari_wave_length = random.randint(wave_max, wave_max * 3)
        vasari_wave_intensity = random.uniform(0.7, 1.0)

    wave_progress = frame_count - vasari_wave_start
    in_wave = wave_progress < vasari_wave_length or puppet_mode
    if in_wave:
        # Intensity with tuning
        base_intensity = max(0, 1.0 - (wave_progress / max(1, vasari_wave_length))) * vasari_wave_intensity
        intensity = base_intensity * corrupt_intensity

        # === FACE POP: occasionally let real face through ===
        if face_pop_amount > 0 and random.random() < face_pop_amount * 0.3:
            # Flash of real face - blend original RGB into corrupted output
            pop_strength = random.uniform(0.3, 0.8)
            out = cv2.addWeighted(out, 1 - pop_strength, rgb, pop_strength, 0)

        # === HORIZONTAL BAND SHIFTS ===
        num_bands = random.randint(int(8 * corrupt_intensity), int(25 * corrupt_intensity) + 1)
        for _ in range(max(1, num_bands)):
            y = random.randint(0, max(0, h - 2))
            band_h = random.randint(2, max(3, int(50 * intensity) + 2))
            band_h = min(band_h, h - y)
            if band_h > 0:
                max_shift = max(1, int(80 * intensity))
                shift = random.randint(-max_shift, max_shift)
                out[y:y + band_h, :] = np.roll(out[y:y + band_h, :], shift, axis=1)

        # === BLOCK DISPLACEMENT ===
        if intensity > 0.3 / corrupt_intensity and h > 60 and w > 100:
            for _ in range(random.randint(2, int(6 * corrupt_intensity) + 1)):
                bh = random.randint(20, min(60, h - 1))
                bw = random.randint(30, min(100, w - 1))
                by = random.randint(0, max(0, h - bh - 1))
                bx = random.randint(0, max(0, w - bw - 1))
                if by + bh <= h and bx + bw <= w:
                    block = out[by:by+bh, bx:bx+bw].copy()
                    ny = max(0, min(h - bh, by + random.randint(-30, 30)))
                    nx = max(0, min(w - bw, bx + random.randint(-50, 50)))
                    out[ny:ny+bh, nx:nx+bw] = block

        # === CHANNEL CORRUPTION ===
        if intensity > 0.2 / corrupt_intensity and random.random() > 0.4:
            choice = random.randint(0, 3)
            if choice == 0:
                out = np.ascontiguousarray(out[:, :, [2, 1, 0]])
            elif choice == 1:
                out = np.ascontiguousarray(out[:, :, [1, 0, 2]])
            elif choice == 2:
                out = np.ascontiguousarray(out[:, :, [0, 2, 1]])
            else:
                ch = random.randint(0, 2)
                out[:, :, ch] = np.roll(out[:, :, ch], random.randint(-20, 20), axis=1)

        # === PIXELATION BURSTS (this is what shows your face through!) ===
        if intensity > 0.4 / corrupt_intensity and random.random() > 0.5:
            block = random.randint(max(2, int(4 / corrupt_intensity)), int(16 * corrupt_intensity))
            if w // block > 0 and h // block > 0:
                small = cv2.resize(out, (w // block, h // block), interpolation=cv2.INTER_LINEAR)
                out = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        # === VHS TRACKING ===
        if intensity > 0.5 / corrupt_intensity and random.random() > 0.6 and h > 10:
            for _ in range(random.randint(3, 10)):
                src_y = random.randint(0, max(0, h - 6))
                dst_y = random.randint(0, max(0, h - 6))
                num_lines = random.randint(1, 5)
                num_lines = min(num_lines, h - src_y, h - dst_y)
                if num_lines > 0:
                    out[dst_y:dst_y + num_lines, :] = out[src_y:src_y + num_lines, :]

        # === FACE POP: bigger chunks of real face ===
        if face_pop_amount > 0.2 and random.random() < face_pop_amount * 0.15 and h > 100 and w > 100:
            # Show a rectangle of real face
            fy = random.randint(0, h - 80)
            fx = random.randint(0, w - 80)
            fh = random.randint(40, 120)
            fw = random.randint(40, 120)
            fh, fw = min(fh, h - fy), min(fw, w - fx)
            out[fy:fy+fh, fx:fx+fw] = rgb[fy:fy+fh, fx:fx+fw]

    # Blend with original if slider > 0
    if blend_amount > 0:
        out = cv2.addWeighted(out, 1 - blend_amount, rgb, blend_amount, 0)

    return out


def effect_vasari_breakup(rgb, depth):
    """Mode B - AGGRESSIVE corruption on normal image (no invert) + PUPPET MODE + TUNING"""
    global breakup_wave_start, breakup_wave_length, breakup_wave_intensity, blend_amount, frame_count, vasari_force_corrupt, puppet_mode
    global corrupt_intensity, corrupt_speed, face_pop_amount

    out = rgb.copy()
    h, w = out.shape[:2]

    # Wave timing adjusted by speed
    wave_min = max(10, int(15 / corrupt_speed))
    wave_max = max(20, int(40 / corrupt_speed))
    cooldown_min = max(10, int(20 / corrupt_speed))
    cooldown_max = max(30, int(60 / corrupt_speed))

    # PUPPET MODE: continuous corruption
    if puppet_mode:
        if frame_count - breakup_wave_start >= breakup_wave_length:
            breakup_wave_start = frame_count
            breakup_wave_length = random.randint(wave_min, wave_max)
            breakup_wave_intensity = 1.0
    elif 'vasari_force_corrupt' in globals() and vasari_force_corrupt:
        globals()['vasari_force_corrupt'] = False
        breakup_wave_start = frame_count
        breakup_wave_length = random.randint(wave_max, wave_max * 3)
        breakup_wave_intensity = 1.0
    elif frame_count - breakup_wave_start > breakup_wave_length + random.randint(cooldown_min, cooldown_max) and random.random() > 0.95:
        breakup_wave_start = frame_count
        breakup_wave_length = random.randint(wave_max, wave_max * 3)
        breakup_wave_intensity = random.uniform(0.7, 1.0)

    wave_progress = frame_count - breakup_wave_start
    in_wave = wave_progress < breakup_wave_length or puppet_mode
    if in_wave:
        base_intensity = max(0, 1.0 - (wave_progress / max(1, breakup_wave_length))) * breakup_wave_intensity
        intensity = base_intensity * corrupt_intensity

        # === FACE POP: occasionally let real face through clearly ===
        if face_pop_amount > 0 and random.random() < face_pop_amount * 0.25:
            # Breakup mode already shows face, so we do LESS corruption instead
            intensity = intensity * 0.3  # Reduce corruption for this frame

        # === HORIZONTAL BAND SHIFTS ===
        num_bands = random.randint(int(8 * corrupt_intensity), int(25 * corrupt_intensity) + 1)
        for _ in range(max(1, num_bands)):
            y = random.randint(0, max(0, h - 2))
            band_h = random.randint(2, max(3, int(50 * intensity) + 2))
            band_h = min(band_h, h - y)
            if band_h > 0:
                max_shift = max(1, int(80 * intensity))
                shift = random.randint(-max_shift, max_shift)
                out[y:y + band_h, :] = np.roll(out[y:y + band_h, :], shift, axis=1)

        # === BLOCK DISPLACEMENT ===
        if intensity > 0.3 / corrupt_intensity and h > 60 and w > 100:
            for _ in range(random.randint(2, 6)):
                bh = random.randint(20, min(60, h - 1))
                bw = random.randint(30, min(100, w - 1))
                by = random.randint(0, max(0, h - bh - 1))
                bx = random.randint(0, max(0, w - bw - 1))
                if by + bh <= h and bx + bw <= w:
                    block = out[by:by+bh, bx:bx+bw].copy()
                    ny = max(0, min(h - bh, by + random.randint(-30, 30)))
                    nx = max(0, min(w - bw, bx + random.randint(-50, 50)))
                    out[ny:ny+bh, nx:nx+bw] = block

        # === COLOR GLITCHING (hue shift for breakup mode) ===
        if intensity > 0.2 and random.random() > 0.4:
            hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
            hsv[:, :, 0] = ((hsv[:, :, 0].astype(np.int16) + random.randint(20, 80)) % 180).astype(np.uint8)
            out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # === PIXELATION BURSTS ===
        if intensity > 0.4 and random.random() > 0.5:
            block = random.randint(4, 16)
            if w // block > 0 and h // block > 0:
                small = cv2.resize(out, (w // block, h // block), interpolation=cv2.INTER_LINEAR)
                out = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        # === HORIZONTAL LINE DUPLICATION ===
        if intensity > 0.5 and random.random() > 0.6 and h > 10:
            for _ in range(random.randint(3, 10)):
                src_y = random.randint(0, max(0, h - 6))
                dst_y = random.randint(0, max(0, h - 6))
                num_lines = random.randint(1, 5)
                num_lines = min(num_lines, h - src_y, h - dst_y)
                if num_lines > 0:
                    out[dst_y:dst_y + num_lines, :] = out[src_y:src_y + num_lines, :]

    if blend_amount > 0:
        out = cv2.addWeighted(out, 1 - blend_amount, rgb, blend_amount, 0)

    return out


# === POINT CLOUD VARIANTS (Mode 9 family) ===

def effect_pointcloud_base(rgb, depth):
    """9 - Original point cloud"""
    if depth is None:
        return rgb
    h, w = rgb.shape[:2]
    d = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h))
    out = np.zeros_like(rgb)
    for y in range(0, h - 1, 4):
        for x in range(0, w - 1, 4):
            if d[y, x] > 50:
                r = max(1, int((255 - d[y, x]) / 50))
                cv2.circle(out, (x, y), r, rgb[y, x].tolist(), -1)
    return out


def effect_pointcloud_sparse(rgb, depth):
    """W - Bigger gaps between points"""
    if depth is None:
        return rgb
    h, w = rgb.shape[:2]
    d = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h))
    out = np.zeros_like(rgb)
    for y in range(0, h - 1, 8):  # Bigger step
        for x in range(0, w - 1, 8):
            if d[y, x] > 50:
                r = max(2, int((255 - d[y, x]) / 40))
                cv2.circle(out, (x, y), r, rgb[y, x].tolist(), -1)
    return out


def effect_pointcloud_dense(rgb, depth):
    """E - Tight packed points"""
    if depth is None:
        return rgb
    h, w = rgb.shape[:2]
    d = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h))
    out = np.zeros_like(rgb)
    for y in range(0, h - 1, 2):  # Smaller step
        for x in range(0, w - 1, 2):
            if d[y, x] > 40:
                r = max(1, int((255 - d[y, x]) / 80))
                cv2.circle(out, (x, y), r, rgb[y, x].tolist(), -1)
    return out


def effect_pointcloud_colorshift(rgb, depth):
    """R - Depth mapped to hue"""
    if depth is None:
        return rgb
    h, w = rgb.shape[:2]
    d = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h))
    out = np.zeros_like(rgb)
    for y in range(0, h - 1, 4):
        for x in range(0, w - 1, 4):
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
    h, w = rgb.shape[:2]
    d = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h))
    out = np.zeros_like(rgb)
    for y in range(0, h - 2, 6):
        for x in range(0, w):
            if d[y, x] > 50:
                out[y:min(y + 2, h), x] = rgb[y, x]
    return out


def effect_pointcloud_rain(rgb, depth):
    """Y - Vertical streaks (sparse sampling for distinct rain drops)"""
    if depth is None:
        return rgb
    h, w = rgb.shape[:2]
    d = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h))
    out = np.zeros_like(rgb)
    for y in range(0, h - 1, 8):  # Sparse vertical sampling for distinct streaks
        for x in range(0, w - 2, 6):
            if d[y, x] > 50:
                streak_len = max(1, int((255 - d[y, x]) / 20))
                y2 = min(h, y + streak_len)
                x2 = min(w, x + 2)
                out[y:y2, x:x2] = rgb[y, x]
    return out


def effect_pointcloud_scatter(rgb, depth):
    """U - Randomized positions (deterministic per-pixel for stability)"""
    if depth is None:
        return rgb
    h, w = rgb.shape[:2]
    d = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h))
    out = np.zeros_like(rgb)
    # Use local RandomState to avoid corrupting global random state
    for y in range(0, h - 1, 5):
        for x in range(0, w - 1, 5):
            if d[y, x] > 50:
                # Deterministic offset based on position for stable scatter
                rng = np.random.RandomState((y * w + x) % 2147483647)
                ox = x + rng.randint(-3, 4)
                oy = y + rng.randint(-3, 4)
                ox = max(0, min(w - 1, ox))
                oy = max(0, min(h - 1, oy))
                r = max(1, int((255 - d[y, x]) / 50))
                cv2.circle(out, (ox, oy), r, rgb[y, x].tolist(), -1)
    return out


# === DISPLACEMENT VARIANTS (Mode ] family) ===

def effect_displace_base(rgb, depth):
    """] - Base displacement using backward mapping"""
    if depth is None:
        return rgb
    h, w = rgb.shape[:2]
    d = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h))

    # Create mapping using backward mapping (sample from source)
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            offset = int((d[y, x] - 0.5) * 30)
            map_x[y, x] = max(0, min(w - 1, x - offset))  # Backward mapping
            map_y[y, x] = y

    return cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def effect_displace_wave(rgb, depth):
    """A - Sinusoidal warp"""
    global frame_count
    if depth is None:
        return rgb
    h, w = rgb.shape[:2]
    d = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h))

    # Create mapping
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            wave = math.sin(y / 20 + frame_count / 10) * d[y, x] * 20
            map_x[y, x] = x + wave
            map_y[y, x] = y

    return cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def effect_displace_shatter(rgb, depth):
    """S - Fragmented blocks with randomized offsets"""
    if depth is None:
        return rgb
    h, w = rgb.shape[:2]
    d = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h))
    out = rgb.copy()

    block = 20
    for by in range(0, h, block):
        for bx in range(0, w, block):
            # Calculate actual block size (handle edges)
            bh = min(block, h - by)
            bw = min(block, w - bx)

            # Get average depth of block
            block_d = d[by:by + bh, bx:bx + bw]
            if block_d.size > 0:
                avg_d = np.mean(block_d)
                # Use local RandomState to avoid corrupting global random state
                rng = np.random.RandomState(int(by * w + bx) % 2147483647)
                rand_offset = rng.randint(-15, 16)
                offset_x = int((avg_d / 255 - 0.5) * 40) + rand_offset
                offset_y = int((avg_d / 255 - 0.5) * 20) + rng.randint(-10, 11)

                src_x = max(0, min(w - bw, bx + offset_x))
                src_y = max(0, min(h - bh, by + offset_y))
                out[by:by + bh, bx:bx + bw] = rgb[src_y:src_y + bh, src_x:src_x + bw]

    return out


def effect_displace_ripple(rgb, depth):
    """D - Circular ripple from center"""
    global frame_count
    if depth is None:
        return rgb
    h, w = rgb.shape[:2]
    d = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h))

    cx, cy = w // 2, h // 2
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            dx, dy = x - cx, y - cy
            dist = math.sqrt(dx * dx + dy * dy)
            angle = math.atan2(dy, dx)
            ripple = math.sin(dist / 15 - frame_count / 5) * d[y, x] * 10

            map_x[y, x] = x + math.cos(angle) * ripple
            map_y[y, x] = y + math.sin(angle) * ripple

    return cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def effect_displace_stretch(rgb, depth):
    """F - Vertical pull based on depth"""
    if depth is None:
        return rgb
    h, w = rgb.shape[:2]
    d = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h))

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            stretch = (d[y, x] - 0.5) * 40
            map_x[y, x] = x
            map_y[y, x] = y + stretch

    return cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def effect_displace_twist(rgb, depth):
    """G - Spiral distortion"""
    global frame_count
    if depth is None:
        return rgb
    h, w = rgb.shape[:2]
    d = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h))

    cx, cy = w // 2, h // 2
    max_dist = math.sqrt(cx * cx + cy * cy)  # Use actual max distance
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            dx, dy = x - cx, y - cy
            dist = math.sqrt(dx * dx + dy * dy)
            angle = math.atan2(dy, dx)

            # Twist amount based on depth, clamped to prevent reversal
            falloff = max(0, 1 - dist / max_dist)
            twist = d[y, x] * 0.5 * falloff
            new_angle = angle + twist

            map_x[y, x] = cx + dist * math.cos(new_angle)
            map_y[y, x] = cy + dist * math.sin(new_angle)

    return cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


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


# === UNCANNY/SUBTLE 3D EFFECTS ===

def effect_depth_focus(rgb, depth):
    """3 - Depth-based focus: mid-range sharp, near/far blurry (like broken eyes)"""
    if depth is None:
        return rgb
    h, w = rgb.shape[:2]
    d = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h))

    # Blur the image
    blurred = cv2.GaussianBlur(rgb, (21, 21), 0)

    # Focus on mid-range (0.3-0.6 depth)
    # Calculate "focus" amount - 1.0 at sweet spot, 0.0 at extremes
    focus = 1.0 - np.abs(d - 0.45) * 2.5
    focus = np.clip(focus, 0, 1)
    focus_3ch = focus.reshape(h, w, 1)

    # Blend sharp and blurry based on focus
    out = (rgb.astype(np.float32) * focus_3ch + blurred.astype(np.float32) * (1 - focus_3ch)).astype(np.uint8)
    return out


def effect_depth_desaturate(rgb, depth):
    """4 - Atmospheric perspective: far = desaturated and slightly blue"""
    if depth is None:
        return rgb
    d = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Reduce saturation based on depth (far = less saturated)
    hsv[:, :, 1] = hsv[:, :, 1] * (1 - d * 0.7)

    # Slight blue tint for distance
    hsv[:, :, 0] = hsv[:, :, 0] * (1 - d * 0.3) + 100 * d * 0.3

    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def effect_depth_lag(rgb, depth):
    """8 - Temporal depth: far things ghost/lag behind (creepy)"""
    global _lag_buffer, lag_intensity
    if depth is None:
        return rgb

    h, w = rgb.shape[:2]

    # Initialize buffer if needed
    if _lag_buffer is None or _lag_buffer.shape != rgb.shape:
        globals()['_lag_buffer'] = rgb.copy().astype(np.float32)

    d = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h))
    d_3ch = d.reshape(h, w, 1)

    # Update rate: close = fast, far = very slow (ghosty)
    # lag_intensity controls how slow far things update
    close_rate = 0.95  # Close things update fast
    far_rate = 1.0 - lag_intensity  # Far things update slow (0.05 at max)
    update_rate = close_rate - d_3ch * (close_rate - far_rate)

    # Blend current frame into buffer based on depth
    globals()['_lag_buffer'] = _lag_buffer * (1 - update_rate) + rgb.astype(np.float32) * update_rate

    # Add slight color shift to lagged areas for visibility
    out = _lag_buffer.copy()
    lag_amount = (1 - update_rate) * d_3ch  # How much each pixel is lagging
    # Slight blue/cool tint on ghosted areas
    out[:, :, 0] = out[:, :, 0] + lag_amount[:, :, 0] * 20  # add blue
    out[:, :, 2] = out[:, :, 2] - lag_amount[:, :, 0] * 10  # reduce red

    return np.clip(out, 0, 255).astype(np.uint8)


def effect_depth_grain(rgb, depth):
    """C - Film grain that increases with distance (subtle wrongness)"""
    if depth is None:
        return rgb
    h, w = rgb.shape[:2]
    d = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h))

    # Generate noise
    noise = np.random.normal(0, 30, rgb.shape).astype(np.float32)

    # Apply noise based on depth (more grain = farther)
    d_3ch = d.reshape(h, w, 1)
    out = rgb.astype(np.float32) + noise * d_3ch * 0.7

    return np.clip(out, 0, 255).astype(np.uint8)


def effect_depth_pixelate(rgb, depth):
    """H - Resolution decreases with distance"""
    if depth is None:
        return rgb
    h, w = rgb.shape[:2]
    d = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h))
    out = rgb.copy()

    # Process in blocks, pixelation based on average depth
    block = 8
    for by in range(0, h, block):
        for bx in range(0, w, block):
            bh = min(block, h - by)
            bw = min(block, w - bx)

            avg_d = np.mean(d[by:by+bh, bx:bx+bw])

            # Far = more pixelated
            if avg_d > 80:
                # Average the block colors
                avg_color = np.mean(rgb[by:by+bh, bx:bx+bw], axis=(0, 1))
                out[by:by+bh, bx:bx+bw] = avg_color

    return out


def effect_depth_glow(rgb, depth):
    """I - Subtle glow on depth edges (things feel outlined by light)"""
    if depth is None:
        return rgb
    d = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Find depth edges
    edges = cv2.Canny(d, 30, 80)

    # Dilate for glow
    kernel = np.ones((5, 5), np.uint8)
    glow = cv2.dilate(edges, kernel, iterations=2)
    glow_blur = cv2.GaussianBlur(glow, (15, 15), 0)

    # Add subtle warm glow
    glow_color = np.zeros_like(rgb, dtype=np.float32)
    glow_color[:, :, 1] = glow_blur * 0.3  # slight green
    glow_color[:, :, 2] = glow_blur * 0.5  # warm

    out = rgb.astype(np.float32) + glow_color
    return np.clip(out, 0, 255).astype(np.uint8)


def effect_parallax(rgb, depth):
    """J - Subtle parallax shift based on depth (fake head tracking)"""
    global frame_count
    if depth is None:
        return rgb
    h, w = rgb.shape[:2]
    d = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h))

    # Subtle oscillating "head position"
    head_x = math.sin(frame_count / 60) * 8

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            # Parallax: closer things shift more opposite to "head"
            shift = head_x * (1 - d[y, x])
            map_x[y, x] = x + shift
            map_y[y, x] = y

    return cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def effect_selective_real(rgb, depth):
    """K - Only one depth layer looks normal, rest is slightly off"""
    if depth is None:
        return rgb
    h, w = rgb.shape[:2]
    d = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h))

    # "Real" zone is mid-depth
    real_zone = np.exp(-((d - 0.4) ** 2) / 0.05)  # Gaussian around 0.4
    real_zone_3ch = real_zone.reshape(h, w, 1)

    # Make non-real areas slightly wrong
    weird = rgb.copy()
    # Slight color shift
    weird = cv2.cvtColor(weird, cv2.COLOR_BGR2HSV)
    weird[:, :, 0] = (weird[:, :, 0].astype(np.int16) + 10) % 180
    weird[:, :, 1] = np.clip(weird[:, :, 1].astype(np.int16) - 30, 0, 255).astype(np.uint8)
    weird = cv2.cvtColor(weird, cv2.COLOR_HSV2BGR)
    # Slight blur
    weird = cv2.GaussianBlur(weird, (5, 5), 0)

    out = (rgb.astype(np.float32) * real_zone_3ch + weird.astype(np.float32) * (1 - real_zone_3ch)).astype(np.uint8)
    return out


def effect_depth_shadow(rgb, depth):
    """L - Objects cast depth-based shadows (sun from above)"""
    if depth is None:
        return rgb
    h, w = rgb.shape[:2]
    d = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h))

    # Shift depth map down to create shadow offset
    shadow_offset = 15
    shadow = np.roll(d, shadow_offset, axis=0)
    shadow[:shadow_offset, :] = d[:shadow_offset, :]

    # Shadow where shifted depth is closer than actual depth
    shadow_mask = (shadow < d - 0.1).astype(np.float32)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (11, 11), 0)

    # Darken shadowed areas
    shadow_3ch = shadow_mask.reshape(h, w, 1)
    out = rgb.astype(np.float32) * (1 - shadow_3ch * 0.5)

    return out.astype(np.uint8)


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
    h, w = rgb.shape[:2]
    d = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h))

    out = np.zeros_like(rgb)
    for y in range(0, h - 6, 8):
        offset = int((d[y, w // 2] / 255) * 20 - 10)
        y2 = max(0, min(h - 6, y + offset))
        # Ensure source and destination slices are same size
        slice_h = min(6, h - y, h - y2)
        if slice_h > 0:
            out[y:y + slice_h] = rgb[y2:y2 + slice_h]

    return out


# === DEPTH-SELECTIVE INVERT (V mode) ===

def effect_depth_invert(rgb, depth):
    """5 - Background inverted, face/foreground normal (uses depth to separate)"""
    global depth_threshold, depth_softness
    if depth is None:
        return rgb
    h, w = rgb.shape[:2]
    d = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h))

    # Invert the full image
    inverted = cv2.bitwise_not(rgb)

    # Create smooth mask: close (low depth) = normal, far (high depth) = inverted
    # Use global threshold and softness for tuning
    mask = 1.0 / (1.0 + np.exp(-depth_softness * (d - depth_threshold)))
    mask_3ch = mask.reshape(h, w, 1)

    # Blend: mask=0 (close) -> rgb, mask=1 (far) -> inverted
    out = rgb.astype(np.float32) * (1 - mask_3ch) + inverted.astype(np.float32) * mask_3ch
    return out.astype(np.uint8)


# === VASARI + 3D COMBO EFFECTS (home row keys) ===

def effect_vasari_ripple(rgb, depth):
    """D - Breakup + depth ripple overlay"""
    # First apply breakup corruption
    corrupted = effect_vasari_breakup(rgb, depth)
    # Then apply ripple distortion
    if depth is None:
        return corrupted
    return effect_displace_ripple(corrupted, depth)


def effect_vasari_stretch(rgb, depth):
    """F - Breakup + depth stretch overlay"""
    corrupted = effect_vasari_breakup(rgb, depth)
    if depth is None:
        return corrupted
    return effect_displace_stretch(corrupted, depth)


def effect_vasari_twist(rgb, depth):
    """G - Breakup + depth twist overlay"""
    corrupted = effect_vasari_breakup(rgb, depth)
    if depth is None:
        return corrupted
    return effect_displace_twist(corrupted, depth)


def effect_vasari_pixelate(rgb, depth):
    """H - Breakup + depth pixelate overlay"""
    corrupted = effect_vasari_breakup(rgb, depth)
    if depth is None:
        return corrupted
    return effect_depth_pixelate(corrupted, depth)


def effect_vasari_parallax(rgb, depth):
    """J - Breakup + parallax wobble overlay"""
    corrupted = effect_vasari_breakup(rgb, depth)
    if depth is None:
        return corrupted
    return effect_parallax(corrupted, depth)


def effect_vasari_selective(rgb, depth):
    """K - Breakup + selective real layer"""
    corrupted = effect_vasari_breakup(rgb, depth)
    if depth is None:
        return corrupted
    return effect_selective_real(corrupted, depth)


def effect_vasari_shadow(rgb, depth):
    """L - Breakup + depth shadow overlay"""
    corrupted = effect_vasari_breakup(rgb, depth)
    if depth is None:
        return corrupted
    return effect_depth_shadow(corrupted, depth)


# === EFFECTS MAP (includes both lowercase and uppercase) ===
effects = {
    # RGB modes (2D)
    ord('1'): ("Normal", effect_normal),
    ord('2'): ("Edge", effect_edge),
    ord('i'): ("Invert", effect_invert),
    ord('I'): ("Invert", effect_invert),
    ord('6'): ("Thermal", effect_thermal_rgb),
    ord('v'): ("VASARI", effect_vasari),
    ord('V'): ("VASARI", effect_vasari),
    ord('b'): ("Breakup", effect_vasari_breakup),
    ord('B'): ("Breakup", effect_vasari_breakup),

    # UNCANNY 3D (subtle/strange depth effects)
    ord('3'): ("DepthFocus", effect_depth_focus),
    ord('4'): ("Atmosphere", effect_depth_desaturate),
    ord('5'): ("DepthInvert", effect_depth_invert),
    ord('8'): ("DepthLag", effect_depth_lag),
    ord('c'): ("DepthGrain", effect_depth_grain),
    ord('C'): ("DepthGrain", effect_depth_grain),
    ord('o'): ("DepthGlow", effect_depth_glow),
    ord('O'): ("DepthGlow", effect_depth_glow),

    # VASARI + 3D COMBOS (home row)
    ord('d'): ("VASARI+Ripple", effect_vasari_ripple),
    ord('D'): ("VASARI+Ripple", effect_vasari_ripple),
    ord('f'): ("VASARI+Stretch", effect_vasari_stretch),
    ord('F'): ("VASARI+Stretch", effect_vasari_stretch),
    ord('g'): ("VASARI+Twist", effect_vasari_twist),
    ord('G'): ("VASARI+Twist", effect_vasari_twist),
    ord('h'): ("VASARI+Pixel", effect_vasari_pixelate),
    ord('H'): ("VASARI+Pixel", effect_vasari_pixelate),
    ord('j'): ("VASARI+Parallax", effect_vasari_parallax),
    ord('J'): ("VASARI+Parallax", effect_vasari_parallax),
    ord('k'): ("VASARI+Selective", effect_vasari_selective),
    ord('K'): ("VASARI+Selective", effect_vasari_selective),
    ord('l'): ("VASARI+Shadow", effect_vasari_shadow),
    ord('L'): ("VASARI+Shadow", effect_vasari_shadow),

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

    # Displacement family (A/S)
    ord('a'): ("Disp-Wave", effect_displace_wave),
    ord('A'): ("Disp-Wave", effect_displace_wave),
    ord('s'): ("Disp-Shatter", effect_displace_shatter),
    ord('S'): ("Disp-Shatter", effect_displace_shatter),

    # Contour family (7)
    ord('7'): ("Contours", effect_contours_base),
    ord('z'): ("Cont-Thick", effect_contours_thick),
    ord('Z'): ("Cont-Thick", effect_contours_thick),
    ord('x'): ("Cont-Filled", effect_contours_filled),
    ord('X'): ("Cont-Filled", effect_contours_filled),

    # Other 3D
    ord('0'): ("Portal", effect_portal_fixed),
}


def main():
    global line_weight, blend_amount, frame_count, zoom_level, lag_intensity, puppet_mode, vasari_wave_start, breakup_wave_start
    global corrupt_intensity, corrupt_speed, face_pop_amount, depth_threshold, depth_softness

    print(__doc__)

    # DepthAI v3 API
    pipeline = dai.Pipeline()

    # RGB Camera (v3 API uses Camera node with .build())
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam_out = cam.requestOutput((W, H), dai.ImgFrame.Type.BGR888p)
    q_rgb = cam_out.createOutputQueue(maxSize=2, blocking=False)

    # Stereo cameras (v3 API)
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    left_out = left.requestOutput((640, 480))

    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    right_out = right.requestOutput((640, 480))

    # Stereo depth
    stereo = pipeline.create(dai.node.StereoDepth)
    left_out.link(stereo.left)
    right_out.link(stereo.right)
    q_depth = stereo.depth.createOutputQueue(maxSize=2, blocking=False)

    mode = ord('V')  # Start in VASARI mode
    print("Starting in VASARI mode (V)")

    pipeline.start()

    try:
        with pyvirtualcam.Camera(width=W, height=H, fps=FPS) as vcam:
            print(f"Vcam: {vcam.device}")

            while pipeline.isRunning():
                frame_count += 1

                rgb_msg = q_rgb.tryGet()
                if rgb_msg is None:
                    continue
                rgb = rgb_msg.getCvFrame()
                rgb = crop_center(rgb, zoom_level)

                depth = None
                depth_msg = q_depth.tryGet()
                if depth_msg is not None:
                    depth = depth_msg.getFrame()
                    depth = crop_center(depth, zoom_level)
                    if depth.shape[:2] != (H, W):
                        depth = cv2.resize(depth, (W, H))

                name, fx = effects.get(mode, ("Normal", effect_normal))
                try:
                    out = fx(rgb, depth)
                except Exception as e:
                    print(f"Effect error: {e}")
                    out = rgb.copy()

                # ALWAYS make array contiguous and correct type for OpenCV
                if out is None:
                    out = rgb.copy()
                out = np.ascontiguousarray(out, dtype=np.uint8)
                if out.shape != rgb.shape:
                    out = cv2.resize(out, (rgb.shape[1], rgb.shape[0]))

                # All VASARI-based modes (V/B and home row combos)
                vasari_modes = [ord('v'), ord('V'), ord('b'), ord('B'), ord('d'), ord('D'), ord('f'), ord('F'),
                                ord('g'), ord('G'), ord('h'), ord('H'), ord('j'), ord('J'),
                                ord('k'), ord('K'), ord('l'), ord('L')]

                # HUD
                info = f"{name}"
                if zoom_level > 1.0:
                    info += f" | Zoom: {zoom_level:.1f}x"
                if mode == ord('2'):
                    info += f" | Weight: {line_weight}"
                if mode in vasari_modes:
                    if puppet_mode:
                        info += f" | PUPPET"
                    info += f" | I:{corrupt_intensity:.1f} S:{corrupt_speed:.1f} F:{face_pop_amount:.1f}"
                if mode == ord('5'):
                    info += f" | T:{depth_threshold:.2f} S:{depth_softness:.1f}"
                if mode == ord('8'):
                    info += f" | Lag: {int(lag_intensity * 100)}%"

                cv2.putText(out, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                vcam.send(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
                cv2.imshow("Vasari V4", out)

                k = cv2.waitKey(1) & 0xFF
                # Mode switching
                if k in effects:
                    # Clear lag buffer when leaving DepthLag mode
                    if mode == ord('8') and k != ord('8'):
                        globals()['_lag_buffer'] = None
                    mode = k
                    print(f"Mode: {effects[k][0]}")

                # === ZOOM: - out, = in ===
                elif k == ord('-'):
                    zoom_level = max(1.0, zoom_level - 0.5)
                    print(f"Zoom: {zoom_level:.1f}x")
                elif k == ord('='):
                    zoom_level = min(5.0, zoom_level + 0.5)
                    print(f"Zoom: {zoom_level:.1f}x")

                # === DEPTH INVERT TUNING (mode 5) ===
                # < > = Threshold (where cut happens)
                elif k == ord('<') and mode == ord('5'):
                    depth_threshold = max(0.1, depth_threshold - 0.05)
                    print(f"Depth threshold: {depth_threshold:.2f}")
                elif k == ord('>') and mode == ord('5'):
                    depth_threshold = min(0.9, depth_threshold + 0.05)
                    print(f"Depth threshold: {depth_threshold:.2f}")
                # , . = Softness (edge sharpness)
                elif k == ord(',') and mode == ord('5'):
                    depth_softness = max(1.0, depth_softness - 1.0)
                    print(f"Depth softness: {depth_softness:.1f}")
                elif k == ord('.') and mode == ord('5'):
                    depth_softness = min(20.0, depth_softness + 1.0)
                    print(f"Depth softness: {depth_softness:.1f}")

                # === VASARI TUNING (V/B and combos) ===
                # [ ] = Speed
                elif k == ord('[') and mode in vasari_modes:
                    corrupt_speed = max(0.3, corrupt_speed - 0.2)
                    print(f"Speed: {corrupt_speed:.1f}")
                elif k == ord(']') and mode in vasari_modes:
                    corrupt_speed = min(3.0, corrupt_speed + 0.2)
                    print(f"Speed: {corrupt_speed:.1f}")
                # { } = Intensity
                elif k == ord('{') and mode in vasari_modes:
                    corrupt_intensity = max(0.2, corrupt_intensity - 0.2)
                    print(f"Intensity: {corrupt_intensity:.1f}")
                elif k == ord('}') and mode in vasari_modes:
                    corrupt_intensity = min(3.0, corrupt_intensity + 0.2)
                    print(f"Intensity: {corrupt_intensity:.1f}")
                # ; ' = Face pop
                elif k == ord(';') and mode in vasari_modes:
                    face_pop_amount = max(0.0, face_pop_amount - 0.1)
                    print(f"Face pop: {face_pop_amount:.1f}")
                elif k == ord("'") and mode in vasari_modes:
                    face_pop_amount = min(1.0, face_pop_amount + 0.1)
                    print(f"Face pop: {face_pop_amount:.1f}")
                # _ + = Face pop (shift - =)
                elif k == ord('_') and mode in vasari_modes:
                    face_pop_amount = max(0.0, face_pop_amount - 0.1)
                    print(f"Face pop: {face_pop_amount:.1f}")
                elif k == ord('+') and mode in vasari_modes:
                    face_pop_amount = min(1.0, face_pop_amount + 0.1)
                    print(f"Face pop: {face_pop_amount:.1f}")
                # N M = Blend with original
                elif k == ord('n') or k == ord('N'):
                    if mode in vasari_modes:
                        blend_amount = max(0.0, blend_amount - 0.1)
                        print(f"Blend: {int(blend_amount * 100)}%")
                elif k == ord('m') or k == ord('M'):
                    if mode in vasari_modes:
                        blend_amount = min(1.0, blend_amount + 0.1)
                        print(f"Blend: {int(blend_amount * 100)}%")

                # === DEPTH LAG TUNING (mode 8) ===
                elif k == ord(',') and mode == ord('8'):
                    lag_intensity = max(0.5, lag_intensity - 0.05)
                    print(f"Lag intensity: {int(lag_intensity * 100)}%")
                elif k == ord('.') and mode == ord('8'):
                    lag_intensity = min(0.99, lag_intensity + 0.05)
                    print(f"Lag intensity: {int(lag_intensity * 100)}%")

                # === EDGE MODE TUNING (mode 2) ===
                elif k == ord(',') and mode == ord('2'):
                    line_weight = max(1, line_weight - 1)
                    print(f"Line weight: {line_weight}")
                elif k == ord('.') and mode == ord('2'):
                    line_weight = min(10, line_weight + 1)
                    print(f"Line weight: {line_weight}")

                # SPACE = toggle PUPPET MODE in VASARI modes
                elif k == ord(' '):
                    if mode in vasari_modes:
                        puppet_mode = not puppet_mode
                        if puppet_mode:
                            vasari_wave_start = frame_count
                            breakup_wave_start = frame_count
                            print("PUPPET ON")
                        else:
                            print("PUPPET OFF")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
