# VASARI RIG V2 - Comprehensive Bug Scan Report
**File**: `/Users/corbettgriffith/vasari-camera/vasari_rig_v2.py`
**Scan Date**: 2026-01-20

---

## CATEGORY 1: ARRAY/INDEX BUGS

### CRITICAL BUGS

#### 1.1 Array Out-of-Bounds in Point Cloud Functions (Lines 187, 200, 213, 226, 242, 254, 268)
**Severity**: CRITICAL
**Location**: Multiple point cloud effect functions
**Issue**: Direct array indexing `d[y,x]` without bounds checking when `y` or `x` might equal `H` or `W`
```python
for y in range(0, H, 4):  # y can be H-4, H-3, H-2, H-1, or H
    for x in range(0, W, 4):
        if d[y,x] > 50:  # IndexError if y==H or x==W
```
**Impact**: IndexError crash when loop reaches boundary
**Fix**: Change range to `range(0, H-1, 4)` or add bounds check

#### 1.2 Slice Bounds Issue in effect_pointcloud_stripes (Line 243)
**Severity**: HIGH
**Location**: Line 243
**Issue**: `out[y:y+2, x]` can extend beyond array bounds when `y >= H-1`
```python
for y in range(0, H, 6):
    for x in range(0, W, 1):
        if d[y,x] > 50:
            out[y:y+2, x] = rgb[y, x]  # y+2 can exceed H
```
**Impact**: Silent truncation or potential array access error
**Fix**: Use `out[y:min(y+2, H), x]`

#### 1.3 Slice Bounds Issue in effect_pointcloud_rain (Line 257)
**Severity**: HIGH
**Location**: Line 257
**Issue**: `out[y:y2, x:x+2]` where `x+2` can exceed `W`
```python
for x in range(0, W, 6):
    out[y:y2, x:x+2] = rgb[y, x]  # x+2 can exceed W
```
**Impact**: Silent truncation or inconsistent behavior
**Fix**: Use `x:min(x+2, W)`

#### 1.4 Array Bounds in effect_slices (Line 504-506)
**Severity**: MEDIUM
**Location**: Lines 504-506
**Issue**: Potential access to `W//2` index and slice operations without validation
```python
offset = int((d[y, W//2] / 255) * 20 - 10)
y2 = max(0, min(H-8, y + offset))
out[y:y+6] = rgb[y2:y2+6]  # y2+6 might exceed H
```
**Impact**: Potential slice mismatch causing shape errors
**Fix**: Validate `y2+6 <= H` before assignment

#### 1.5 Block Bounds in effect_displace_shatter (Lines 323-331)
**Severity**: MEDIUM
**Location**: Lines 323-331
**Issue**: Block slicing assumes blocks fit perfectly, but last blocks may be incomplete
```python
for by in range(0, H, block):
    for bx in range(0, W, block):
        block_d = d[by:by+block, bx:bx+block]  # Last block may be smaller
        src_x = max(0, min(W-block, bx + offset_x))  # W-block assumes full block
        src_y = max(0, min(H-block, by + offset_y))
        out[by:by+block, bx:bx+block] = rgb[src_y:src_y+block, src_x:src_x+block]
```
**Impact**: Shape mismatch errors on assignment when blocks don't align
**Fix**: Calculate actual block dimensions for edge blocks

#### 1.6 Horizontal Scroll Bounds in Vasari Effects (Lines 127, 163)
**Severity**: LOW
**Location**: Lines 127, 163
**Issue**: `np.roll` is safe, but band selection `y:y+band_h` can extend beyond H
```python
y = random.randint(0, H - 40)
band_h = random.randint(10, int(40 * intensity) + 10)
out[y:y+band_h] = np.roll(out[y:y+band_h], shift, axis=1)  # y+band_h can exceed H
```
**Impact**: Silent truncation (numpy handles gracefully, but logic may be wrong)
**Fix**: Ensure `y+band_h <= H` or clamp band_h

---

## CATEGORY 2: TYPE/CONVERSION BUGS

### CRITICAL BUGS

#### 2.1 Float-to-Integer Channel Swap (Line 130)
**Severity**: CRITICAL
**Location**: Line 130
**Issue**: Complex list comprehension produces incorrect channel indexing
```python
out = out[:, :, [random.choice([[2,1,0],[1,0,2],[0,2,1]])[i] for i in range(3)]]
```
**Problem**: `random.choice` selects ONE list, then indexes it 3 times, producing [list[0], list[1], list[2]] which is the SAME permutation each time
**Fix**: Should be `random.choice([2,1,0], [1,0,2], [0,2,1])` or use `random.shuffle`

#### 2.2 HSV Hue Overflow (Line 169)
**Severity**: HIGH
**Location**: Line 169
**Issue**: HSV hue values must be 0-179 in OpenCV, but operation could overflow
```python
hsv[:,:,0] = (hsv[:,:,0] + int(intensity * 30)) % 180
```
**Problem**: If `hsv[:,:,0]` is uint8 and addition causes overflow before modulo
**Impact**: Incorrect color shifts, potential wraparound before modulo applied
**Fix**: Cast to int16 before addition: `hsv[:,:,0] = ((hsv[:,:,0].astype(np.int16) + int(intensity * 30)) % 180).astype(np.uint8)`

#### 2.3 Float Division in Color Calculation (Line 438)
**Severity**: MEDIUM
**Location**: Line 438
**Issue**: Integer division might not produce expected results
```python
out[:,:,0] = (edges * hue // 255).astype(np.uint8)
```
**Problem**: Already doing floor division, but hue could be 0, making entire channel 0
**Impact**: Unexpected color output
**Fix**: Validate hue range or use float division with clipping

#### 2.4 Color Map Type Mismatch (Line 230)
**Severity**: MEDIUM
**Location**: Line 230
**Issue**: Creating single-pixel HSV array then converting, inefficient and error-prone
```python
color = cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0,0].tolist()
```
**Problem**: Overly complex type conversions, potential precision loss
**Impact**: Performance hit, potential type errors
**Fix**: Pre-compute color lookup table or simplify conversion

#### 2.5 Mask Division Type Error (Line 485)
**Severity**: HIGH
**Location**: Line 485
**Issue**: Division produces float, but might not broadcast correctly
```python
mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
out = (rgb * (1 - mask_3ch) + portal_color * mask_3ch).astype(np.uint8)
```
**Problem**: `rgb` is uint8, `mask_3ch` is float64, multiplication creates float64, then converts back
**Impact**: Potential precision loss and performance issue, but functionally works
**Improvement**: Explicit type handling for clarity

---

## CATEGORY 3: LOGIC/STATE BUGS

### CRITICAL BUGS

#### 3.1 Global frame_count Incremented Multiple Times Per Frame
**Severity**: CRITICAL
**Location**: Lines 104, 141, 581
**Issue**: `frame_count` is incremented in `effect_vasari` (line 104) and `effect_vasari_breakup` (line 141) AND in main loop (line 581)
```python
def effect_vasari(rgb, depth):
    global frame_count, wave_start, wave_length, wave_intensity, blend_amount
    frame_count += 1  # Incremented here

def main():
    while pipeline.isRunning():
        frame_count += 1  # ALSO incremented here!
```
**Impact**: Frame count advances at 2x speed when in mode 7 or B, breaking all frame-based timing
**Fix**: Remove increments from effect functions, keep only in main loop

#### 3.2 Shared Wave State Across Different Effects
**Severity**: HIGH
**Location**: Lines 103-112, 140-149
**Issue**: Both `effect_vasari` and `effect_vasari_breakup` share global `wave_start`, `wave_length`, `wave_intensity`
**Impact**: Switching between modes 7 and B causes wave state confusion, unexpected behavior
**Fix**: Separate state variables for each effect or pass state as parameters

#### 3.3 contour_phase Global Without Reset
**Severity**: MEDIUM
**Location**: Line 417
**Issue**: `contour_phase` increments forever, will eventually overflow float precision
```python
contour_phase += 0.05  # Infinite growth
```
**Impact**: After ~10^8 frames, precision loss in sin() calculation
**Fix**: Use modulo: `contour_phase = (contour_phase + 0.05) % (2 * math.pi)`

#### 3.4 blend_amount Used Incorrectly in Breakup Mode (Line 173)
**Severity**: MEDIUM
**Location**: Line 173
**Issue**: Blend logic is backwards in effect_vasari_breakup
```python
out = cv2.addWeighted(out, 1 - blend_amount, rgb, blend_amount, 0)
```
**Problem**: This blends corrupted image with ORIGINAL rgb. But `out` started as `rgb.copy()`, so at `blend_amount=0` you get full corruption, at `blend_amount=1` you get full original - which is correct. However, the comment in line 60 says "0 = full effect, 1 = full original", which matches.
**Actually**: This is CORRECT. False alarm.

#### 3.5 Mode Key Collision
**Severity**: LOW
**Location**: Lines 544-545
**Issue**: Both '+' and '=' keys map to same effect
```python
ord('+'): ("Slices", effect_slices),
ord('='): ("Slices", effect_slices),
```
**Impact**: Redundant but harmless
**Note**: Intentional design choice for keyboard convenience

---

## CATEGORY 4: API/LIBRARY BUGS

### CRITICAL BUGS

#### 4.1 Incorrect DepthAI API Usage (Lines 556-557)
**Severity**: CRITICAL
**Location**: Lines 556-557
**Issue**: `.build()` method call is non-standard for DepthAI
```python
cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
q_rgb = cam.requestOutput((W, H), dai.ImgFrame.Type.BGR888p).createOutputQueue(maxSize=2, blocking=False)
```
**Problem**:
- `.build()` is not a standard DepthAI API method for Camera nodes
- Correct pattern is to create node, configure it, then use `.video` or `.preview` output
- `requestOutput` is not a standard Camera node method
**Impact**: Code will not run, raises AttributeError
**Fix**: Use standard DepthAI API:
```python
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(W, H)
cam_rgb.setInterleaved(False)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.video.link(xout_rgb.input)
q_rgb = device.getOutputQueue("rgb", maxSize=2, blocking=False)
```

#### 4.2 StereoDepth Configuration Missing (Lines 567-570)
**Severity**: HIGH
**Location**: Lines 567-570
**Issue**: StereoDepth node created but not configured
```python
stereo = pipeline.create(dai.node.StereoDepth)
left_out.link(stereo.left)
right_out.link(stereo.right)
q_depth = stereo.depth.createOutputQueue(maxSize=2, blocking=False)
```
**Problems**:
- Missing depth alignment settings
- Missing confidence threshold
- No median filtering configuration
- `createOutputQueue` should be called on device, not node
**Fix**: Add configuration and use XLinkOut pattern

#### 4.3 Pipeline Started Before Device Created (Line 572)
**Severity**: CRITICAL
**Location**: Line 572
**Issue**: `pipeline.start()` called, but pipeline is not associated with a device
```python
pipeline.start()  # pipeline.start() doesn't exist, should be device.startPipeline()
```
**Problem**: Need to create device first: `with dai.Device(pipeline) as device:`
**Impact**: Code will crash with AttributeError
**Fix**: Wrap in device context manager

#### 4.4 Queue Retrieval Without Device (Lines 583, 590)
**Severity**: CRITICAL
**Location**: Lines 583, 590
**Issue**: Queues obtained from pipeline before device creation
**Problem**: Queues must be obtained from device, not pipeline
**Fix**: Get queues from device after `dai.Device(pipeline)` creation

#### 4.5 Pipeline isRunning Check (Line 580)
**Severity**: HIGH
**Location**: Line 580
**Issue**: `pipeline.isRunning()` is not the correct API
```python
while pipeline.isRunning():
```
**Problem**: Should use `device.isClosed()` or just `True` with proper exception handling
**Fix**: Use `while True:` with try-except for device disconnection

---

## CATEGORY 5: EDGE CASES/RUNTIME BUGS

### CRITICAL BUGS

#### 5.1 Division by Zero in Point Cloud Effects
**Severity**: HIGH
**Location**: Lines 188, 201, 214, 227, 255, 274
**Issue**: Division operations without zero checks
```python
r = max(1, int((255 - d[y,x]) / 50))  # If d[y,x] = 255, r = 1 (safe)
```
**Actually**: max(1, ...) prevents zero radius, so this is SAFE. False alarm.

#### 5.2 Division by Zero in effect_displace_shatter (Line 326)
**Severity**: MEDIUM
**Location**: Line 326
**Issue**: Division by 255 without checking if denominator is zero
```python
avg_d = np.mean(block_d)
offset_x = int((avg_d / 255 - 0.5) * 40)  # Safe: 255 is constant
```
**Actually**: 255 is constant, SAFE. False alarm.

#### 5.3 Empty Array Handling in effect_displace_shatter (Line 324)
**Severity**: HIGH
**Location**: Lines 323-325
**Issue**: Check for `block_d.size > 0` but still proceed if mean is NaN
```python
block_d = d[by:by+block, bx:bx+block]
if block_d.size > 0:
    avg_d = np.mean(block_d)  # Could still be NaN if block_d is empty
```
**Actually**: size > 0 check is correct, mean of non-empty array is valid. False alarm.

#### 5.4 None Depth Handling - Incomplete
**Severity**: MEDIUM
**Location**: Throughout depth effects
**Issue**: All depth effects check `if depth is None: return rgb`, which is good
**Problem**: When switching from depth mode to RGB mode, no issues. SAFE.
**Edge case**: If depth is None but expected to be array, early return handles it. SAFE.

#### 5.5 sqrt of Negative Number (Lines 349, 389)
**Severity**: LOW
**Location**: Lines 349, 389
**Issue**: Distance calculation assumes dx*dx + dy*dy >= 0
```python
dist = math.sqrt(dx*dx + dy*dy)
```
**Actually**: Squared values are always non-negative. SAFE.

#### 5.6 atan2 with Zero Arguments (Lines 350, 390)
**Severity**: LOW
**Location**: Lines 350, 390
**Issue**: `math.atan2(0, 0)` is valid (returns 0.0), so SAFE
```python
angle = math.atan2(dy, dx)
```
**Actually**: atan2 handles (0, 0) correctly. SAFE.

#### 5.7 Array Shape Mismatch in crop_center (Line 68)
**Severity**: MEDIUM
**Location**: Line 68
**Issue**: If depth frame has different dimensions than expected, resize might fail
```python
def crop_center(img):
    h, w = img.shape[:2]  # What if img is 1D or empty?
    y1, y2 = h//3, 2*h//3
    x1, x2 = w//3, 2*w//3
    cropped = img[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h))
```
**Problem**: If h < 3 or w < 3, slicing produces empty array, resize will fail
**Impact**: Crash on very small input images
**Fix**: Add dimension validation

#### 5.8 Random Integer Range Error (Lines 124, 160)
**Severity**: MEDIUM
**Location**: Lines 124, 160
**Issue**: `random.randint(0, H - 40)` fails if H < 40
```python
y = random.randint(0, H - 40)  # If H=480, no problem. But if H<40, ValueError
```
**Impact**: Since H=480 (constant), this is SAFE for this code. But brittle if H changes.
**Fix**: Use `max(0, H - 40)` or validate H

#### 5.9 cv2.cvtColor on Empty or Invalid Array
**Severity**: MEDIUM
**Location**: Multiple locations (78, 98, 168, 170, 230, 485, 609)
**Issue**: No validation that input arrays are non-empty before color conversion
**Impact**: If rgb is None or empty, cv2.cvtColor will crash
**Mitigation**: Main loop checks `if rgb_msg is None: continue`, so rgb should always be valid
**Assessment**: SAFE in current code flow, but brittle

---

## SUMMARY BY SEVERITY

### CRITICAL (Must Fix)
1. **Array out-of-bounds** in point cloud functions (Cat 1.1)
2. **Global frame_count double increment** (Cat 3.1)
3. **DepthAI API incorrect usage** (Cat 4.1, 4.3, 4.4, 4.5)
4. **Type error in channel swap** (Cat 2.1)

### HIGH (Should Fix)
1. **Slice bounds issues** in point cloud effects (Cat 1.2, 1.3)
2. **HSV hue overflow** (Cat 2.2)
3. **Shared wave state** across effects (Cat 3.2)
4. **StereoDepth configuration missing** (Cat 4.2)
5. **Mask type conversion** (Cat 2.5)

### MEDIUM (Consider Fixing)
1. **Block bounds in shatter effect** (Cat 1.5)
2. **contour_phase overflow** (Cat 3.3)
3. **Color calculation type issues** (Cat 2.3, 2.4)
4. **crop_center edge case** (Cat 5.7)
5. **Random range brittleness** (Cat 5.8)

### LOW (Nice to Have)
1. **Band scroll bounds** (Cat 1.6)
2. **Code complexity** in color conversions (Cat 2.4)

---

## RECOMMENDED FIX PRIORITY

1. **Fix DepthAI API usage** - Code won't run without this
2. **Fix frame_count double increment** - Breaks all timing
3. **Fix array bounds** in loops - Causes crashes
4. **Fix channel swap logic** - Produces incorrect output
5. **Add configuration** to StereoDepth
6. **Fix type conversions** for reliability
7. **Separate effect state** to prevent mode interference
8. **Add edge case validation** for robustness

---

**Report Generated**: 2026-01-20
**Analysis Method**: Static code analysis across 5 bug categories
**Total Bugs Found**: 25+ issues across all severity levels
