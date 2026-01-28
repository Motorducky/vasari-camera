#!/usr/bin/env python3
"""VASARI CAM - Mode-based UI with number inputs + fixed aspect video"""

import cv2, numpy as np, webview, threading, base64, random, math, time

W, H, FPS = 640, 480, 30

# Try to import camera libs, but allow running without them
try:
    import depthai as dai
    import pyvirtualcam
    HAS_CAM = True
except:
    HAS_CAM = False

class S:
    fc=0; blend=0.0; zoom=1.0; lag_int=0.95; _lag=None; puppet=False
    intensity=1.0; speed=0.0; facepop=0.3; d_thresh=0.35; d_soft=8.0
    v_start=-999; v_len=200; v_int=0.7; b_start=-999; b_len=200; b_int=0.7
    _depth=None; mode='v'; name="VASARI"; run=True; frame=None; logo=True
    line_weight=2; contour_phase=0.0; invert=False; mirror=False
    _left=None; _right=None  # stereo camera frames
    hue_shift=0  # color shift for stereo effects (0-180)
    fuzz_frames=0  # frames remaining for universal fuzz effect
s = S()

# === ALL EFFECTS ===

def fx_norm(r,d): return r
def fx_inv(r,d): return cv2.bitwise_not(r)
def fx_therm(r,d): return cv2.applyColorMap(cv2.cvtColor(r,cv2.COLOR_BGR2GRAY),cv2.COLORMAP_INFERNO)

def fx_edge(r,d):
    g=cv2.cvtColor(r,cv2.COLOR_BGR2GRAY); e=cv2.Canny(g,50,150)
    if s.line_weight>1: e=cv2.dilate(e,np.ones((s.line_weight,s.line_weight),np.uint8),iterations=1)
    o=np.zeros_like(r); o[:,:,1]=e; o[:,:,0]=e//2; return o

def corrupt(out,rgb,h,w,ws,wl,wi,vasari=True):
    if s.speed>0: wmin,wmax,cmin,cmax=max(10,int(15/s.speed)),max(20,int(40/s.speed)),max(10,int(20/s.speed)),max(30,int(60/s.speed))
    else: wmin,wmax,cmin,cmax=15,40,9999999,9999999
    if s.puppet:
        if s.fc-ws>=wl: ws,wl,wi=s.fc,random.randint(wmin,wmax),1.0
    elif s.speed>0 and s.fc-ws>wl+random.randint(cmin,cmax) and random.random()>0.95:
        ws,wl,wi=s.fc,random.randint(wmax,wmax*3),random.uniform(0.7,1.0)
    if s.fc-ws<wl or s.puppet:
        inten=max(0,1.0-((s.fc-ws)/max(1,wl)))*wi*s.intensity
        if not vasari and s.facepop>0 and random.random()<s.facepop*0.25: inten*=0.3
        for _ in range(max(1,random.randint(int(8*s.intensity),int(25*s.intensity)+1))):
            y=random.randint(0,max(0,h-2)); bh=min(random.randint(2,max(3,int(50*inten)+2)),h-y)
            if bh>0: out[y:y+bh,:]=np.roll(out[y:y+bh,:],random.randint(-max(1,int(80*inten)),max(1,int(80*inten))),axis=1)
        if inten>0.3/s.intensity and h>60 and w>100:
            for _ in range(random.randint(2,int(6*s.intensity)+1)):
                bh,bw=random.randint(20,min(60,h-1)),random.randint(30,min(100,w-1))
                by,bx=random.randint(0,max(0,h-bh-1)),random.randint(0,max(0,w-bw-1))
                if by+bh<=h and bx+bw<=w:
                    blk=out[by:by+bh,bx:bx+bw].copy()
                    ny,nx=max(0,min(h-bh,by+random.randint(-30,30))),max(0,min(w-bw,bx+random.randint(-50,50)))
                    out[ny:ny+bh,nx:nx+bw]=blk
        if inten>0.2/s.intensity and random.random()>0.4:
            if vasari:
                c=random.randint(0,3)
                if c==0: out=np.ascontiguousarray(out[:,:,[2,1,0]])
                elif c==1: out=np.ascontiguousarray(out[:,:,[1,0,2]])
                elif c==2: out=np.ascontiguousarray(out[:,:,[0,2,1]])
                else: out[:,:,random.randint(0,2)]=np.roll(out[:,:,random.randint(0,2)],random.randint(-20,20),axis=1)
            else:
                hsv=cv2.cvtColor(out,cv2.COLOR_BGR2HSV); hsv[:,:,0]=((hsv[:,:,0].astype(np.int16)+random.randint(20,80))%180).astype(np.uint8)
                out=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        if inten>0.4/s.intensity and random.random()>0.5:
            blk=random.randint(max(2,int(4/s.intensity)),int(16*s.intensity))
            if w//blk>0 and h//blk>0: out=cv2.resize(cv2.resize(out,(w//blk,h//blk),interpolation=cv2.INTER_LINEAR),(w,h),interpolation=cv2.INTER_NEAREST)
        if inten>0.5/s.intensity and random.random()>0.6 and h>10:
            for _ in range(random.randint(3,10)):
                sy,dy=random.randint(0,max(0,h-6)),random.randint(0,max(0,h-6)); nl=min(random.randint(1,5),h-sy,h-dy)
                if nl>0: out[dy:dy+nl,:]=out[sy:sy+nl,:]
        if vasari:
            if s.facepop>0 and random.random()<s.facepop*0.3: out=cv2.addWeighted(out,1-random.uniform(0.3,0.8),rgb,random.uniform(0.3,0.8),0)
            if s.facepop>0.2 and random.random()<s.facepop*0.15 and h>100 and w>100:
                fy,fx=random.randint(0,h-80),random.randint(0,w-80); fh,fw=min(random.randint(40,120),h-fy),min(random.randint(40,120),w-fx)
                out[fy:fy+fh,fx:fx+fw]=rgb[fy:fy+fh,fx:fx+fw]
    return out,ws,wl,wi

def fx_vasari(r,d):
    o=cv2.bitwise_not(r); h,w=o.shape[:2]
    o,s.v_start,s.v_len,s.v_int=corrupt(o,r,h,w,s.v_start,s.v_len,s.v_int,True)
    if s.blend>0: o=cv2.addWeighted(o,1-s.blend,r,s.blend,0)
    return o

def fx_break(r,d):
    o=r.copy(); h,w=o.shape[:2]
    o,s.b_start,s.b_len,s.b_int=corrupt(o,r,h,w,s.b_start,s.b_len,s.b_int,False)
    if s.blend>0: o=cv2.addWeighted(o,1-s.blend,r,s.blend,0)
    return o

# Depth effects
def fx_dfocus(r,d):
    if d is None: return r
    h,w=r.shape[:2]; dn=cv2.normalize(d,None,0,1,cv2.NORM_MINMAX).astype(np.float32)
    if dn.shape[:2]!=(h,w): dn=cv2.resize(dn,(w,h))
    bl=cv2.GaussianBlur(r,(21,21),0); f=np.clip(1.0-np.abs(dn-0.45)*2.5,0,1).reshape(h,w,1)
    return (r.astype(np.float32)*f+bl.astype(np.float32)*(1-f)).astype(np.uint8)

def fx_atmos(r,d):
    if d is None: return r
    dn=cv2.normalize(d,None,0,1,cv2.NORM_MINMAX).astype(np.float32)
    hsv=cv2.cvtColor(r,cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1]*=(1-dn*0.7); hsv[:,:,0]=hsv[:,:,0]*(1-dn*0.3)+100*dn*0.3
    return cv2.cvtColor(np.clip(hsv,0,255).astype(np.uint8),cv2.COLOR_HSV2BGR)

def fx_dinv(r,d):
    if d is None: return r
    h,w=r.shape[:2]; dn=cv2.normalize(d,None,0,1,cv2.NORM_MINMAX).astype(np.float32)
    if dn.shape[:2]!=(h,w): dn=cv2.resize(dn,(w,h))
    inv=cv2.bitwise_not(r); m=(1.0/(1.0+np.exp(-s.d_soft*(dn-s.d_thresh)))).reshape(h,w,1)
    return (r.astype(np.float32)*(1-m)+inv.astype(np.float32)*m).astype(np.uint8)

def fx_dlag(r,d):
    if d is None: return r
    h,w=r.shape[:2]
    if s._lag is None or s._lag.shape!=r.shape: s._lag=r.copy().astype(np.float32)
    dn=cv2.normalize(d,None,0,1,cv2.NORM_MINMAX).astype(np.float32)
    if dn.shape[:2]!=(h,w): dn=cv2.resize(dn,(w,h))
    d3=dn.reshape(h,w,1); rate=0.95-d3*(0.95-(1.0-s.lag_int))
    s._lag=s._lag*(1-rate)+r.astype(np.float32)*rate
    o=s._lag.copy(); lag=(1-rate)*d3; o[:,:,0]+=lag[:,:,0]*20; o[:,:,2]-=lag[:,:,0]*10
    return np.clip(o,0,255).astype(np.uint8)

def fx_dgrain(r,d):
    if d is None: return r
    h,w=r.shape[:2]; dn=cv2.normalize(d,None,0,1,cv2.NORM_MINMAX).astype(np.float32)
    if dn.shape[:2]!=(h,w): dn=cv2.resize(dn,(w,h))
    n=np.random.normal(0,30,r.shape).astype(np.float32)
    return np.clip(r.astype(np.float32)+n*dn.reshape(h,w,1)*0.7,0,255).astype(np.uint8)

def fx_dglow(r,d):
    if d is None: return r
    dn=cv2.normalize(d,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    e=cv2.Canny(dn,30,80); gl=cv2.GaussianBlur(cv2.dilate(e,np.ones((5,5),np.uint8),iterations=2),(15,15),0)
    gc=np.zeros_like(r,dtype=np.float32); gc[:,:,1]=gl*0.3; gc[:,:,2]=gl*0.5
    return np.clip(r.astype(np.float32)+gc,0,255).astype(np.uint8)

# Point cloud family
def fx_pc(r,d,step=4,thresh=50,size_div=50):
    if d is None: return r
    h,w=r.shape[:2]; dn=cv2.normalize(d,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    if dn.shape[:2]!=(h,w): dn=cv2.resize(dn,(w,h))
    o=np.zeros_like(r)
    for y in range(0,h-1,step):
        for x in range(0,w-1,step):
            if dn[y,x]>thresh: cv2.circle(o,(x,y),max(1,int((255-dn[y,x])/size_div)),r[y,x].tolist(),-1)
    return o

def fx_pc_base(r,d): return fx_pc(r,d,4,50,50)
def fx_pc_sparse(r,d): return fx_pc(r,d,8,50,40)
def fx_pc_dense(r,d): return fx_pc(r,d,2,40,80)

def fx_pc_color(r,d):
    if d is None: return r
    h,w=r.shape[:2]; dn=cv2.normalize(d,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    if dn.shape[:2]!=(h,w): dn=cv2.resize(dn,(w,h))
    o=np.zeros_like(r)
    for y in range(0,h-1,4):
        for x in range(0,w-1,4):
            if dn[y,x]>50:
                hue=int(dn[y,x]*0.7); col=cv2.cvtColor(np.array([[[hue,255,255]]],dtype=np.uint8),cv2.COLOR_HSV2BGR)[0,0].tolist()
                cv2.circle(o,(x,y),max(1,int((255-dn[y,x])/50)),col,-1)
    return o

def fx_pc_stripes(r,d):
    if d is None: return r
    h,w=r.shape[:2]; dn=cv2.normalize(d,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    if dn.shape[:2]!=(h,w): dn=cv2.resize(dn,(w,h))
    o=np.zeros_like(r)
    for y in range(0,h-2,6):
        for x in range(w):
            if dn[y,x]>50: o[y:min(y+2,h),x]=r[y,x]
    return o

def fx_pc_rain(r,d):
    if d is None: return r
    h,w=r.shape[:2]; dn=cv2.normalize(d,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    if dn.shape[:2]!=(h,w): dn=cv2.resize(dn,(w,h))
    o=np.zeros_like(r)
    for y in range(0,h-1,8):
        for x in range(0,w-2,6):
            if dn[y,x]>50:
                slen=max(1,int((255-dn[y,x])/20)); o[y:min(h,y+slen),x:min(w,x+2)]=r[y,x]
    return o

def fx_pc_scatter(r,d):
    if d is None: return r
    h,w=r.shape[:2]; dn=cv2.normalize(d,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    if dn.shape[:2]!=(h,w): dn=cv2.resize(dn,(w,h))
    o=np.zeros_like(r)
    for y in range(0,h-1,5):
        for x in range(0,w-1,5):
            if dn[y,x]>50:
                rng=np.random.RandomState((y*w+x)%2147483647)
                ox,oy=max(0,min(w-1,x+rng.randint(-3,4))),max(0,min(h-1,y+rng.randint(-3,4)))
                cv2.circle(o,(ox,oy),max(1,int((255-dn[y,x])/50)),r[y,x].tolist(),-1)
    return o

# Displacement family
def fx_disp_wave(r,d):
    if d is None: return r
    h,w=r.shape[:2]; dn=cv2.normalize(d,None,0,1,cv2.NORM_MINMAX).astype(np.float32)
    if dn.shape[:2]!=(h,w): dn=cv2.resize(dn,(w,h))
    mx,my=np.zeros((h,w),np.float32),np.zeros((h,w),np.float32)
    for y in range(h):
        for x in range(w):
            wave=math.sin(y/20+s.fc/10)*dn[y,x]*20; mx[y,x],my[y,x]=x+wave,y
    return cv2.remap(r,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)

def fx_disp_shatter(r,d):
    if d is None: return r
    h,w=r.shape[:2]; dn=cv2.normalize(d,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    if dn.shape[:2]!=(h,w): dn=cv2.resize(dn,(w,h))
    o=r.copy(); blk=20
    for by in range(0,h,blk):
        for bx in range(0,w,blk):
            bh,bw=min(blk,h-by),min(blk,w-bx)
            bd=dn[by:by+bh,bx:bx+bw]
            if bd.size>0:
                avg=np.mean(bd); rng=np.random.RandomState(int(by*w+bx)%2147483647)
                ox=int((avg/255-0.5)*40)+rng.randint(-15,16); oy=int((avg/255-0.5)*20)+rng.randint(-10,11)
                sx,sy=max(0,min(w-bw,bx+ox)),max(0,min(h-bh,by+oy))
                o[by:by+bh,bx:bx+bw]=r[sy:sy+bh,sx:sx+bw]
    return o

def fx_disp_ripple(r,d):
    if d is None: return r
    h,w=r.shape[:2]; dn=cv2.normalize(d,None,0,1,cv2.NORM_MINMAX).astype(np.float32)
    if dn.shape[:2]!=(h,w): dn=cv2.resize(dn,(w,h))
    cx,cy=w//2,h//2; mx,my=np.zeros((h,w),np.float32),np.zeros((h,w),np.float32)
    for y in range(h):
        for x in range(w):
            dx,dy=x-cx,y-cy; dist=math.sqrt(dx*dx+dy*dy); ang=math.atan2(dy,dx)
            rip=math.sin(dist/15-s.fc/5)*dn[y,x]*10
            mx[y,x],my[y,x]=x+math.cos(ang)*rip,y+math.sin(ang)*rip
    return cv2.remap(r,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)

def fx_disp_stretch(r,d):
    if d is None: return r
    h,w=r.shape[:2]; dn=cv2.normalize(d,None,0,1,cv2.NORM_MINMAX).astype(np.float32)
    if dn.shape[:2]!=(h,w): dn=cv2.resize(dn,(w,h))
    mx,my=np.zeros((h,w),np.float32),np.zeros((h,w),np.float32)
    for y in range(h):
        for x in range(w): mx[y,x],my[y,x]=x,y+(dn[y,x]-0.5)*40
    return cv2.remap(r,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)

def fx_disp_twist(r,d):
    if d is None: return r
    h,w=r.shape[:2]; dn=cv2.normalize(d,None,0,1,cv2.NORM_MINMAX).astype(np.float32)
    if dn.shape[:2]!=(h,w): dn=cv2.resize(dn,(w,h))
    cx,cy=w//2,h//2; maxd=math.sqrt(cx*cx+cy*cy)
    mx,my=np.zeros((h,w),np.float32),np.zeros((h,w),np.float32)
    for y in range(h):
        for x in range(w):
            dx,dy=x-cx,y-cy; dist=math.sqrt(dx*dx+dy*dy); ang=math.atan2(dy,dx)
            fall=max(0,1-dist/maxd); tw=dn[y,x]*0.5*fall; na=ang+tw
            mx[y,x],my[y,x]=cx+dist*math.cos(na),cy+dist*math.sin(na)
    return cv2.remap(r,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)

# Contour family
def fx_cont(r,d):
    if d is None: return r
    dn=cv2.normalize(d,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    e=cv2.Canny(dn,30,100); o=np.zeros_like(r); o[:,:,1]=e; o[:,:,2]=e//2; return o

def fx_cont_thick(r,d):
    if d is None: return r
    s.contour_phase=(s.contour_phase+0.05)%(2*math.pi)
    dn=cv2.normalize(d,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    e1,e2,e3=cv2.Canny(dn,20,80),cv2.Canny(dn,40,120),cv2.Canny(dn,60,160)
    e=cv2.dilate(cv2.bitwise_or(cv2.bitwise_or(e1,e2),e3),np.ones((4,4),np.uint8),iterations=2)
    hue=max(0,min(90,int((math.sin(s.contour_phase)+1)*45)+30))
    o=np.zeros_like(r)
    o[:,:,0]=(e.astype(np.int32)*hue//255).astype(np.uint8); o[:,:,1]=e
    o[:,:,2]=(e.astype(np.int32)*max(0,90-hue)//90).astype(np.uint8)
    return o

def fx_cont_filled(r,d):
    if d is None: return r
    dn=cv2.normalize(d,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    q=(dn//32)*32; col=cv2.applyColorMap(q,cv2.COLORMAP_VIRIDIS)
    e=cv2.dilate(cv2.Canny(dn,30,100),np.ones((2,2),np.uint8),iterations=1)
    col[e>0]=[255,255,255]; return col

# Other
def fx_portal(r,d):
    if d is None: return r
    dn=cv2.normalize(d,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    m=cv2.inRange(dn,80,160); p=np.zeros_like(r); p[:,:]=150,40,20
    m3=cv2.cvtColor(m,cv2.COLOR_GRAY2BGR).astype(np.float32)/255.0
    o=(r.astype(np.float32)*(1-m3)+p.astype(np.float32)*m3).astype(np.uint8)
    e=cv2.dilate(cv2.Canny(m,50,150),np.ones((3,3),np.uint8),iterations=1); o[e>0]=[255,150,50]
    return o

def fx_parallax(r,d):
    if d is None: return r
    h,w=r.shape[:2]; dn=cv2.normalize(d,None,0,1,cv2.NORM_MINMAX).astype(np.float32)
    if dn.shape[:2]!=(h,w): dn=cv2.resize(dn,(w,h))
    hx=math.sin(s.fc/60)*8
    mx,my=np.zeros((h,w),np.float32),np.zeros((h,w),np.float32)
    for y in range(h):
        for x in range(w): mx[y,x],my[y,x]=x+hx*(1-dn[y,x]),y
    return cv2.remap(r,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)

def fx_selective(r,d):
    if d is None: return r
    h,w=r.shape[:2]; dn=cv2.normalize(d,None,0,1,cv2.NORM_MINMAX).astype(np.float32)
    if dn.shape[:2]!=(h,w): dn=cv2.resize(dn,(w,h))
    rz=np.exp(-((dn-0.4)**2)/0.05).reshape(h,w,1)
    weird=cv2.GaussianBlur(cv2.cvtColor(cv2.cvtColor(r,cv2.COLOR_BGR2HSV),cv2.COLOR_HSV2BGR),(5,5),0)
    return (r.astype(np.float32)*rz+weird.astype(np.float32)*(1-rz)).astype(np.uint8)

def fx_shadow(r,d):
    if d is None: return r
    h,w=r.shape[:2]; dn=cv2.normalize(d,None,0,1,cv2.NORM_MINMAX).astype(np.float32)
    if dn.shape[:2]!=(h,w): dn=cv2.resize(dn,(w,h))
    sh=np.roll(dn,15,axis=0); sh[:15,:]=dn[:15,:]
    sm=cv2.GaussianBlur((sh<dn-0.1).astype(np.float32),(11,11),0).reshape(h,w,1)
    return (r.astype(np.float32)*(1-sm*0.5)).astype(np.uint8)

def fx_dpixel(r,d):
    if d is None: return r
    h,w=r.shape[:2]; dn=cv2.normalize(d,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    if dn.shape[:2]!=(h,w): dn=cv2.resize(dn,(w,h))
    o=r.copy(); blk=8
    for by in range(0,h,blk):
        for bx in range(0,w,blk):
            bh,bw=min(blk,h-by),min(blk,w-bx)
            if np.mean(dn[by:by+bh,bx:bx+bw])>80:
                o[by:by+bh,bx:bx+bw]=np.mean(r[by:by+bh,bx:bx+bw],axis=(0,1))
    return o

# Stereo effects - use left/right cameras
def get_stereo_frames(r):
    """Get left/right frames, cropped and resized to match output"""
    if s._left is None or s._right is None: return None, None
    h,w=r.shape[:2]
    left = crop(s._left, s.zoom) if s.zoom > 1 else s._left
    right = crop(s._right, s.zoom) if s.zoom > 1 else s._right
    left = cv2.resize(left,(w,h)) if left.shape[:2]!=(h,w) else left
    right = cv2.resize(right,(w,h)) if right.shape[:2]!=(h,w) else right
    return left, right

def apply_line_thickness(edges):
    """Apply line thickness based on s.line_weight"""
    if s.line_weight > 1:
        return cv2.dilate(edges, np.ones((s.line_weight, s.line_weight), np.uint8), iterations=1)
    return edges

def stereo_blend_rgb(effect, r):
    """Blend stereo effect with normal RGB based on s.blend"""
    if s.blend > 0:
        return cv2.addWeighted(effect, 1-s.blend, r, s.blend, 0)
    return effect

def fx_stereo_edge(r,d):
    """Green left edge + Blue right edge combined"""
    left, right = get_stereo_frames(r)
    if left is None: return r
    h,w=r.shape[:2]
    el = apply_line_thickness(cv2.Canny(left,50,150))
    er = apply_line_thickness(cv2.Canny(right,50,150))
    o = np.zeros((h,w,3),dtype=np.uint8)
    if s.hue_shift == 0:  # default green/blue
        o[:,:,1] = el  # green = left
        o[:,:,0] = er  # blue = right
    else:  # apply hue shift via HSV
        o[:,:,1] = el
        o[:,:,0] = er
        hsv = cv2.cvtColor(o, cv2.COLOR_BGR2HSV)
        hsv[:,:,0] = (hsv[:,:,0].astype(np.int16) + s.hue_shift) % 180
        o = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return stereo_blend_rgb(o, r)

def fx_stereo_anaglyph(r,d):
    """Classic red/cyan anaglyph from stereo pair"""
    left, right = get_stereo_frames(r)
    if left is None: return r
    h,w=r.shape[:2]
    o = np.zeros((h,w,3),dtype=np.uint8)
    o[:,:,2] = left   # red = left
    o[:,:,1] = right  # green = right (cyan-ish)
    o[:,:,0] = right  # blue = right
    return stereo_blend_rgb(o, r)

def fx_stereo_diff(r,d):
    """Difference between left and right - shows depth displacement"""
    left, right = get_stereo_frames(r)
    if left is None: return r
    diff = cv2.absdiff(left, right)
    o = cv2.applyColorMap(diff, cv2.COLORMAP_VIRIDIS)
    return stereo_blend_rgb(o, r)

def fx_stereo_blend(r,d):
    """Blend left green + right blue with RGB overlay"""
    left, right = get_stereo_frames(r)
    if left is None: return r
    h,w=r.shape[:2]
    o = np.zeros((h,w,3),dtype=np.uint8)
    o[:,:,1] = left   # green = left
    o[:,:,0] = right  # blue = right
    return cv2.addWeighted(o, 1-s.blend, r, s.blend, 0) if s.blend > 0 else o

# Combos
def combo(base):
    def c(r,d): return base(fx_break(r,d),d) if d else fx_break(r,d)
    return c

FX={
    '1':("Normal",fx_norm),'2':("Edge",fx_edge),'i':("Invert",fx_inv),'6':("Thermal",fx_therm),
    'v':("VASARI",fx_vasari),'b':("Breakup",fx_break),
    '3':("Focus",fx_dfocus),'4':("Atmos",fx_atmos),'5':("D-Inv",fx_dinv),'8':("Lag",fx_dlag),
    'c':("Grain",fx_dgrain),'o':("Glow",fx_dglow),
    '9':("Points",fx_pc_base),'w':("Sparse",fx_pc_sparse),'e':("Dense",fx_pc_dense),
    'r':("Rainbow",fx_pc_color),'t':("Stripes",fx_pc_stripes),'y':("Rain",fx_pc_rain),'u':("Scatter",fx_pc_scatter),
    'a':("Wave",fx_disp_wave),'s':("Shatter",fx_disp_shatter),
    '7':("Contour",fx_cont),'z':("Thick",fx_cont_thick),'x':("Filled",fx_cont_filled),
    '0':("Portal",fx_portal),
    # Stereo (n=main, variants use shift numbers)
    'n':("StereoEdge",fx_stereo_edge),',':("Anaglyph",fx_stereo_anaglyph),
    '.':("StereoDiff",fx_stereo_diff),'/':("StereoBlend",fx_stereo_blend),
    # Combos
    'd':("V+Ripple",combo(fx_disp_ripple)),'f':("V+Stretch",combo(fx_disp_stretch)),
    'g':("V+Twist",combo(fx_disp_twist)),'h':("V+Pixel",combo(fx_dpixel)),
    'j':("V+Parallax",combo(fx_parallax)),'k':("V+Select",combo(fx_selective)),'l':("V+Shadow",combo(fx_shadow)),
}

class Api:
    def st(self):
        return {'m':s.name,'k':s.mode,'p':s.puppet,'i':round(s.intensity,2),'sp':round(s.speed,2),
                'f':round(s.facepop,2),'bl':round(s.blend,2),'z':round(s.zoom,1),
                'dt':round(s.d_thresh,2),'ds':round(s.d_soft,0),'lg':round(s.lag_int,2),'lo':s.logo,
                'inv':s.invert,'mir':s.mirror,'lw':s.line_weight,'hu':s.hue_shift}
    def mode(self,k):
        k=k.lower()
        if k in FX:
            if s.mode=='8' and k!='8': s._lag=None
            s.mode,s.name=k,FX[k][0]
        return self.st()
    def pup(self):
        s.puppet=not s.puppet
        s.v_start=s.b_start=s.fc if s.puppet else s.v_start
        s.fuzz_frames = 20  # trigger universal fuzz burst
        return self.st()
    def si(self,v): s.intensity=max(0.2,min(3.0,float(v))); return self.st()
    def ss(self,v): s.speed=max(0,min(3.0,float(v))); return self.st()
    def sf(self,v): s.facepop=max(0,min(1.0,float(v))); return self.st()
    def sb(self,v): s.blend=max(0,min(1.0,float(v))); return self.st()
    def sz(self,v): s.zoom=max(1.0,min(5.0,float(v))); return self.st()
    def sdt(self,v): s.d_thresh=max(0.1,min(0.9,float(v))); return self.st()
    def sds(self,v): s.d_soft=max(1.0,min(20.0,float(v))); return self.st()
    def slg(self,v): s.lag_int=max(0.5,min(0.99,float(v))); return self.st()
    def slw(self,v): s.line_weight=max(1,min(10,int(v))); return self.st()
    def shu(self,v): s.hue_shift=max(0,min(180,int(v))); return self.st()
    def tlo(self): s.logo=not s.logo; return self.st()
    def tinv(self): s.invert=not s.invert; return self.st()
    def tmir(self): s.mirror=not s.mirror; return self.st()
    def reset(self):
        """Reset to normal - keeps zoom"""
        z = s.zoom  # preserve zoom
        s.mode='1'; s.name='Normal'; s.intensity=1.0; s.speed=0.0; s.facepop=0.3
        s.blend=0.0; s.d_thresh=0.35; s.d_soft=8.0; s.lag_int=0.95
        s.line_weight=2; s.hue_shift=0; s.invert=False; s.mirror=False
        s.puppet=False; s.zoom=z
        return self.st()
    def fr(self):
        if s.frame is not None:
            _,b=cv2.imencode('.jpg',s.frame,[cv2.IMWRITE_JPEG_QUALITY,85])
            return base64.b64encode(b).decode('utf-8')
        return None

def crop(img,z=1.0):
    if z<=1.0: return img
    h,w=img.shape[:2]; ch,cw=int(h/z),int(w/z); y1,x1=(h-ch)//2,(w-cw)//2
    c=img[y1:y1+ch,x1:x1+cw]; return cv2.resize(c,(w,h)) if c.size>0 else img

def demo_loop(logo):
    """Demo mode - generate test pattern when no camera"""
    while s.run:
        s.fc+=1
        rgb = np.zeros((H,W,3), dtype=np.uint8)
        rgb[:,:,0] = 40; rgb[:,:,1] = 30; rgb[:,:,2] = 50
        cv2.putText(rgb, "NO CAMERA", (W//2-100, H//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100,100,100), 2)
        cv2.putText(rgb, f"Frame: {s.fc}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,80,80), 1)
        _,fx_fn=FX.get(s.mode,("Normal",fx_norm))
        try: o=fx_fn(rgb,None)
        except: o=rgb.copy()
        if o is None: o=rgb.copy()
        if s.mirror: o=cv2.flip(o,1)
        if s.invert: o=cv2.bitwise_not(o)
        o=np.ascontiguousarray(o,dtype=np.uint8)
        if s.logo and logo is not None:
            lh,lw=logo.shape[:2]; xo,yo=W-lw-10,10
            if logo.shape[2]==4:
                a=logo[:,:,3]/255.0
                for ch in range(3): o[yo:yo+lh,xo:xo+lw,ch]=(1-a)*o[yo:yo+lh,xo:xo+lw,ch]+a*logo[:,:,ch]
            else: o[yo:yo+lh,xo:xo+lw]=logo[:,:,:3]
        s.frame=o.copy()
        time.sleep(1/30)

def cam(logo):
    if not HAS_CAM:
        demo_loop(logo); return
    try:
        p=dai.Pipeline()
        c=p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        co=c.requestOutput((W,H),dai.ImgFrame.Type.BGR888p)
        qr=co.createOutputQueue(maxSize=2,blocking=False)
        l=p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        lo=l.requestOutput((640,480))
        ql=lo.createOutputQueue(maxSize=2,blocking=False)  # left cam queue
        r=p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        ro=r.requestOutput((640,480))
        qright=ro.createOutputQueue(maxSize=2,blocking=False)  # right cam queue
        st=p.create(dai.node.StereoDepth); lo.link(st.left); ro.link(st.right)
        qd=st.depth.createOutputQueue(maxSize=2,blocking=False)
        p.start()
    except Exception as e:
        print(f"Camera init failed: {e}, running demo mode")
        demo_loop(logo); return
    try:
        with pyvirtualcam.Camera(width=W,height=H,fps=FPS) as vc:
            print(f"VCam:{vc.device}")
            while s.run and p.isRunning():
                s.fc+=1; rm=qr.tryGet()
                if rm is None: time.sleep(0.001); continue
                rgb=crop(rm.getCvFrame(),s.zoom); d=None; dm=qd.tryGet()
                if dm: d=dm.getFrame(); d=crop(d,s.zoom); d=cv2.resize(d,(W,H)) if d.shape[:2]!=(H,W) else d; s._depth=d
                elif s._depth is not None: d=s._depth
                # Get left/right stereo frames
                lm=ql.tryGet()
                if lm: s._left=lm.getCvFrame()
                rm2=qright.tryGet()
                if rm2: s._right=rm2.getCvFrame()
                _,fx_fn=FX.get(s.mode,("Normal",fx_norm))
                try: o=fx_fn(rgb,d)
                except: o=rgb.copy()
                if o is None: o=rgb.copy()
                # Universal fuzz on spacebar (works in any mode)
                if s.fuzz_frames > 0:
                    s.fuzz_frames -= 1
                    fint = min(1.0, s.fuzz_frames / 15) * s.intensity
                    h,w = o.shape[:2]
                    # Band shifts
                    for _ in range(random.randint(3, 12)):
                        y = random.randint(0, h-2)
                        bh = min(random.randint(2, 30), h-y)
                        if bh > 0:
                            o[y:y+bh,:] = np.roll(o[y:y+bh,:], random.randint(-int(40*fint), int(40*fint)), axis=1)
                    # Noise
                    noise = np.random.randint(-int(30*fint), int(30*fint)+1, o.shape, dtype=np.int16)
                    o = np.clip(o.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                if s.mirror: o=cv2.flip(o,1)
                if s.invert: o=cv2.bitwise_not(o)
                o=np.ascontiguousarray(o,dtype=np.uint8)
                if s.logo and logo is not None:
                    lh,lw=logo.shape[:2]; xo,yo=W-lw-10,10
                    if logo.shape[2]==4:
                        a=logo[:,:,3]/255.0
                        for ch in range(3): o[yo:yo+lh,xo:xo+lw,ch]=(1-a)*o[yo:yo+lh,xo:xo+lw,ch]+a*logo[:,:,ch]
                    else: o[yo:yo+lh,xo:xo+lw]=logo[:,:,:3]
                s.frame=o.copy(); vc.send(cv2.cvtColor(o,cv2.COLOR_BGR2RGB))
    except Exception as e: print(f"Err:{e}")
    finally: p.stop()

HTML="""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>VASARI</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font:11px/1.3 system-ui;background:#0a0a0a;color:#ccc;overflow:hidden}
.wrap{display:flex;flex-direction:column;height:100vh}
.main{display:flex;flex:1;min-height:0}
.vidwrap{position:relative;flex:1;display:flex;align-items:center;justify-content:center;background:#000;overflow:hidden}
.aspect{position:relative;width:100%;height:100%;max-width:calc((100vh - 70px) * 4 / 3);max-height:calc((100vw - 160px) * 3 / 4)}
#vid{width:100%;height:100%;object-fit:contain}
.status{position:absolute;top:8px;left:8px;display:flex;gap:6px;z-index:10}
.st{font-size:10px;background:#234;padding:2px 8px;border-radius:3px;color:#fff;font-weight:600}
.pup{background:#a22;animation:p .5s infinite}@keyframes p{50%{opacity:.6}}
.side{width:160px;background:#111;border-left:1px solid #222;padding:8px;display:flex;flex-direction:column;gap:8px;overflow-y:auto}
.stitle{font-size:8px;color:#567;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px}
.param{display:flex;flex-direction:column;gap:1px;margin-bottom:6px}
.param .row{display:flex;align-items:center;gap:4px}
.param label{width:55px;font-size:9px;color:#aaa;font-weight:600}
.param .ctrl{display:flex;align-items:center;gap:2px}
.param input[type=text]{width:45px;background:#1a1a1a;border:1px solid #333;color:#fff;padding:3px 4px;font-size:10px;text-align:center;font-family:monospace}
.param input[type=text]:focus{outline:none;border-color:#567}
.arr{background:#222;border:1px solid #333;color:#888;width:18px;height:20px;cursor:pointer;font-size:11px;display:flex;align-items:center;justify-content:center;user-select:none}
.arr:hover{background:#345;color:#fff;border-color:#567}
.arr:active{background:#456}
.param .key{font-size:8px;color:#567;text-align:center;margin-top:1px;font-weight:600}
.checks{display:flex;flex-direction:column;gap:4px}
.chk{display:flex;align-items:center;gap:6px;cursor:pointer}
.chk input{accent-color:#567;width:12px;height:12px}
.chk span{font-size:9px;color:#888}
.chk .key{font-size:8px;color:#567;font-weight:600;margin-left:auto}
.bot{background:#111;border-top:1px solid #222;padding:6px 8px}
.modes{display:flex;flex-wrap:wrap;gap:3px;margin-bottom:4px}
.m{background:#1a1a1a;border:1px solid #333;border-radius:3px;padding:4px 8px;cursor:pointer;font-size:9px;font-weight:600;transition:all .1s}
.m:hover{border-color:#567;color:#fff}
.m.on{background:#234;border-color:#567;color:#fff}
.vars{display:flex;flex-wrap:wrap;gap:3px;min-height:14px}
.vars a{background:#1a1a1a;border:1px solid #333;border-radius:3px;padding:4px 8px;cursor:pointer;font-size:9px;font-weight:600;color:#888;text-decoration:none;transition:all .1s}
.vars a:hover{border-color:#567;color:#fff}
.vars a.on{background:#234;border-color:#567;color:#fff}
.pbutton{background:#1a1a1a;border:1px solid #333;border-radius:3px;padding:4px 12px;cursor:pointer;font-size:10px;font-weight:600;color:#888;margin-left:auto}
.pbutton:hover{border-color:#567;color:#fff}
.pbutton.on{background:#822;border-color:#a44;color:#fff}
.botrow{display:flex;align-items:center;gap:8px}
.note{font-size:8px;color:#444;margin-top:6px;line-height:1.4}
</style></head><body>
<div class="wrap">
<div class="main">
<div class="vidwrap">
<div class="status"><span class="st" id="ms">VASARI</span><span class="st pup" id="ps" style="display:none">PUPPET</span></div>
<div class="aspect"><img id="vid"></div>
</div>
<div class="side">
<div>
<div class="stitle">Adjustments</div>
<div class="param"><div class="row"><label>Intensity</label><div class="ctrl"><span class="arr dn" data-p="i">-</span><input type="text" id="vi" value="1.00"><span class="arr up" data-p="i">+</span></div></div><div class="key">Q / A</div></div>
<div class="param"><div class="row"><label>Speed</label><div class="ctrl"><span class="arr dn" data-p="sp">-</span><input type="text" id="vs" value="0.00"><span class="arr up" data-p="sp">+</span></div></div><div class="key">W / S</div></div>
<div class="param"><div class="row"><label>FacePop</label><div class="ctrl"><span class="arr dn" data-p="f">-</span><input type="text" id="vf" value="0.30"><span class="arr up" data-p="f">+</span></div></div><div class="key">E / D</div></div>
<div class="param"><div class="row"><label>Blend</label><div class="ctrl"><span class="arr dn" data-p="bl">-</span><input type="text" id="vb" value="0.00"><span class="arr up" data-p="bl">+</span></div></div><div class="key">R / F</div></div>
<div class="param"><div class="row"><label>Zoom</label><div class="ctrl"><span class="arr dn" data-p="z">-</span><input type="text" id="vz" value="1.0"><span class="arr up" data-p="z">+</span></div></div><div class="key">T / G</div></div>
</div>
<div>
<div class="stitle">Depth</div>
<div class="param"><div class="row"><label>Thresh</label><div class="ctrl"><span class="arr dn" data-p="dt">-</span><input type="text" id="vdt" value="0.35"><span class="arr up" data-p="dt">+</span></div></div></div>
<div class="param"><div class="row"><label>Soft</label><div class="ctrl"><span class="arr dn" data-p="ds">-</span><input type="text" id="vds" value="8"><span class="arr up" data-p="ds">+</span></div></div></div>
<div class="param"><div class="row"><label>Lag</label><div class="ctrl"><span class="arr dn" data-p="lg">-</span><input type="text" id="vlg" value="0.95"><span class="arr up" data-p="lg">+</span></div></div></div>
</div>
<div>
<div class="stitle">Lines/Stereo</div>
<div class="param"><div class="row"><label>Thickness</label><div class="ctrl"><span class="arr dn" data-p="lw">-</span><input type="text" id="vlw" value="2"><span class="arr up" data-p="lw">+</span></div></div><div class="key">Y / H</div></div>
<div class="param"><div class="row"><label>Hue</label><div class="ctrl"><span class="arr dn" data-p="hu">-</span><input type="text" id="vhu" value="0"><span class="arr up" data-p="hu">+</span></div></div><div class="key">U / J</div></div>
</div>
<div>
<div class="stitle">Overlays (stack with any mode)</div>
<div class="checks">
<label class="chk"><input type="checkbox" id="clo" checked><span>Logo</span><span class="key">L</span></label>
<label class="chk"><input type="checkbox" id="cinv"><span>Invert</span><span class="key">I</span></label>
<label class="chk"><input type="checkbox" id="cmir"><span>Mirror</span><span class="key">M</span></label>
</div>
</div>
<div class="note">ESC=reset. OAK-D required: DEPTH, POINTS, DISPLACE, CONTOUR, PORTAL, STEREO, COMBOS</div>
</div>
</div>
<div class="bot">
<div class="botrow">
<div class="modes">
<span class="m" data-mode="NORMAL">NORMAL</span>
<span class="m on" data-mode="VASARI">VASARI</span>
<span class="m" data-mode="DEPTH">DEPTH</span>
<span class="m" data-mode="POINTS">POINTS</span>
<span class="m" data-mode="DISPLACE">DISPLACE</span>
<span class="m" data-mode="CONTOUR">CONTOUR</span>
<span class="m" data-mode="STEREO">STEREO</span>
<span class="m" data-mode="EDGE">EDGE</span>
<span class="m" data-mode="THERMAL">THERMAL</span>
<span class="m" data-mode="PORTAL">PORTAL</span>
<span class="m" data-mode="COMBOS">COMBOS</span>
</div>
<button class="pbutton" id="pb">PUPPET [SPACE]</button>
</div>
<div class="vars" id="vars"><a data-k="b">Breakup</a></div>
</div>
</div>
<script>
let a=null;
const MODES={
NORMAL:{key:'1',vars:[]},
VASARI:{key:'v',vars:[['b','Breakup']]},
DEPTH:{key:'3',vars:[['4','Atmos'],['5','D-Inv'],['8','Lag'],['c','Grain'],['o','Glow']]},
POINTS:{key:'9',vars:[['w','Sparse'],['e','Dense'],['r','Rainbow'],['t','Stripes'],['y','Rain'],['u','Scatter']]},
DISPLACE:{key:'a',vars:[['s','Shatter']]},
CONTOUR:{key:'7',vars:[['z','Thick'],['x','Filled']]},
STEREO:{key:'n',vars:[[',','Anaglyph'],['.','Diff'],['/','Blend']]},
EDGE:{key:'2',vars:[]},
THERMAL:{key:'6',vars:[]},
PORTAL:{key:'0',vars:[]},
COMBOS:{key:'d',vars:[['f','Stretch'],['g','Twist'],['h','Pixel'],['j','Parallax'],['k','Select'],['l','Shadow']]}
};
const PARAMS={
i:{el:'vi',min:0.2,max:3,step:0.1,fn:'si'},
sp:{el:'vs',min:0,max:3,step:0.1,fn:'ss'},
f:{el:'vf',min:0,max:1,step:0.05,fn:'sf'},
bl:{el:'vb',min:0,max:1,step:0.05,fn:'sb'},
z:{el:'vz',min:1,max:5,step:0.5,fn:'sz'},
dt:{el:'vdt',min:0.1,max:0.9,step:0.05,fn:'sdt'},
ds:{el:'vds',min:1,max:20,step:1,fn:'sds'},
lg:{el:'vlg',min:0.5,max:0.99,step:0.02,fn:'slg'},
lw:{el:'vlw',min:1,max:10,step:1,fn:'slw'},
hu:{el:'vhu',min:0,max:180,step:10,fn:'shu'}
};
let curMode='VASARI';
async function tick(){if(!a){requestAnimationFrame(tick);return}try{const f=await a.fr();if(f)document.getElementById('vid').src='data:image/jpeg;base64,'+f}catch(e){}requestAnimationFrame(tick)}
function getMode(k){for(let m in MODES){if(MODES[m].key===k)return m;for(let v of MODES[m].vars)if(v[0]===k)return m}return null}
function fmt(v,p){return ['ds','lw','hu'].includes(p)?v.toFixed(0):p==='z'?v.toFixed(1):v.toFixed(2)}
function sync(d){
document.getElementById('ms').textContent=d.m;
const cm=getMode(d.k);if(cm){curMode=cm;updateVars()}
document.querySelectorAll('.m').forEach(b=>b.classList.toggle('on',b.dataset.mode===curMode));
document.querySelectorAll('.vars a').forEach(el=>el.classList.toggle('on',el.dataset.k===d.k));
document.getElementById('pb').classList.toggle('on',d.p);
document.getElementById('ps').style.display=d.p?'inline':'none';
document.getElementById('vi').value=fmt(d.i,'i');
document.getElementById('vs').value=fmt(d.sp,'sp');
document.getElementById('vf').value=fmt(d.f,'f');
document.getElementById('vb').value=fmt(d.bl,'bl');
document.getElementById('vz').value=fmt(d.z,'z');
document.getElementById('vdt').value=fmt(d.dt,'dt');
document.getElementById('vds').value=fmt(d.ds,'ds');
document.getElementById('vlg').value=fmt(d.lg,'lg');
document.getElementById('vlw').value=fmt(d.lw,'lw');
document.getElementById('vhu').value=fmt(d.hu,'hu');
document.getElementById('clo').checked=d.lo;
document.getElementById('cinv').checked=d.inv;
document.getElementById('cmir').checked=d.mir;
}
function updateVars(){
const vd=document.getElementById('vars');
const m=MODES[curMode];
if(!m||m.vars.length===0){vd.innerHTML='';return}
vd.innerHTML=m.vars.map(v=>`<a data-k="${v[0]}">${v[1]}</a>`).join('');
document.querySelectorAll('.vars a').forEach(el=>el.onclick=async()=>sync(await a.mode(el.dataset.k)));
}
async function adj(p,dir){
if(!a)return;
const cfg=PARAMS[p];if(!cfg)return;
const cur=parseFloat(document.getElementById(cfg.el).value)||0;
const nv=Math.max(cfg.min,Math.min(cfg.max,cur+dir*cfg.step));
sync(await a[cfg.fn](nv));
}
document.querySelectorAll('.arr').forEach(b=>{
b.onclick=async()=>{
const p=b.dataset.p;
const dir=b.classList.contains('up')?1:-1;
await adj(p,dir);
};
});
Object.keys(PARAMS).forEach(p=>{
const cfg=PARAMS[p];
document.getElementById(cfg.el).onchange=async e=>{
const v=Math.max(cfg.min,Math.min(cfg.max,parseFloat(e.target.value)||cfg.min));
sync(await a[cfg.fn](v));
};
});
document.querySelectorAll('.m').forEach(b=>{
b.onclick=async()=>{
curMode=b.dataset.mode;
updateVars();
sync(await a.mode(MODES[curMode].key));
};
});
document.getElementById('pb').onclick=async()=>sync(await a.pup());
document.getElementById('clo').onchange=async()=>sync(await a.tlo());
document.getElementById('cinv').onchange=async()=>sync(await a.tinv());
document.getElementById('cmir').onchange=async()=>sync(await a.tmir());
document.addEventListener('keydown',async e=>{
if(!a)return;
const k=e.key.toLowerCase();
if(e.target.tagName==='INPUT')return;
if(e.code==='Space'){e.preventDefault();sync(await a.pup());return}
if(e.code==='Escape'||e.code==='Backspace'){e.preventDefault();sync(await a.reset());return}
if(k==='q'){await adj('i',1);return}if(k==='a'){await adj('i',-1);return}
if(k==='w'){await adj('sp',1);return}if(k==='s'){await adj('sp',-1);return}
if(k==='e'){await adj('f',1);return}if(k==='d'){await adj('f',-1);return}
if(k==='r'){await adj('bl',1);return}if(k==='f'){await adj('bl',-1);return}
if(k==='t'){await adj('z',1);return}if(k==='g'){await adj('z',-1);return}
if(k==='y'){await adj('lw',1);return}if(k==='h'){await adj('lw',-1);return}
if(k==='u'){await adj('hu',1);return}if(k==='j'){await adj('hu',-1);return}
if(k==='l'){sync(await a.tlo());return}
if(k==='i'){sync(await a.tinv());return}
if(k==='m'){sync(await a.tmir());return}
if('1234567890vbcon,./'.includes(k)&&!e.metaKey&&!e.ctrlKey){sync(await a.mode(k))}
});
function init(){
a=window.pywebview.api;
tick();
}
if(window.pywebview&&window.pywebview.api){init()}
else{window.addEventListener('pywebviewready',init)}
</script></body></html>"""

def main():
    logo=None
    try:
        logo=cv2.imread("/Users/corbettgriffith/projects/vasari-webcam/logo.png",cv2.IMREAD_UNCHANGED)
        if logo is not None: logo=cv2.resize(logo,None,fx=40/logo.shape[0],fy=40/logo.shape[0])
    except: pass
    threading.Thread(target=cam,args=(logo,),daemon=True).start()
    api=Api()
    w=webview.create_window('VASARI',html=HTML,js_api=api,width=900,height=550,resizable=True,min_size=(700,400))
    w.events.closed+=lambda:setattr(s,'run',False)
    webview.start()
    s.run=False

if __name__=="__main__": main()
