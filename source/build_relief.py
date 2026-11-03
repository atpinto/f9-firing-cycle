# Builds the F9 outer-body relief data from two side-on product photos (right side = 1.png, left side = 4.png).
import sys, io, base64, json, numpy as np
from PIL import Image, ImageFilter
sys.path.insert(0, '/private/tmp/claude-502/-Users-ateixeirapinto/83aa92c5-1878-4a2b-8e6c-d4c4690e93c9/scratchpad')
from seg import segment
IMG = '/private/tmp/claude-502/-Users-ateixeirapinto/83aa92c5-1878-4a2b-8e6c-d4c4690e93c9/images/'
OUT = '/private/tmp/claude-502/-Users-ateixeirapinto/83aa92c5-1878-4a2b-8e6c-d4c4690e93c9/scratchpad/photo/'
S = 0.0259          # cm per px of the right-side photo
BREECH_PX, AXIS_PY = 397, 225   # breech face (rear edge of the ejection port) and bore axis
SEAM_PY, SLIDE_REAR_PX = 256, 118
ZMAX = 1.7

imR, aR, mR = segment(IMG + '1.png')
imLf, aLf, mLf = segment(IMG + '4.png', flip=True)
H, W = mR.shape
# align the (flipped) left photo onto the right photo's pixel grid: beavertail tip and slide front, optic top and pad bottom
sx, sy = (733) / (736), (646) / (648)
ys, xs = np.mgrid[0:H, 0:W]
x4 = np.clip(np.round(52 + (xs - 40) * sx).astype(int), 0, aLf.shape[1] - 1)
y4 = np.clip(np.round(17 + (ys - 62) * sy).astype(int), 0, aLf.shape[0] - 1)
aL = aLf[y4, x4]; mL = mLf[y4, x4]
lw = (aL.min(2) > 215) & (aL.max(2) - aL.min(2) < 25)  # backdrop showing through gaps in the left photo
aL = np.where((mL & mR & ~lw)[..., None], aL, aR)   # where the silhouettes disagree, borrow the right side's colour

def box(x0, x1, y0, y1): return (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
def erode(m):
    e = m.copy(); e[1:] &= m[:-1]; e[:-1] &= m[1:]; e[:, 1:] &= m[:, :-1]; e[:, :-1] &= m[:, 1:]; return e
def dist(m, n=40):
    d = np.zeros(m.shape, float); cur = m.copy()
    for _ in range(n): d += cur; cur = erode(cur)
    return d
gun = erode(mR)
# the trigger blade is animated in 3D, so cut the photo's (dark) trigger out of the frame
gun &= ~((aR.min(2) > 225) & (aR.max(2) - aR.min(2) < 18))  # small specks of backdrop (gaps between light and frame)
gun &= ~box(372, 452, 343, 431)   # the opening of the trigger guard holds nothing but the trigger and its halo
slide = gun & (ys < SEAM_PY) & (xs >= SLIDE_REAR_PX)
frame = gun & ~slide

# half-thickness (cm) by region, then rounded off towards each part's own outline
base_s = np.full((H, W), 1.27)
base_s[box(195, 380, 0, 137)] = 1.38                       # ROMEO2 housing
base_s[box(185, 380, 137, 172)] = 1.22                     # its black mount
base_s[box(0, 195, 0, 175)] = 0.6                          # rear sight
base_s[box(700, 834, 0, 172)] = 0.32                       # front sight
base_f = np.full((H, W), 1.3)
base_f[box(0, 335, 330, 690)] = 1.65                       # grip
base_f[box(0, 120, 250, 345)] = 1.2                        # beavertail
base_f[box(338, 545, 385, 480)] = 0.55                     # trigger guard
base_f[box(505, 834, 318, 460)] = 1.18                     # FOXTROT2 light
base_f[box(0, 834, 676, 806)] = 1.55                       # magazine base pad
def relief(mask, base, rmax=0.45):
    b = np.asarray(Image.fromarray((base * 100).astype(np.uint8)).filter(ImageFilter.BoxBlur(4))).astype(float) / 100
    d = dist(mask) * S
    R = np.minimum(rmax, b * 0.5)
    k = np.clip(d / R, 0, 1)
    return np.where(mask, b * np.sqrt(1 - (1 - k) ** 2), 0)
hs, hf = relief(slide, base_s), relief(frame, base_f, 0.5)

# part ids for hover (G channel): slide 1, optic 2, port cut-out 3 | frame 1, light 2, takedown 4, catch 5, safety 6, magrel 7, pad 8
ids_s = np.where(slide, 1, 0)
ids_s[slide & (box(185, 380, 0, 172))] = 2
ids_s[slide & box(403, 527, 175, 221)] = 3
ids_f = np.where(frame, 1, 0)
ids_f[frame & box(505, 834, 318, 460)] = 2
ids_f[frame & box(462, 500, 278, 314)] = 4
ids_f[frame & box(250, 300, 260, 280)] = 5
ids_f[frame & box(148, 230, 250, 310)] = 6
ids_f[frame & box(286, 330, 405, 450)] = 7
ids_f[frame & box(90, 280, 676, 720)] = 8

def pack(mask, h, ids, name):
    yy, xx = np.nonzero(mask); x0, x1, y0, y1 = xx.min() - 1, xx.max() + 2, yy.min() - 1, yy.max() + 2
    sl = (slice(y0, y1), slice(x0, x1))
    def jpg(a):
        # bleed edge colours outwards so texture filtering never pulls in the white backdrop
        a = a[sl].astype(float); m = mask[sl].copy()
        for _ in range(4):
            acc = np.zeros_like(a); cnt = np.zeros(m.shape)
            for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                sm = np.roll(m, (dy, dx), (0, 1)); sa = np.roll(a, (dy, dx), (0, 1))
                acc += sa * sm[..., None]; cnt += sm
            grow = ~m & (cnt > 0); a[grow] = acc[grow] / cnt[grow][:, None]; m = m | grow
        b = io.BytesIO(); Image.fromarray(a.astype(np.uint8)).save(b, 'JPEG', quality=88); return b.getvalue()
    hm = np.zeros(h[sl].shape + (3,), np.uint8)
    hm[..., 0] = np.where(mask[sl], np.clip(np.round(h[sl] / ZMAX * 254) + 1, 1, 255), 0)
    hm[..., 1] = ids[sl]
    b = io.BytesIO(); Image.fromarray(hm).save(b, 'PNG', optimize=True)
    out = {'R': jpg(aR), 'L': jpg(aL), 'H': b.getvalue()}
    for k, v in out.items(): open(OUT + f'{name}_{k}.' + ('png' if k == 'H' else 'jpg'), 'wb').write(v)
    meta = {'W': int(x1 - x0), 'H': int(y1 - y0), 'x0': int(x0), 'y0': int(y0)}
    return {k: base64.b64encode(v).decode() for k, v in out.items()}, meta
data = {}
for name, m, h, ids in (('slide', slide, hs, ids_s), ('frame', frame, hf, ids_f)):
    enc, meta = pack(m, h, ids, name); data[name] = (enc, meta)
    print(name, meta, {k: len(v) for k, v in enc.items()})
js = ['const F9_PHOTO = {']
for name, (enc, meta) in data.items():
    js.append(f'  {name}: {{ R: "data:image/jpeg;base64,{enc["R"]}", L: "data:image/jpeg;base64,{enc["L"]}", H: "data:image/png;base64,{enc["H"]}", W: {meta["W"]}, H_: {meta["H"]}, x0: {meta["x0"]}, y0: {meta["y0"]} }},')
js.append(f'}};\nconst F9_META = {{ s: {S}, breech: {BREECH_PX}, axis: {AXIS_PY}, ZMAX: {ZMAX} }};')
open(OUT + 'f9_photo.js', 'w').write('\n'.join(js) + '\n')
