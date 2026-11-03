import sys, numpy as np
from PIL import Image
from collections import deque
def segment(path, flip=False):
    im = Image.open(path).convert('RGB')
    if flip: im = im.transpose(Image.FLIP_LEFT_RIGHT)
    a = np.asarray(im).astype(int); H, W, _ = a.shape
    mx, mn = a.max(2), a.min(2)
    white = (mn > 228) & (mx - mn < 18)
    # label white components; big ones are background (outside + trigger-guard opening)
    lab = np.zeros((H, W), int); bg = np.zeros((H, W), bool); n = 0
    for y0 in range(H):
        for x0 in range(W):
            if white[y0, x0] and not lab[y0, x0]:
                n += 1; q = deque([(y0, x0)]); lab[y0, x0] = n; pts = []
                while q:
                    y, x = q.popleft(); pts.append((y, x))
                    for yy, xx in ((y+1,x),(y-1,x),(y,x+1),(y,x-1)):
                        if 0 <= yy < H and 0 <= xx < W and white[yy, xx] and not lab[yy, xx]:
                            lab[yy, xx] = n; q.append((yy, xx))
                if len(pts) > 400 or any(p[0] in (0, H-1) or p[1] in (0, W-1) for p in pts):
                    ys, xs = zip(*pts); bg[list(ys), list(xs)] = True
    m = ~bg
    return im, a, m
if __name__ == '__main__':
    for f, fl in (('1.png', False), ('4.png', True)):
        im, a, m = segment('/private/tmp/claude-502/-Users-ateixeirapinto/83aa92c5-1878-4a2b-8e6c-d4c4690e93c9/images/' + f, fl)
        ys, xs = np.nonzero(m); print(f, 'bbox x', xs.min(), xs.max(), 'y', ys.min(), ys.max())
        Image.fromarray((m * 255).astype(np.uint8)).save('/private/tmp/claude-502/-Users-ateixeirapinto/83aa92c5-1878-4a2b-8e6c-d4c4690e93c9/scratchpad/photo/mask_' + f.split('.')[0] + '.png')
        # row profile: leftmost/rightmost per row every 10 px
        for y in range(ys.min(), ys.max()+1, 20):
            r = np.nonzero(m[y])[0]; print(y, r.min(), r.max(), end=' | ')
        print()
