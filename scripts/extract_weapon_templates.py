import cv2
import easyocr
import numpy as np
import os
import glob

v = easyocr.Reader(['en'], gpu=False, verbose=False)
killfeed_dir = 'data/frames/killfeed/'
out_dir = 'data/annotations/weapons_unlabeled/'
os.makedirs(out_dir, exist_ok=True)

saved_templates = []

def is_different(crop):
    if crop.size == 0: return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    for t in saved_templates:
        if t.shape == gray.shape:
            # check difference
            diff = cv2.absdiff(t, gray)
            if np.mean(diff) < 20: # quite similar
                return False
        else:
            # resize and check
            t_res = cv2.resize(t, (gray.shape[1], gray.shape[0]))
            diff = cv2.absdiff(t_res, gray)
            if np.mean(diff) < 20:
                return False
    return True

files = glob.glob(killfeed_dir + '*.png')
count = 0
for f in files[:200]: # check first 200 frames
    img = cv2.imread(f)
    if img is None: continue
    
    # Check if there's any bright pixels (a kill feed entry)
    if img.max() < 100: continue
    
    results = v.readtext(img)
    lines = []
    current_y = -100
    current_line = []
    for bbox, text, conf in sorted(results, key=lambda r: r[0][0][1]):
        if conf < 0.3: continue
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        if abs(y_center - current_y) > 20:
            if current_line: lines.append(current_line)
            current_line = [(bbox, text)]
            current_y = y_center
        else:
            current_line.append((bbox, text))
    if current_line: lines.append(current_line)
    
    for line in lines:
        if len(line) >= 2:
            # assume killer and victim are first and last
            box1 = line[0][0]
            box2 = line[-1][0]
            
            # x2 of first word
            x1 = int(max(p[0] for p in box1))
            # x1 of last word
            x2 = int(min(p[0] for p in box2))
            
            y1 = int(min(p[1] for p in box1) - 5)
            y2 = int(max(p[1] for p in box1) + 5)
            
            if x2 > x1 + 10: # gap exists
                crop = img[max(0, y1):min(img.shape[0], y2), x1:x2]
                if crop.size > 0 and crop.shape[1] > 20:
                    if is_different(crop):
                        cv2.imwrite(os.path.join(out_dir, f'weapon_{count}.png'), crop)
                        os.system(f"cp {os.path.join(out_dir, f'weapon_{count}.png')} /Users/mac/.gemini/antigravity/brain/067a86bf-6692-4712-96de-6f77bd24c41f/weapon_{count}.png")
                        saved_templates.append(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
                        count += 1
                        print(f"Saved weapon {count}")

