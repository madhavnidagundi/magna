import onnxruntime as ort
import numpy as np
from PIL import Image
import json
import pkgutil
import os

def preprocess(image_path: str) -> np.ndarray:
    img  = Image.open(image_path).convert('RGB')
    img  = img.resize((256, 256), Image.BILINEAR)
    w, h = img.size
    l, t = (w - 224) // 2, (h - 224) // 2
    img  = img.crop((l, t, l + 224, t + 224))
    arr  = np.asarray(img, dtype=np.float32) / 255.0
    arr  = np.transpose(arr, (2, 0, 1))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    arr  = (arr - mean) / std
    return np.ascontiguousarray(arr[np.newaxis], dtype=np.float32)

sess = ort.InferenceSession("models/efficientnet_v2_s.onnx")

real_bytes = pkgutil.get_data('timm.data', os.path.join('_info', 'imagenet_real_labels.json'))
real_list  = json.loads(real_bytes.decode('utf-8'))

for i in range(10):
    fname = f'ILSVRC2012_val_{i + 1:08d}.JPEG'
    fpath = os.path.join("imagenet_val/images", fname)
    if not os.path.exists(fpath):
        print(f"{fname} not found!")
        continue
    
    img_data = preprocess(fpath)
    outputs = sess.run(None, {"x": img_data})
    logits = outputs[0][0]
    
    top5 = np.argsort(logits)[-5:][::-1]
    lbl_set = real_list[i]
    print(f"{fname}: Predicted Top-5: {top5}, Ground Truth ReaL: {lbl_set}")
