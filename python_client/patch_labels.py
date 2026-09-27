import json

with open('/home/nvidia/magna/imagenet_class_index.json', 'r') as f:
    orig = json.load(f)

py_dict = "IMAGENET_LABELS = {\n"
for k, v in orig.items():
    py_dict += f'    "{k}": "{v[1]}",\n'
py_dict += "}\n"

with open('/home/nvidia/magna/python_client/camera_inference.py', 'r') as f:
    text = f.read()

old_block = """LABELS_FILE = '/home/nvidia/magna/imagenet_class_index.json'

# Load the class indices to human-readable names
try:
    with open(LABELS_FILE, 'r') as f:
        IMAGENET_LABELS = json.load(f)
except Exception as e:
    print(f"[Warning] Could not load labels file: {e}")
    IMAGENET_LABELS = {}"""

text = text.replace(old_block, py_dict)
text = text.replace("IMAGENET_LABELS[class_idx_str][1]", "IMAGENET_LABELS[class_idx_str]")
text = text.replace("IMAGENET_LABELS[pred_idx_str][1]", "IMAGENET_LABELS[pred_idx_str]")

with open('/home/nvidia/magna/python_client/camera_inference.py', 'w') as f:
    f.write(text)
