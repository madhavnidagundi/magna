import torch
import timm
from torchvision import transforms
from PIL import Image
import os
import urllib.request
import json

url = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
urllib.request.urlretrieve(url, "imagenet_class_index.json")

with open("imagenet_class_index.json", "r") as f:
    class_idx = json.load(f)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
device = torch.device('cuda')

m = timm.create_model('mobilenetv4_conv_medium.e500_r256_in1k', pretrained=False)
m.load_state_dict(torch.load('/home/nvidia/magna/models/mobilenetv4_medium.pth'))
m.eval()
m.to(device)

print("eval image 4") # It should be ID 822 (val_labels.txt)
img = Image.open('/home/nvidia/magna/imagenet_val/images/ILSVRC2012_val_00000004.JPEG').convert('RGB')
t = transform(img).unsqueeze(0).to(device)
with torch.no_grad():
    out = m(t)
    pred = out.argmax(-1).item()
    print("Pred ID:", pred, "->", class_idx[str(pred)])
