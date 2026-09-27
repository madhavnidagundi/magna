import grpc
import sys
import numpy as np
import cv2
from pathlib import Path

import magna_pb2
import magna_pb2_grpc

SERVER_ADDRESS = 'localhost:50051'

def preprocess(image_path):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = 256 / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    start_h = (new_h - 224) // 2
    start_w = (new_w - 224) // 2
    img = img[start_h:start_h+224, start_w:start_w+224]
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    return np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)  # [1,3,224,224]

def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not img_path:
        all_imgs = sorted(Path("/opt/imagenet1k_val").rglob("*.jpg"))
        img_path = str(all_imgs[0])

    print(f"[Client] Using image: {img_path}")
    img_data = preprocess(img_path).astype(np.float32)

    channel = grpc.insecure_channel(SERVER_ADDRESS)
    stub = magna_pb2_grpc.InferenceServiceStub(channel)

    health = stub.HealthCheck(magna_pb2.HealthRequest(), timeout=2.0)
    print(f"[Client] Server ready. Backend: {health.backend}, Model: {health.model_name}")

    input_tensor = magna_pb2.InferInputTensor(
        name="input",
        datatype="FP32",
        shape=[1, 3, 224, 224],
        raw_data=img_data.tobytes()
    )
    request = magna_pb2.InferenceRequest(inputs=[input_tensor], model_name=health.model_name)

    response = stub.Infer(request)
    print(f"[Client] Inference time: {response.inference_time_ms:.2f} ms")

    output = response.outputs[0]
    out_arr = np.frombuffer(output.raw_data, dtype=np.float32)
    top5_idx = np.argsort(out_arr)[-5:][::-1]
    print(f"[Client] Output shape: {list(output.shape)}")
    print(f"[Client] Top-5 class indices: {top5_idx.tolist()}")
    print(f"[Client] Top-5 scores: {out_arr[top5_idx].tolist()}")

if __name__ == '__main__':
    main()
