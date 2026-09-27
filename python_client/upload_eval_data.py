import os
import sys
import grpc
import subprocess
import argparse
import hashlib
import time

import magna_pb2
import magna_pb2_grpc

def generate_chunks(source_dir, target_filename, target_dir):
    print(f"Starting TAR stream from {source_dir}...")
    # Using 'cf -' to write uncompressed tar to stdout
    proc = subprocess.Popen(
        ["tar", "cf", "-", "-C", source_dir, "."],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    sha256 = hashlib.sha256()
    is_first = True
    total_sent = 0

    while True:
        chunk = proc.stdout.read(1024 * 1024 * 2) # 2 MB chunks
        if not chunk:
            break

        sha256.update(chunk)
        total_sent += len(chunk)

        if is_first:
            yield magna_pb2.ModelChunk(
                filename=target_filename,
                target_dir=target_dir,
                data=chunk,
                total_size=0 # We don't know total size of tar stream in advance
            )
            is_first = False
        else:
            yield magna_pb2.ModelChunk(data=chunk)

    proc.wait()
    if proc.returncode != 0:
        err = proc.stderr.read().decode()
        print(f"TAR stream failed with code {proc.returncode}: {err}")
        sys.exit(1)

    print(f"Finished generating TAR stream. Total size sent: {total_sent / (1024**2):.2f} MB")
    print(f"HOST SHA-256 of TAR stream: {sha256.hexdigest()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="radxa-dragon-q6a.local:50051")
    parser.add_argument("--source_dir", type=str, default="models/eval_data/raw")
    parser.add_argument("--target_filename", type=str, default="eval_data.tar")
    parser.add_argument("--target_dir", type=str, default="eval_data")
    args = parser.parse_args()

    channel = grpc.insecure_channel(args.host, options=[
        ('grpc.max_send_message_length', 100 * 1024 * 1024),
        ('grpc.max_receive_message_length', 100 * 1024 * 1024),
    ])
    stub = magna_pb2_grpc.InferenceServiceStub(channel)

    start_time = time.time()
    try:
        print(f"Uploading via gRPC to {args.host}...")
        response = stub.UploadModel(generate_chunks(args.source_dir, args.target_filename, args.target_dir))

        if response.success:
            print(f"Upload successful! Saved at: {response.remote_path}")
            print(f"Bytes received by server: {response.bytes_received}")
        else:
            print(f"Upload failed: {response.message}")

    except grpc.RpcError as e:
        print(f"gRPC Error: {e.details()}")

    print(f"Total time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
