import socket
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description="Test CAPI Inference via TCP Socket")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=7891, help="Server port")
    parser.add_argument("--glass_id", default="TEST_GLASS_001", help="Glass ID")
    parser.add_argument("--model_id", default="T55A1911AG01", help="Model ID")
    parser.add_argument("--machine_no", default="M01", help="Machine No")
    parser.add_argument("--resolution", default="1920,1080", help="Resolution (Width,Height)")
    parser.add_argument("--judgment", default="NG", help="Machine Judgment")
    parser.add_argument("--image_dir", default=r"C:\Users\rh.syu\Desktop\CAPI01_AD\test_images", help="Path to panel images")
    
    args = parser.parse_args()

    # Protocol format: AOI@GlassID;ModelID;MachineNo;ResolutionX,ResolutionY;MachineJudgment;ImageDirectory
    payload = f"AOI@{args.glass_id};{args.model_id};{args.machine_no};{args.resolution};{args.judgment};{args.image_dir}"
    
    print(f"Connecting to TCP server at {args.host}:{args.port}...")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(60.0) # Inference might take a few seconds
            s.connect((args.host, args.port))
            print(f"Connected successfully!")
            print(f"Sending request:\n{payload}\n")
            
            start_time = time.time()
            s.sendall((payload + "\n").encode('utf-8'))
            
            print("Waiting for prediction response from the server...")
            response = s.recv(4096).decode('utf-8')
            elapsed_time = time.time() - start_time
            
            print(f"\n======================================")
            print(f"Server Response:\n{response.strip()}")
            print(f"======================================")
            print(f"Elapsed Time: {elapsed_time:.2f}s")
            print("Please check the CAPI Web UI -> Inference Records to view detailed results.")
            
    except Exception as e:
        print(f"Error: Could not connect or communicate with the server: {e}")

if __name__ == '__main__':
    main()
