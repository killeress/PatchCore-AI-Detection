"""
CAPI AI Server 測試客戶端

用於在 Windows 本地測試 TCP 通訊與推論流程。

使用方式:
    # 1. 先啟動 server (另一個終端)
    python capi_server.py --config server_config.yaml

    # 2. 執行測試
    python test_client.py                     # 基本連線測試
    python test_client.py --real <panel_dir>  # 真實推論測試
"""

import socket
import sys
import time
import argparse


def send_request(host: str, port: int, message: str, timeout: float = 300) -> str:
    """
    發送請求並接收回覆

    Args:
        host: Server 地址
        port: Server 端口
        message: 請求訊息
        timeout: 超時秒數 (預設 300s，推論可能需要較長時間)

    Returns:
        Server 回覆字串
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)  # 先用短超時做 recv，以便顯示等待進度

    try:
        print(f">>> Connecting to {host}:{port}...")
        sock.connect((host, port))
        print(f">>> Connected!")

        print(f">>> Sending: {message}")
        sock.sendall((message + "\n").encode("utf-8"))

        # 接收回覆 (帶等待進度顯示)
        response = b""
        start_time = time.time()
        print(f">>> Waiting for response (timeout={timeout}s)...", end="", flush=True)

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(f"\n❌ Timeout after {timeout:.0f}s")
                break
            try:
                chunk = sock.recv(4096)
                if not chunk:
                    print(f" ({elapsed:.1f}s)")
                    if not response:
                        print("⚠️  Server closed connection without response!")
                        print("    Check server terminal for error messages.")
                    break
                response += chunk
                if b"\n" in response:
                    print(f" ({elapsed:.1f}s)")
                    break
            except socket.timeout:
                # 每 10 秒顯示一個點，表示仍在等待
                print(".", end="", flush=True)
                continue

        result = response.decode("utf-8", errors="ignore").strip()
        if result:
            print(f"<<< Response: {result}")
        else:
            print(f"<<< Response: (empty)")
        return result

    except ConnectionRefusedError:
        print(f"\n❌ Connection refused! Is the server running on {host}:{port}?")
        return ""
    except socket.timeout:
        print(f"\n❌ Connection timeout")
        return ""
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return ""
    finally:
        sock.close()


def test_basic(host: str, port: int):
    """基本連線與協議測試"""
    print("=" * 60)
    print("Test 1: 基本連線 + 假資料 (應回覆 ERR)")
    print("=" * 60)
    # 使用不存在的路徑，Server 應回覆 ERR:DIR_NOT_FOUND
    msg = r"AOI@TEST_GLASS_001;TEST_MODEL;CAPI1403;1920,1080;OK;C:\fake\path\to\images"
    resp = send_request(host, port, msg)
    if "ERR" in resp:
        print("✅ PASSED — Server correctly returned ERR for invalid path\n")
    elif "OK" in resp or "NG" in resp:
        print("⚠️ Unexpected — got OK/NG for invalid path\n")
    else:
        print("❌ FAILED — No valid response\n")

    print("=" * 60)
    print("Test 2: 無效協議格式 (應回覆 ERR)")
    print("=" * 60)
    msg = "INVALID_FORMAT_DATA"
    resp = send_request(host, port, msg)
    if "ERR" in resp or "PROTOCOL" in resp:
        print("✅ PASSED — Server correctly detected invalid protocol\n")
    else:
        print("❌ FAILED\n")

    print("=" * 60)
    print("Test 3: 欄位不足 (應回覆 ERR)")
    print("=" * 60)
    msg = "AOI@GLASS;MODEL;MACHINE"
    resp = send_request(host, port, msg)
    if "ERR" in resp:
        print("✅ PASSED — Server correctly detected insufficient fields\n")
    else:
        print("❌ FAILED\n")


def test_real_inference(host: str, port: int, panel_dir: str):
    """真實推論測試"""
    print("=" * 60)
    print(f"Test: Real Inference")
    print(f"  Panel dir: {panel_dir}")
    print("=" * 60)

    msg = f"AOI@TEST_GLASS_REAL;GN156HCAB6G0S;CAPI1403;1920,1080;OK;{panel_dir}"

    start = time.time()
    resp = send_request(host, port, msg, timeout=300)
    elapsed = time.time() - start

    print(f"\n--- Result ---")
    print(f"  Response : {resp}")
    print(f"  Time     : {elapsed:.2f}s")

    if resp:
        parts = resp.split(";")
        if len(parts) >= 5:
            ai_judgment = parts[4]
            print(f"  AI Judge : {ai_judgment}")
            if ai_judgment == "OK":
                print("  🟢 Panel is OK")
            elif ai_judgment.startswith("NG"):
                print("  🔴 Panel is NG")
                # 解析座標
                if "@" in ai_judgment:
                    coords_part = ai_judgment.split("@", 1)[1]
                    for item in coords_part.split("|"):
                        print(f"     → {item}")
            elif ai_judgment.startswith("ERR"):
                print(f"  🟡 Error: {ai_judgment}")
    print()


def main():
    parser = argparse.ArgumentParser(description="CAPI AI Server Test Client")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7891, help="Server port (default: 7891)")
    parser.add_argument("--real", metavar="PANEL_DIR", help="Real panel directory for inference test")
    parser.add_argument("--message", "-m", help="Send a custom raw message")
    args = parser.parse_args()

    if args.message:
        # 手動發送自訂訊息
        send_request(args.host, args.port, args.message)
    elif args.real:
        # 真實推論測試
        test_real_inference(args.host, args.port, args.real)
    else:
        # 基本測試
        test_basic(args.host, args.port)


if __name__ == "__main__":
    main()
