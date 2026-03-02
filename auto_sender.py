"""
自動發送測試腳本

每隔 15 秒從 D:\CAPI_3F\ng 資料夾中隨機選取一個 panel，
組合成 AOI 協議訊息並發送至 CAPI AI Server。

欄位說明:
  1. Glass ID   — 從 D:\CAPI_3F\ng 資料夾名稱取得
  2. Model ID   — 固定 GN156HCAB6G0S
  3. Machine ID — 隨機 CAPI01 或 CAPI02
  4. Resolution — 固定 1920,1080
  5. AOI Judge  — 隨機 OK 或 NG
  6. Panel Path — 從 D:\CAPI_3F\ng 隨機選取

使用方式:
    python auto_sender.py
    python auto_sender.py --host 127.0.0.1 --port 7891
    python auto_sender.py --interval 10
"""

import socket
import os
import sys
import time
import random
import argparse


# ── 設定 ──────────────────────────────────────────────────────
NG_FOLDER = r"D:\CAPI_3F\ng"
MODEL_ID = "GN156HCAB6G0S"
MACHINE_IDS = ["CAPI01", "CAPI02"]
RESOLUTION = "1920,1080"
AOI_JUDGES = ["OK", "NG"]


def get_panel_list(folder: str) -> list:
    """取得資料夾內所有 panel 子目錄名稱"""
    panels = [
        d for d in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, d))
    ]
    panels.sort()
    return panels


def send_message(host: str, port: int, message: str, timeout: float = 300) -> str:
    """發送 TCP 訊息並接收回覆"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)

    try:
        sock.connect((host, port))
        sock.sendall((message + "\n").encode("utf-8"))

        response = b""
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(f"  ❌ Timeout after {timeout:.0f}s")
                break
            try:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
                if b"\n" in response:
                    break
            except socket.timeout:
                continue

        return response.decode("utf-8", errors="ignore").strip()

    except ConnectionRefusedError:
        return f"ERR:CONNECTION_REFUSED ({host}:{port})"
    except socket.timeout:
        return "ERR:CONNECTION_TIMEOUT"
    except Exception as e:
        return f"ERR:{e}"
    finally:
        sock.close()


def main():
    parser = argparse.ArgumentParser(description="CAPI Auto Sender — 自動週期發送測試訊息")
    parser.add_argument("--host", default="127.0.0.1", help="Server IP (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7891, help="Server port (default: 7891)")
    parser.add_argument("--interval", type=int, default=10, help="發送間隔秒數 (default: 15)")
    parser.add_argument("--count", type=int, default=0, help="發送次數，0=無限 (default: 0)")
    parser.add_argument("--ng-folder", default=NG_FOLDER, help=f"NG panel 資料夾路徑 (default: {NG_FOLDER})")
    args = parser.parse_args()

    # 取得所有 panel 目錄
    panels = get_panel_list(args.ng_folder)
    if not panels:
        print(f"❌ 找不到任何 panel 資料夾: {args.ng_folder}")
        sys.exit(1)

    print("=" * 65)
    print("  CAPI Auto Sender")
    print("=" * 65)
    print(f"  Server    : {args.host}:{args.port}")
    print(f"  Glass ID  : 從資料夾名稱取得 (隨機)")
    print(f"  Model ID  : {MODEL_ID}")
    print(f"  Machine   : {', '.join(MACHINE_IDS)} (隨機)")
    print(f"  AOI Judge : {', '.join(AOI_JUDGES)} (隨機)")
    print(f"  Panel 數  : {len(panels)}")
    print(f"  間隔      : {args.interval} 秒")
    print(f"  發送次數  : {'無限' if args.count == 0 else args.count}")
    print("=" * 65)
    print()

    sent = 0
    try:
        while True:
            sent += 1

            # 隨機選取參數
            panel_name = random.choice(panels)
            glass_id = panel_name  # Glass ID = 資料夾名稱
            machine_id = random.choice(MACHINE_IDS)
            aoi_judge = random.choice(AOI_JUDGES)
            panel_path = os.path.join(args.ng_folder, panel_name)

            # 組合訊息
            message = f"AOI@{glass_id};{MODEL_ID};{machine_id};{RESOLUTION};{aoi_judge};{panel_path}"

            # 顯示資訊
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] 第 {sent} 次發送")
            print(f"  → {message}")

            # 發送
            response = send_message(args.host, args.port, message)
            print(f"  ← {response}")
            print()

            # 檢查是否達到次數上限
            if args.count > 0 and sent >= args.count:
                print(f"✅ 已完成 {sent} 次發送")
                break

            # 等待
            print(f"  ⏳ 等待 {args.interval} 秒...")
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print(f"\n\n🛑 手動停止，共發送 {sent} 次")


if __name__ == "__main__":
    main()
