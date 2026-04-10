import cv2
import numpy as np
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capi_inference import CAPIInferencer
from capi_config import CAPIConfig

def inspect_edge_cv(image_path: str, config_yaml: str):
    """
    使用傳統 CV (OpenCV) 測試圖片左邊緣的缺陷
    完全模擬天目機台參數: 左側寬度 450, 灰階差 5, 最小面積 70
    """
    print(f"📦 載入圖片: {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"❌ 找不到圖片: {image_path}")
        return
        
    img_h, img_w = image.shape[:2]
    print(f"   尺寸: {img_w}x{img_h}")

    # 1. 取得 raw_bounds (產品實際邊界)
    # 利用 CAPIInferencer 來拿，確保基礎線一致
    config = CAPIConfig.from_yaml(config_yaml)
    inf = CAPIInferencer(config=config, device='cpu')
    raw_bounds, _ = inf._find_raw_object_bounds(image)
    rx1, ry1, rx2, ry2 = raw_bounds
    print(f"📐 產品邊界 (raw_bounds): {raw_bounds}")
    
    # 2. 擷取「左邊緣 ROI」 (寬度 450)
    # 根據機台圖: 寬度 450, 上下不判定 80, 80
    left_width = 450
    exclude_top = 80
    exclude_bottom = 80
    
    roi_x1 = rx1
    roi_x2 = min(rx1 + left_width, img_w)
    roi_y1 = ry1 + exclude_top
    roi_y2 = ry2 - exclude_bottom
    
    print(f"🎯 左側 ROI: X[{roi_x1}:{roi_x2}], Y[{roi_y1}:{roi_y2}]")
    roi_color = image[roi_y1:roi_y2, roi_x1:roi_x2].copy()
    
    # 彩圖轉灰階
    if len(roi_color.shape) == 3:
        roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    else:
        roi_gray = roi_color.copy()
        
    print(f"   ROI 灰階尺寸: {roi_gray.shape}")
    
    # 3. 特徵提取 (高斯模糊 3x3)
    # 天目參數「特徵 3x3」通常指 smoothing filter
    blurred = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    
    # 4. 估計背景 (找出正常平滑的表面)
    # 這裡我們用一個較大的 Median Blur 來「抹平」細小缺陷，當作背景
    # 這樣背景圖就不會包含突變的缺陷
    bg_estimated = cv2.medianBlur(blurred, 65)  # window 需夠大，比最大的可容忍雜訊大
    
    # 5. 計算差異圖 (取絕對值)
    diff = cv2.absdiff(blurred, bg_estimated)
    
    # 6. 二值化 (明暗度閾值 = 5)
    # 天目設定明暗度為 5，也就是差值大於 5 就視為候選缺陷
    threshold_value = 5
    _, binary_mask = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
    
    # 7. 連通組件分析 (最小面積 = 70)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    min_area = 70
    defects = []
    
    # 畫出 DEBUG 圖
    debug_img = roi_color.copy()
    
    # stats[0] 是整個背景，從 1 開始遍歷
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # 篩選面積 >= 70
        if area >= min_area:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[i]
            
            # 轉換回圖片絕對座標
            abs_x = roi_x1 + x
            abs_y = roi_y1 + y
            
            defects.append({
                'id': i,
                'area': area,
                'roi_rect': (x, y, w, h),
                'abs_rect': (abs_x, abs_y, w, h),
                'center': (int(roi_x1 + cx), int(roi_y1 + cy))
            })
            
            # 畫框
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(debug_img, f"A:{area}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    print(f"\n==================================================")
    print(f"🚀 CV 邊緣檢測結果 (左側)")
    print(f"==================================================")
    print(f"總共找到 >= {min_area} px 的異常數量: {len(defects)}")
    
    # 驗證座標 3, 493 是否在其中
    # 產品 (3, 493) → 圖片 (419, 1765)
    target_img_x = 419
    target_img_y = 1765
    
    target_found = False
    for d in defects:
        rx, ry, rw, rh = d['abs_rect']
        cx, cy = d['center']
        dist = np.sqrt((cx - target_img_x)**2 + (cy - target_img_y)**2)
        
        print(f"   ⚠️ 缺陷 #{d['id']}: 面積 {d['area']}, 絕對座標 [{rx},{ry} {rw}x{rh}], 中心: ({cx},{cy})")
        if dist < 100:  # 在目標附近
            print(f"      🎯 這就是我們的目標缺陷！距離: {dist:.1f}px")
            target_found = True
            
    if not target_found:
        print(f"\n❌ 沒有在預測位置 ({target_img_x}, {target_img_y}) 附近找到缺陷。")
        print("   可能原因：")
        print("   1. 背景估算方式 (MedianBlur) 無法突顯該特徵")
        print("   2. 明暗差實際不到 5 (對比度太低)")
        print("   3. 連通面積未達 70 (散佈而非集中)")
    
    # 儲存 Debug 圖片
    result_path = "cv_edge_debug_left.png"
    
    # 將灰階圖轉回 BGR 方便合併
    if len(roi_color.shape) == 2:
        roi_color_bgr = cv2.cvtColor(roi_color, cv2.COLOR_GRAY2BGR)
    else:
        roi_color_bgr = roi_color
        
    bg_bgr = cv2.cvtColor(bg_estimated, cv2.COLOR_GRAY2BGR)
    diff_colored = cv2.applyColorMap(diff * 10, cv2.COLORMAP_JET) # 放大差異以便觀察
    mask_bgr = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    
    if len(debug_img.shape) == 2:
        debug_img_bgr = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
    else:
        debug_img_bgr = debug_img
    
    # 直向拼接 (ROI可能很高)，所以橫向排開
    h_stack = np.hstack((roi_color_bgr, bg_bgr, diff_colored, mask_bgr, debug_img_bgr))
    
    # 影像如果太高（如 3000px），縮小一點儲存
    scale_percent = 50
    width = int(h_stack.shape[1] * scale_percent / 100)
    height = int(h_stack.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_stack = cv2.resize(h_stack, dim, interpolation=cv2.INTER_AREA)

    cv2.imwrite(result_path, resized_stack)
    print(f"\n💾 Debug 影像已儲存: {result_path}")
    print(f"   格式: [原圖] [估計背景] [差異圖(放大10倍)] [二值化Mask] [有缺陷框的原圖]")

if __name__ == "__main__":
    img_path = "R0F00000_082857.tif"
    config = "configs/capi_3f.yaml"
    inspect_edge_cv(img_path, config)
