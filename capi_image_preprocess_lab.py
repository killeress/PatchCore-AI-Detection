"""Image preprocessing functions for the debug preprocessing lab."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


METHOD_SPECS: List[Dict[str, Any]] = [
    {
        "id": "median",
        "label": "標準中值濾波",
        "purpose": "孤立亮點/暗點、顆粒狀雜訊；窗口太大會吃掉小 defect。",
        "noise_types": ["低光顆粒狀雜訊", "孤立亮點/暗點", "salt-and-pepper 類脈衝雜訊"],
        "risk": "窗口越大越容易移除小尺寸、低對比或線狀 defect；建議先從 3 或 5 開始。",
        "suggested": "初始值 kernel_size=3 或 5；若 defect 尺寸接近窗口大小，應降低窗口或改用雙邊濾波。",
        "mix": "可放在第一步先移除孤立噪點，再接低強度雙邊濾波。",
        "params": [
            {"name": "窗口大小", "detail": "奇數，代表取鄰域中位數的範圍；越大去噪越強但細節損失越大。"},
        ],
    },
    {
        "id": "mean",
        "label": "空間鄰域平均法",
        "purpose": "低 SNR 隨機雜訊；最容易模糊細小 defect。",
        "noise_types": ["均勻隨機雜訊", "低 SNR 造成的細碎背景起伏"],
        "risk": "會把 defect 與周邊背景一起平均，是目前最容易造成漏檢的方法。",
        "suggested": "只建議用 kernel_size=3 或 5 做弱平滑；若結果 defect 邊界變淡，不建議進正式管線。",
        "mix": "若要混用，通常放最後做很弱的平滑；不建議接在條紋校正前。",
        "params": [
            {"name": "窗口大小", "detail": "奇數，代表平均範圍；調大會快速降低噪聲，也會快速降低 defect 對比。"},
        ],
    },
    {
        "id": "gaussian",
        "label": "高斯平滑",
        "purpose": "光電轉換或 sensor 類高斯雜訊；用 sigma 控制強度。",
        "noise_types": ["光電轉換造成的高斯雜訊", "sensor 隨機白雜訊", "輕度低光顆粒"],
        "risk": "會降低高頻細節；細線、針孔、小亮點 defect 可能被削弱。",
        "suggested": "初始值 kernel_size=3 或 5、sigma=0.8~1.2；先看差異圖中 defect 是否被吃掉。",
        "mix": "可接在條紋校正後做輕度平滑；若 defect 很細，優先改試雙邊濾波。",
        "params": [
            {"name": "窗口大小", "detail": "奇數，限制平滑影響範圍；通常 3/5/7 足夠做初步實驗。"},
            {"name": "Sigma", "detail": "高斯權重強度；越大越平滑，細節越容易變淡。"},
        ],
    },
    {
        "id": "bilateral",
        "label": "雙邊濾波",
        "purpose": "希望降噪但保留明顯邊界；參數過強仍會抹平低對比 defect。",
        "noise_types": ["低光顆粒狀雜訊", "高斯雜訊", "需要保留明顯邊界的背景噪聲"],
        "risk": "對低對比 defect 不一定保護得住；sigma_color 太大會把亮度差異也一起平均。",
        "suggested": "初始值 diameter=5~9、sigma_color=25~50、sigma_space=25~50。",
        "mix": "常作為第二步：中值濾波後接雙邊，或條紋校正後接雙邊。",
        "params": [
            {"name": "Diameter", "detail": "空間鄰域直徑；越大越慢，影響範圍也越大。"},
            {"name": "Sigma Color", "detail": "亮度差異容忍度；越大越會跨亮度差平均，漏檢風險上升。"},
            {"name": "Sigma Space", "detail": "空間距離權重；越大代表較遠像素也會參與平滑。"},
        ],
    },
    {
        "id": "laplace_sharpen",
        "label": "Laplace 銳化",
        "purpose": "強化邊緣與細節對比；會同步放大顆粒噪點與條紋雜訊。",
        "noise_types": ["前處理後邊界偏淡", "低對比 defect 視覺增強", "人工檢視用細節強化"],
        "risk": "不是去噪方法；若直接套在低光顆粒或條紋影像上，噪聲可能被放大並造成模型誤檢。",
        "suggested": "初始值 kernel_size=3、strength=0.3~0.6；通常先去噪再銳化。",
        "mix": "建議放在最後一步：中值/雙邊/NLM 去噪後，再用低 strength 做細節補償。",
        "params": [
            {"name": "Kernel Size", "detail": "Laplacian 運算核大小；越大強化範圍越寬，也越容易產生 halo。"},
            {"name": "Strength", "detail": "銳化強度；越大邊緣越強，但噪點和條紋也會被放大。"},
        ],
    },
    {
        "id": "nlm",
        "label": "Non-local Means",
        "purpose": "隨機白雜訊/低光雜訊；品質較好但速度較慢。",
        "noise_types": ["低光隨機顆粒", "白雜訊", "背景紋理重複但 defect 稀少的影像"],
        "risk": "h 太大會把與背景相似的小 defect 視為噪聲抹掉；search window 太大會變慢。",
        "suggested": "初始值 h=5~8、template=7、search=21；先用小圖或 ROI 試速度。",
        "mix": "通常單獨用或放在條紋校正後；不建議再疊強高斯/均值。",
        "params": [
            {"name": "H", "detail": "亮度去噪強度；越大去噪越強，細節保留越差。"},
            {"name": "H Color", "detail": "彩色影像色彩通道去噪強度；灰階影像不影響。"},
            {"name": "Template", "detail": "比對小區塊大小；通常維持 7。"},
            {"name": "Search", "detail": "搜尋相似區塊範圍；越大越慢，通常維持 21。"},
        ],
    },
    {
        "id": "stripe_profile",
        "label": "週期條紋背景校正",
        "purpose": "行/列方向固定條紋；用 profile 扣除週期性亮度偏移。",
        "noise_types": ["週期性垂直條紋", "週期性水平條紋", "固定方向背景亮度偏移"],
        "risk": "若 defect 本身也沿整行/整列延伸，profile 校正可能把 defect 當成條紋扣掉。",
        "suggested": "先選對條紋方向，strength 從 0.5~1.0 試；profile 平滑窗口需大於條紋週期。",
        "mix": "建議放第一步，再接低強度雙邊或高斯；不要先做均值平均再校正條紋。",
        "params": [
            {"name": "條紋方向", "detail": "垂直條紋會估計每一欄 profile；水平條紋會估計每一列 profile。"},
            {"name": "Profile 平滑窗口", "detail": "估計慢變背景的窗口；太小會保留條紋，太大可能扣到大尺度 defect。"},
            {"name": "校正強度", "detail": "扣除 profile 偏移的比例；越大條紋越淡，但漏檢風險也越高。"},
        ],
    },
]


def get_method_specs() -> List[Dict[str, Any]]:
    return [dict(spec) for spec in METHOD_SPECS]


def apply_preprocess_method(
    image: np.ndarray,
    method: str,
    params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Apply one preprocessing method and return processed image plus metadata."""
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("image must be a numpy array")

    params = params or {}
    work, conversion = _to_uint8(image)

    if work.ndim == 3 and work.shape[2] == 4:
        alpha = work[:, :, 3]
        work = work[:, :, :3]
    else:
        alpha = None

    method = str(method or "").strip().lower()
    applied_params: Dict[str, Any] = {}
    notes: List[str] = []

    if method == "median":
        ksize = _odd_int(params.get("kernel_size"), 3, 31, 5)
        processed = cv2.medianBlur(work, ksize)
        applied_params["kernel_size"] = ksize
        notes.append("中值濾波適合抑制孤立脈衝/顆粒噪點。")
    elif method == "mean":
        ksize = _odd_int(params.get("kernel_size"), 3, 99, 5)
        processed = cv2.blur(work, (ksize, ksize))
        applied_params["kernel_size"] = ksize
        notes.append("空間鄰域平均會降低隨機雜訊，也會同步降低細節對比。")
    elif method == "gaussian":
        ksize = _odd_int(params.get("kernel_size"), 3, 99, 5)
        sigma = _float_param(params.get("sigma"), 0.0, 20.0, 1.0)
        processed = cv2.GaussianBlur(work, (ksize, ksize), sigmaX=sigma)
        applied_params.update({"kernel_size": ksize, "sigma": sigma})
        notes.append("高斯平滑適合近似高斯分布的 sensor/光電轉換雜訊。")
    elif method == "bilateral":
        diameter = _int_param(params.get("diameter"), 1, 31, 9)
        sigma_color = _float_param(params.get("sigma_color"), 1.0, 200.0, 35.0)
        sigma_space = _float_param(params.get("sigma_space"), 1.0, 200.0, 35.0)
        processed = cv2.bilateralFilter(work, diameter, sigma_color, sigma_space)
        applied_params.update({
            "diameter": diameter,
            "sigma_color": sigma_color,
            "sigma_space": sigma_space,
        })
        notes.append("雙邊濾波會依亮度差異降低跨邊界平均，通常比均值/高斯更保邊。")
    elif method == "laplace_sharpen":
        ksize = _odd_int(params.get("kernel_size"), 1, 31, 3)
        strength = _float_param(params.get("strength"), 0.0, 3.0, 0.5)
        laplace = cv2.Laplacian(work.astype(np.float32), cv2.CV_32F, ksize=ksize)
        processed = np.clip(work.astype(np.float32) - strength * laplace, 0, 255).astype(np.uint8)
        applied_params.update({"kernel_size": ksize, "strength": strength})
        notes.append("Laplace 銳化會強化邊緣，也會放大噪點；建議放在去噪後並使用低 strength。")
    elif method == "nlm":
        h = _float_param(params.get("h"), 1.0, 50.0, 7.0)
        h_color = _float_param(params.get("h_color"), 1.0, 50.0, 7.0)
        template_window = _odd_int(params.get("template_window"), 3, 21, 7)
        search_window = _odd_int(params.get("search_window"), 7, 41, 21)
        if work.ndim == 2:
            processed = cv2.fastNlMeansDenoising(
                work,
                None,
                h=h,
                templateWindowSize=template_window,
                searchWindowSize=search_window,
            )
        else:
            processed = cv2.fastNlMeansDenoisingColored(
                work,
                None,
                h=h,
                hColor=h_color,
                templateWindowSize=template_window,
                searchWindowSize=search_window,
            )
        applied_params.update({
            "h": h,
            "h_color": h_color,
            "template_window": template_window,
            "search_window": search_window,
        })
        notes.append("NLM 的 h 越大去噪越強，但也越可能移除低對比細節。")
    elif method == "stripe_profile":
        orientation = str(params.get("orientation") or "vertical").strip().lower()
        if orientation not in ("vertical", "horizontal"):
            orientation = "vertical"
        smooth_kernel = _odd_int(params.get("smooth_kernel"), 7, 501, 61)
        strength = _float_param(params.get("strength"), 0.0, 1.5, 1.0)
        processed = _stripe_profile_correction(work, orientation, smooth_kernel, strength)
        applied_params.update({
            "orientation": orientation,
            "smooth_kernel": smooth_kernel,
            "strength": strength,
        })
        notes.append("條紋校正會扣除整行/整列亮度偏移，先用低 strength 確認 defect 不被削弱。")
    else:
        raise ValueError(f"unsupported preprocess method: {method}")

    if alpha is not None:
        processed = np.dstack([processed, alpha])

    return {
        "image": processed,
        "method": method,
        "method_label": _method_label(method),
        "applied_params": applied_params,
        "notes": notes,
        "conversion": conversion,
        "stats": image_stats(work, processed[:, :, :3] if processed.ndim == 3 and processed.shape[2] == 4 else processed),
    }


def make_diff_image(original: np.ndarray, processed: np.ndarray, amplify: float = 4.0) -> np.ndarray:
    original8, _ = _to_uint8(original)
    if original8.ndim == 3 and original8.shape[2] == 4:
        original8 = original8[:, :, :3]
    proc = processed[:, :, :3] if processed.ndim == 3 and processed.shape[2] == 4 else processed
    diff = cv2.absdiff(original8, proc)
    if diff.ndim == 3:
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff = np.clip(diff.astype(np.float32) * float(amplify), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(diff, cv2.COLORMAP_JET)


def image_stats(original_uint8: np.ndarray, processed_uint8: np.ndarray) -> Dict[str, float]:
    original_gray = _gray(original_uint8)
    processed_gray = _gray(processed_uint8)
    diff = cv2.absdiff(original_gray, processed_gray)
    original_std = float(np.std(original_gray))
    processed_std = float(np.std(processed_gray))
    std_delta_pct = 0.0
    if original_std > 1e-6:
        std_delta_pct = (processed_std - original_std) / original_std * 100.0
    return {
        "mean_abs_diff": round(float(np.mean(diff)), 3),
        "max_abs_diff": round(float(np.max(diff)), 3),
        "original_std": round(original_std, 3),
        "processed_std": round(processed_std, 3),
        "std_delta_pct": round(std_delta_pct, 2),
    }


def _stripe_profile_correction(
    image: np.ndarray,
    orientation: str,
    smooth_kernel: int,
    strength: float,
) -> np.ndarray:
    arr = image.astype(np.float32)
    if arr.ndim == 2:
        corrected = _stripe_plane(arr, orientation, smooth_kernel, strength)
    else:
        channels = [
            _stripe_plane(arr[:, :, channel], orientation, smooth_kernel, strength)
            for channel in range(arr.shape[2])
        ]
        corrected = np.dstack(channels)
    return np.clip(corrected, 0, 255).astype(np.uint8)


def _stripe_plane(
    plane: np.ndarray,
    orientation: str,
    smooth_kernel: int,
    strength: float,
) -> np.ndarray:
    if orientation == "horizontal":
        profile = np.median(plane, axis=1).astype(np.float32)
        smooth = cv2.GaussianBlur(profile.reshape(-1, 1), (1, smooth_kernel), 0).reshape(-1)
        bias = (profile - smooth).reshape(-1, 1)
    else:
        profile = np.median(plane, axis=0).astype(np.float32)
        smooth = cv2.GaussianBlur(profile.reshape(1, -1), (smooth_kernel, 1), 0).reshape(-1)
        bias = (profile - smooth).reshape(1, -1)
    return plane - float(strength) * bias


def _to_uint8(image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    if image.dtype == np.uint8:
        return image.copy(), {"converted": False, "input_dtype": "uint8"}

    arr = image.astype(np.float32)
    max_value = float(np.nanmax(arr)) if arr.size else 1.0
    if max_value <= 0:
        max_value = 1.0

    out = np.clip(arr / max_value * 255.0, 0, 255).astype(np.uint8)
    return out, {
        "converted": True,
        "input_dtype": str(image.dtype),
        "scale_max": max_value,
    }


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _method_label(method: str) -> str:
    for spec in METHOD_SPECS:
        if spec["id"] == method:
            return spec["label"]
    return method


def _int_param(value: Any, min_value: int, max_value: int, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(max_value, parsed))


def _odd_int(value: Any, min_value: int, max_value: int, default: int) -> int:
    parsed = _int_param(value, min_value, max_value, default)
    if parsed % 2 == 0:
        parsed = parsed + 1 if parsed < max_value else parsed - 1
    return max(min_value, min(max_value, parsed))


def _float_param(value: Any, min_value: float, max_value: float, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(max_value, parsed))
