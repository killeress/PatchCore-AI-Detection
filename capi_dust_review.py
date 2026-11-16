"""Select AOI Top% cores and verify low-coverage regions against image features.

Unlike the legacy two-stage check, this review starts from an NG decision.
An empty or incomplete observation must therefore never suppress a region.
"""

import cv2
import numpy as np


def recover_small_particle_cores(binary, area_min, area_max, extension):
    """Keep native pixels of small particles accepted by the expanded detector.

    Expansion establishes eligibility only. Never return its surrounding pixels
    as precise surface evidence, or lower the threshold for unrelated specks.
    """
    source = (np.asarray(binary) > 0).astype(np.uint8)
    recovered = np.zeros_like(source)
    if extension <= 0 or area_min <= 3 or not np.any(source):
        return recovered
    _, native_labels, native_stats, _ = cv2.connectedComponentsWithStats(source, connectivity=8)
    small = (native_stats[:, cv2.CC_STAT_AREA] >= 3) & (native_stats[:, cv2.CC_STAT_AREA] < area_min)
    small[0] = False
    if not np.any(small):
        return recovered
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (extension * 2 + 1, extension * 2 + 1))
    expanded = cv2.dilate(source, kernel)
    _, expanded_labels, expanded_stats, _ = cv2.connectedComponentsWithStats(expanded, connectivity=8)
    accepted = expanded_stats[:, cv2.CC_STAT_AREA] >= area_min
    if area_max is not None:
        accepted &= expanded_stats[:, cv2.CC_STAT_AREA] <= area_max
    accepted[0] = False
    recovered[small[native_labels] & accepted[expanded_labels]] = 255
    return recovered


def small_particle_core_mask(precise_dust_mask, area_min):
    """Identify accepted, unexpanded components below the candidate area limit."""
    if precise_dust_mask is None:
        return None
    source = np.asarray(precise_dust_mask)
    if source.ndim != 2 or not source.size:
        return None
    _, labels, stats, _ = cv2.connectedComponentsWithStats((source > 0).astype(np.uint8), connectivity=8)
    small = (stats[:, cv2.CC_STAT_AREA] >= 3) & (stats[:, cv2.CC_STAT_AREA] < area_min)
    small[0] = False
    return small[labels].astype(np.uint8) * 255


def build_aoi_roi_core_mask(anomaly_map, heatmap_binary, roi_mask, top_percent):
    """Select the ROI Top% core shared by formal dust judgment and rendering."""
    percent = float(top_percent)
    if not 0 < percent <= 100:
        raise ValueError("top_percent 必須大於 0 且不超過 100")
    values = np.asarray(anomaly_map, dtype=np.float32)
    h, w = values.shape
    allowed = cv2.resize((roi_mask > 0).astype(np.uint8), (w, h),
                         interpolation=cv2.INTER_NEAREST) > 0
    candidates = cv2.resize((heatmap_binary > 0).astype(np.uint8), (w, h),
                            interpolation=cv2.INTER_NEAREST) > 0
    valid = allowed & np.isfinite(values) & (values > 0)
    core = np.zeros(values.shape, dtype=np.uint8)
    if np.any(valid):
        cutoff = np.percentile(values[valid], 100 - percent)
        core[valid & candidates & (values >= cutoff)] = 255
    return core


def _is_real_grayscale_omit(omit_crop, gray_image):
    """確認 omit_crop 為真實灰階 OMIT 影像，而非同圖佔位或二值遮罩。"""
    if omit_crop is None or gray_image is None:
        return False
    omit_gray = np.asarray(omit_crop)
    if omit_gray.ndim == 3 and omit_gray.shape[2] == 3:
        omit_gray = cv2.cvtColor(omit_gray, cv2.COLOR_BGR2GRAY)
    gray = np.asarray(gray_image)
    if gray.ndim == 3 and gray.shape[2] == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if (omit_gray.ndim != 2 or gray.ndim != 2 or omit_gray.shape != gray.shape
            or not np.isfinite(omit_gray).all() or np.array_equal(omit_gray, gray)):
        return False
    return int(np.unique(omit_gray).size) > 2


def _recover_local_omit_dust_mask(original_image, precise_dust_mask, omit_crop):
    """在原圖顯著特徵位置重試局部 OMIT 弱灰塵偵測，補回被全域平滑濾除的灰塵核心。"""
    dust = np.asarray(precise_dust_mask) > 0
    if original_image is None or not _is_real_grayscale_omit(omit_crop, original_image):
        return dust
    gray = np.asarray(original_image)
    if gray.ndim == 3 and gray.shape[2] == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    omit_gray = np.asarray(omit_crop)
    if omit_gray.ndim == 3 and omit_gray.shape[2] == 3:
        omit_gray = cv2.cvtColor(omit_gray, cv2.COLOR_BGR2GRAY)
    if (gray.ndim != 2 or omit_gray.ndim != 2 or gray.shape != dust.shape
            or omit_gray.shape != dust.shape or min(gray.shape) < 31
            or not np.isfinite(gray).all()):
        return dust

    height, width = gray.shape
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    residual = smooth - cv2.GaussianBlur(smooth, (31, 31), 0)
    omit_f = omit_gray.astype(np.float32)
    recovered = dust.copy()
    bg_dilate_kernel = np.ones((7, 7), np.uint8)
    boundary_kernel = np.ones((3, 3), np.uint8)

    for sign in (-1, 1):
        binary = (sign * residual >= 2.0).astype(np.uint8)
        count, comp_labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for comp_id in range(1, count):
            x, y, cw, ch, area = (int(v) for v in stats[comp_id])
            if not (3 <= area <= 256):
                continue
            if x < 2 or y < 2 or x + cw > width - 2 or y + ch > height - 2:
                continue
            y0, y1 = max(0, y - 20), min(height, y + ch + 20)
            x0, x1 = max(0, x - 20), min(width, x + cw + 20)
            comp_win = comp_labels[y0:y1, x0:x1] == comp_id
            bg_mask = ~(dust[y0:y1, x0:x1] | (cv2.dilate(comp_win.astype(np.uint8), bg_dilate_kernel) > 0))
            if int(np.count_nonzero(bg_mask)) < 32:
                continue
            win = omit_f[y0:y1, x0:x1]
            med = float(np.median(win[bg_mask]))
            mad = max(0.5, 1.4826 * float(np.median(np.abs(win[bg_mask] - med))))
            if float(np.max(win[comp_win]) - med) < max(4.5, 3.5 * mad):
                continue
            pos_thr = max(2.5, 2.0 * mad)
            local_omit_pos = (win - med >= pos_thr).astype(np.uint8)
            o_count, o_labels, o_stats, _ = cv2.connectedComponentsWithStats(local_omit_pos, connectivity=8)
            o_ids = [int(i) for i in np.unique(o_labels[comp_win]) if i > 0 and i < o_count]
            if not o_ids:
                continue
            valid_ids = [i for i in o_ids if 3 <= int(o_stats[i, cv2.CC_STAT_AREA]) <= max(96, area * 3)]
            if not valid_ids:
                continue
            o_comp = np.isin(o_labels, valid_ids)
            o_comp_dil = cv2.dilate(o_comp.astype(np.uint8), boundary_kernel) > 0
            overlap = int(np.count_nonzero(comp_win & o_comp))
            if overlap >= max(2, int(np.ceil(0.50 * area))) and not np.any(comp_win & ~o_comp_dil):
                # 僅保留未模糊 OMIT 上確實高於局部背景的像素，不將背景納入 dust mask
                add_mask = comp_win & o_comp
                if np.any(add_mask):
                    recovered[y0:y1, x0:x1] |= add_mask
    return recovered


def review_low_coverage_regions(
    original_image,
    precise_dust_mask,
    anomaly_map,
    region_labels,
    region_details,
    *,
    coverage_threshold,
    evaluation_mask=None,
    support_anomaly_map=None,
    small_core_mask=None,
    omit_crop=None,
    real_omit=None,
    allow_zero_overlap=False,
):
    """Return copied region decisions and feature evidence in image coordinates.

    Coverage remains the original Top% metric. The pre-core, post-exclusion map
    can associate a displaced core with its connected half-peak response inside
    the ROI. Every observed feature must still match the unextended OMIT mask;
    a nearby dust patch or an empty observation cannot clear a region.

    ``real_omit`` lets a caller that already ran the local OMIT recovery pass
    ``omit_crop=None`` (avoid a duplicate search) while keeping the real-OMIT
    gated behaviour. ``allow_zero_overlap`` admits a whole-ROI auxiliary region
    whose only dust lies just outside the ROI (border-halo review).
    """
    regions = [dict(region) for region in region_details]
    candidates = [region for region in regions if (
        not region["is_dust"]
        and (0 < region.get("dust_overlap", 0) or allow_zero_overlap or (
            support_anomaly_map is not None and evaluation_mask is not None
        ))
        and region["coverage"] < coverage_threshold
    )]
    if not candidates:
        return regions, []
    for region in candidates:
        region["feature_reviewed"] = False
        region["feature_review_reason"] = "missing_evidence"
    if any(value is None for value in (
        original_image, precise_dust_mask, anomaly_map, region_labels,
    )):
        return regions, []

    gray = np.asarray(original_image)
    if gray.ndim == 3 and gray.shape[2] == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    amap = np.asarray(anomaly_map)
    labels = np.asarray(region_labels)
    dust = np.asarray(precise_dust_mask)
    if (gray.ndim != 2 or min(gray.shape) < 31 or dust.shape != gray.shape
            or amap.ndim != 2 or labels.shape != amap.shape
            or not np.isfinite(gray).all() or not np.isfinite(amap).all()):
        return regions, []
    height, width = gray.shape
    labels_tile = cv2.resize(labels.astype(np.float32), (width, height),
                             interpolation=cv2.INTER_NEAREST).astype(np.int32)
    allowed = np.ones(gray.shape, dtype=bool)
    if evaluation_mask is not None:
        evaluation = np.asarray(evaluation_mask)
        if evaluation.ndim != 2 or not evaluation.size:
            return regions, []
        allowed = cv2.resize(evaluation.astype(np.uint8), (width, height),
                             interpolation=cv2.INTER_NEAREST) > 0
    dust = dust > 0
    if omit_crop is not None:
        dust = _recover_local_omit_dust_mask(gray, dust, omit_crop)
    _, dust_cc_labels, dust_cc_stats, _ = cv2.connectedComponentsWithStats(
        dust.astype(np.uint8), connectivity=8)
    dust_cc_boundary = cv2.dilate(dust.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    small_labels = small_stats = small_boundary = None
    if small_core_mask is not None:
        small = np.asarray(small_core_mask)
        if small.shape != dust.shape or np.any((small > 0) & ~dust):
            return regions, []
        _, small_labels, small_stats, _ = cv2.connectedComponentsWithStats(
            (small > 0).astype(np.uint8), connectivity=8)
        # Allow only a one-pixel segmentation boundary for matching. These
        # pixels are never added to the surface edit/inpainting mask.
        small_boundary = cv2.dilate((small > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    support = None
    allowed_map = None
    if support_anomaly_map is not None:
        support = np.asarray(support_anomaly_map, dtype=np.float32)
        if (evaluation_mask is None or support.shape != amap.shape
                or not np.isfinite(support).all()):
            return regions, []
        allowed_map = cv2.resize(allowed.astype(np.uint8), (amap.shape[1], amap.shape[0]),
                                 interpolation=cv2.INTER_NEAREST) > 0

    # Suppress the LCD pixel lattice before measuring local dark/bright contrast.
    # A robust noise estimate avoids letting one strong dust spot hide a weaker
    # independent defect, as a percentile of positive contrast would do.
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    residual = smooth - cv2.GaussianBlur(smooth, (31, 31), 0)
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    inner_edge = allowed & (cv2.erode(allowed.astype(np.uint8), np.ones((5, 5), np.uint8)) == 0)
    has_real_omit = (_is_real_grayscale_omit(omit_crop, gray)
                     if real_omit is None else bool(real_omit))

    def _scan_features_in_scope(
        scope_mask, region_id, *, allow_outside_roi=False,
    ):
        bg_mask = scope_mask & ~dust
        background = residual[bg_mask]
        if background.size < 16:
            return None  # 背景不足：無法完成檢查，不可視為「已檢查且無特徵」
        median = float(np.median(background))
        noise = 1.4826 * float(np.median(np.abs(background - median)))
        contrast_threshold = max(2.0, 4.0 * noise)
        ring_allowed = ~allowed if allow_outside_roi else allowed
        found = []
        for sign, spot_type in ((-1, "dark"), (1, "bright")):
            # Components are extracted before ROI clipping: a cut component is
            # unresolved, rather than a smaller artificial match at the ROI edge.
            binary = (sign * residual >= contrast_threshold).astype(np.uint8)
            count, component_labels, stats, _ = cv2.connectedComponentsWithStats(
                binary, connectivity=8)
            intersecting = np.unique(component_labels[scope_mask])
            for feature_id in intersecting:
                if feature_id == 0 or feature_id >= count:
                    continue
                x, y, w, h, area = (int(value) for value in stats[feature_id])
                if area < 3:
                    continue
                component = component_labels == feature_id
                clipped = bool(x < 2 or y < 2 or x + w > width - 2
                               or y + h > height - 2
                               or (not allow_outside_roi and np.any(component & ~allowed))
                               or (allow_outside_roi and np.any(component & allowed)))
                ring = (cv2.dilate(component.astype(np.uint8), ring_kernel) > 0) & ~component
                ring &= ring_allowed if allow_outside_roi else allowed
                if not np.any(ring):
                    local_contrast = 0.0
                else:
                    local_contrast = sign * (
                        float(np.mean(smooth[component])) - float(np.median(smooth[ring])))
                if not clipped and local_contrast < contrast_threshold * 0.5:
                    continue
                overlap = int(np.count_nonzero(component & dust))
                ratio = overlap / area
                # Two nearby spots can merge after smoothing. A small independent
                # tail must not disappear behind the majority's good dust match.
                residual_mask = component & ~dust
                residual_count, residual_labels, residual_stats, _ = cv2.connectedComponentsWithStats(
                    residual_mask.astype(np.uint8), connectivity=8)
                residual_ids = [index for index in range(1, residual_count)
                                if residual_stats[index, cv2.CC_STAT_AREA] >= 3]
                significant_residual = np.isin(residual_labels, residual_ids)
                has_residual = bool(np.any(significant_residual))
                matched = not clipped and ratio >= 0.80 and not has_residual
                small_boundary_match = False
                if not matched and not clipped and ratio >= 0.5:
                    if small_labels is not None:
                        core_ids = np.unique(small_labels[component])
                        core_ids = core_ids[core_ids > 0]
                        if len(core_ids) == 1:
                            core_id = int(core_ids[0])
                            core_overlap = int(np.count_nonzero(component & (small_labels == core_id)))
                            small_boundary_match = bool(
                                core_overlap >= 0.80 * small_stats[core_id, cv2.CC_STAT_AREA]
                                and not np.any(component & ~dust & ~small_boundary)
                            )
                            matched = small_boundary_match
                    if not matched and overlap > 0 and has_real_omit:
                        # 僅在真實灰階 OMIT 影像存在時，對完整落在單一 OMIT 原生元件 1-px 邊界內的灰塵允許比對
                        core_ids = np.unique(dust_cc_labels[component])
                        core_ids = core_ids[core_ids > 0]
                        if len(core_ids) == 1:
                            core_id = int(core_ids[0])
                            core_overlap = int(np.count_nonzero(component & (dust_cc_labels == core_id)))
                            matched = bool(
                                core_overlap >= 0.80 * dust_cc_stats[core_id, cv2.CC_STAT_AREA]
                                and not np.any(component & ~dust & ~dust_cc_boundary)
                            )
                            if matched:
                                small_boundary_match = True
                ys, xs = np.where(component)
                contours, _ = cv2.findContours(component.astype(np.uint8),
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour = max(contours, key=cv2.contourArea).reshape(-1, 2).tolist()
                feature = {
                    "zone_id": region_id,
                    "abs_pos": (int(np.mean(xs)), int(np.mean(ys))),
                    "area": area,
                    "type": spot_type,
                    "feature_bbox": (x, y, w, h),
                    "feature_contour": contour,
                    "dust_overlap": overlap,
                    "dust_ratio": ratio,
                    "is_dust": matched,
                    "dust_reason": ("small_core_boundary" if small_boundary_match else
                                    "precise_overlap" if matched else "clipped" if clipped else "unmatched"),
                    "unmatched_area": int(np.count_nonzero(significant_residual)),
                    "contrast": local_contrast,
                    "contrast_threshold": contrast_threshold,
                }
                if allow_outside_roi:
                    feature["outside_roi_halo"] = True
                if not matched:
                    real_support = significant_residual & scope_mask
                    if not np.any(real_support):
                        real_support = component & scope_mask
                    ry, rx = np.unravel_index(
                        np.argmax(np.where(real_support, sign * residual, -np.inf)), gray.shape)
                    feature["real_pos"] = (int(rx), int(ry))
                found.append(feature)
        return found

    features = []
    border_halo_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (73, 73))
    for region in candidates:
        zone = (labels_tile == region["label_id"]) & allowed
        region["feature_review_scope"] = "core"
        if support is not None:
            core = (labels == region["label_id"]) & allowed_map & (amap > 0)
            if not np.any(core):
                continue
            # This is association evidence only: neither Top% selection nor its
            # coverage denominator is expanded. Preserve excluded/thresholded
            # pixels from the production map instead of reusing the display map.
            half_peak = float(np.max(amap[core])) * 0.5
            response = (support >= half_peak) & allowed_map
            _, response_labels = cv2.connectedComponents(response.astype(np.uint8), connectivity=8)
            response_ids = np.unique(response_labels[core])
            response_ids = response_ids[response_ids > 0]
            scope = np.isin(response_labels, response_ids)
            # A shared warm response cannot let one core borrow another core's
            # surface evidence. Ambiguous associations keep the original scope.
            foreign_core = scope & (labels > 0) & (labels != region["label_id"])
            if np.all(scope[core]) and not np.any(foreign_core):
                zone = cv2.resize(scope.astype(np.uint8), (width, height),
                                   interpolation=cv2.INTER_NEAREST) > 0
                zone &= allowed
                region["feature_review_scope"] = "half_peak_component"
        region["feature_review_scope_pixels"] = int(np.count_nonzero(zone))
        region["feature_reviewed"] = True
        region["feature_review_reason"] = "no_matched_features"
        region_features = []
        border_review = bool(has_real_omit and evaluation_mask is not None
                             and np.any(zone & inner_edge))
        inner_scan = None
        if np.any(zone) and (np.any(zone & dust) or border_review):
            # 外側補判前必須先完成內側檢查（即使 ROI 內無灰塵像素），
            # 否則 ROI 內的獨立暗/亮點會被 ROI 外的灰塵「代為放行」。
            inner_scan = _scan_features_in_scope(zone, region["label_id"])
            region_features = inner_scan or []
        if not region_features and border_review and inner_scan is not None:
            outer_collar = (
                cv2.dilate((zone & inner_edge).astype(np.uint8), border_halo_kernel) > 0
            ) & ~allowed
            if np.any(outer_collar & dust):
                outer_features = _scan_features_in_scope(
                    outer_collar, region["label_id"], allow_outside_roi=True,
                )
                if outer_features and all(f["is_dust"] for f in outer_features):
                    region_features = outer_features
                    region["feature_review_scope"] = "roi_border_halo"
        features.extend(region_features)
        matched = [feature for feature in region_features if feature["is_dust"]]
        unmatched = [feature for feature in region_features if not feature["is_dust"]]
        region["feature_review_matched"] = len(matched)
        region["feature_review_unmatched"] = len(unmatched)
        if unmatched:
            region["feature_review_reason"] = "unmatched_features"
            # Keep reporting the independent real feature, not the dust peak.
            feature = max(unmatched, key=lambda item: item["contrast"])
            fx, fy = feature["real_pos"]
            region["real_feature_peak_yx"] = (
                min(amap.shape[0] - 1, int(fy * amap.shape[0] / height)),
                min(amap.shape[1] - 1, int(fx * amap.shape[1] / width)),
            )
        elif matched:
            region["is_dust"] = True
            region["feature_review_reason"] = "all_features_match_precise_dust"
    return regions, features


def confirmed_roi_surface_mask(
    original_image,
    precise_dust_mask,
    roi_mask,
    support_map,
    *,
    small_core_mask=None,
    omit_crop=None,
    high_coverage_fallback=False,
):
    """Confirm surface components using all active ROI pixels, without Top%/half-peak cuts.

    The result is only an edit mask for residual inference. It is never itself
    sufficient evidence to release the tile, or a replacement for formal COV.
    """
    if any(value is None for value in (original_image, precise_dust_mask, roi_mask, support_map)):
        return None, [], "missing_evidence"
    image = np.asarray(original_image)
    dust = np.asarray(precise_dust_mask)
    roi = np.asarray(roi_mask)
    support = np.asarray(support_map, dtype=np.float32)
    if (image.ndim not in (2, 3) or dust.shape != image.shape[:2]
            or roi.ndim != 2 or not roi.size or support.ndim != 2 or not support.size
            or not np.isfinite(support).all()):
        return None, [], "invalid_evidence"
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    has_real_omit = _is_real_grayscale_omit(omit_crop, gray)
    dust_bool = dust > 0
    if has_real_omit:
        dust_bool = _recover_local_omit_dust_mask(gray, dust_bool, omit_crop)
    dust = dust_bool.astype(np.uint8) * 255
    allowed = cv2.resize((roi > 0).astype(np.uint8), (width, height),
                         interpolation=cv2.INTER_NEAREST) > 0
    active = (support > 0) & (cv2.resize((roi > 0).astype(np.uint8),
              (support.shape[1], support.shape[0]), interpolation=cv2.INTER_NEAREST) > 0)
    if not np.any(active) or not np.any((dust > 0) & allowed):
        # 檢查是否存在 ROI 邊界緊鄰的跨界外圍灰塵（DoG/感受野暈圈可能使 ROI 內無 OMIT 核心像素）
        has_border_dust = False
        if has_real_omit and np.any(dust > 0) and np.any(active):
            inner_edge_check = allowed & (cv2.erode(allowed.astype(np.uint8), np.ones((5, 5), np.uint8)) == 0)
            active_img_check = cv2.resize(active.astype(np.uint8), (width, height),
                                          interpolation=cv2.INTER_NEAREST) > 0
            if np.any(active_img_check & inner_edge_check):
                has_border_dust = True
        if not has_border_dust:
            return None, [], "missing_surface_evidence"
    labels = active.astype(np.int32)
    dust_map = cv2.resize((dust > 0).astype(np.uint8),
                          (support.shape[1], support.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlap = int(np.count_nonzero(active & (dust_map > 0)))
    region = dict(label_id=1, is_dust=False, coverage=overlap / int(active.sum()),
                  dust_overlap=overlap, area=int(active.sum()))
    # This auxiliary region describes every above-threshold ROI pixel. It does
    # not replace the stored Top% labels or their measured coverage.
    reviewed, features = review_low_coverage_regions(
        image, dust, support, labels, [region], coverage_threshold=1.01,
        evaluation_mask=roi,
        small_core_mask=small_core_mask,
        # dust 已完成局部 OMIT 補回，不再重複搜尋；保留真實 OMIT 判定結果
        omit_crop=None,
        real_omit=has_real_omit,
        # 僅 ROI 外側有灰塵時，整區候選重疊量為 0，仍須進入外側補判
        allow_zero_overlap=has_real_omit,
    )
    _, dust_labels = cv2.connectedComponents((dust > 0).astype(np.uint8), connectivity=8)
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    residual = smooth - cv2.GaussianBlur(smooth, (31, 31), 0)

    if not reviewed[0].get("is_dust"):
        review_reason = reviewed[0].get("feature_review_reason", "missing_evidence")
        if (high_coverage_fallback and review_reason == "no_matched_features"
                and not any(not f.get("is_dust") for f in features)):
            active_img = cv2.resize(active.astype(np.uint8), (width, height),
                                    interpolation=cv2.INTER_NEAREST) > 0
            dust_full = dust > 0
            candidate_surface = active_img & dust_full & allowed
            bg_res = residual[allowed & ~dust_full]
            if np.count_nonzero(candidate_surface) >= 16 and bg_res.size >= 16:
                bg_med = float(np.median(bg_res))
                bg_noise = 1.4826 * float(np.median(np.abs(bg_res - bg_med)))
                weak_thr = max(1.0, 2.5 * bg_noise)
                peak_thr = max(1.2, 3.0 * bg_noise)
                # 與特徵掃描相同的顯著門檻：達此強度卻未被掃描列為特徵，
                # 代表不是「漫射弱灰塵」，不得以補判路徑修補。
                strong_thr = max(2.0, 4.0 * bg_noise)
                # 允許面板前景邊界與 AOI ROI 邊界之間極小（<= 2 px）的對齊差異
                allowed_margin = cv2.dilate(allowed.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                edit = np.zeros((height, width), bool)
                for comp_id in np.unique(dust_labels[candidate_surface]):
                    if comp_id <= 0:
                        continue
                    comp = dust_labels == comp_id
                    part = comp & active_img
                    if not np.any(part):
                        continue
                    comp_dev = np.abs(residual[part] - bg_med)
                    # 逐元件判定：僅確認自身具弱訊號的元件，略過無訊號元件
                    if not (float(np.max(comp_dev)) >= peak_thr
                            and int(np.count_nonzero(comp_dev >= weak_thr)) >= 3):
                        continue
                    # 以未裁切的原始灰塵元件檢查截斷：觸及影像邊緣、超出 2 px ROI 緩衝或比例過高皆不可確認
                    outside_px = int(np.count_nonzero(comp & ~allowed))
                    max_boundary_fringe = min(8, int(0.01 * int(np.count_nonzero(comp))))
                    if (np.any(comp & ~allowed_margin) or outside_px > max_boundary_fringe
                            or np.any(comp[:2]) or np.any(comp[-2:])
                            or np.any(comp[:, :2]) or np.any(comp[:, -2:])):
                        return None, features, "surface_mask_clipped"
                    # 與特徵掃描對齊：僅當形成 >= 3 px 連通強特徵元件時才視為非漫射弱灰塵
                    strong_mask = ((np.abs(residual - bg_med) >= strong_thr) & part).astype(np.uint8)
                    if int(np.count_nonzero(strong_mask)) >= 3:
                        s_cnt, _, s_stats, _ = cv2.connectedComponentsWithStats(strong_mask, connectivity=8)
                        if any(int(s_stats[idx, cv2.CC_STAT_AREA]) >= 3 for idx in range(1, s_cnt)):
                            return None, features, "fallback_strong_feature"
                    edit |= comp
                if np.count_nonzero(edit) >= 3:
                    ring = (cv2.dilate(edit.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0) & ~edit & allowed
                    if np.count_nonzero(ring) >= 16:
                        return edit.astype(np.uint8) * 255, features, "confirmed_surface"
        return None, features, review_reason

    selected = set()
    outer_selected = set()
    edit = np.zeros((height, width), bool)
    feature_labels = {}
    for feature in features:
        if not feature["is_dust"]:
            return None, features, "unmatched_features"
        contour = np.asarray(feature["feature_contour"], dtype=np.int32).reshape(-1, 1, 2)
        sign = -1 if feature["type"] == "dark" else 1
        key = (sign, feature["contrast_threshold"])
        if key not in feature_labels:
            _, feature_labels[key] = cv2.connectedComponents(
                (sign * residual >= key[1]).astype(np.uint8), connectivity=8)
        fx, fy = contour[0, 0]
        component = feature_labels[key] == feature_labels[key][fy, fx]
        ids_in_comp = [int(value) for value in np.unique(dust_labels[component]) if value > 0]
        if feature.get("outside_roi_halo"):
            outer_selected.update(ids_in_comp)
        else:
            selected.update(ids_in_comp)
        # Preserve unverified pixels even inside an oversized OMIT component.
        # In particular, do not fill a contour's holes or a neighboring faint spot.
        edit |= component & (dust > 0)

    # 若 ROI 內部已確認（例如 Region#1），但同時存在緊貼 ROI 邊界、從 ROI 外跨界暈入的熱度，
    # 將該外側鄰近灰塵核心一併納入 edit_mask，避免外圍灰塵在整 ROI 複檢時仍留下暈圈分數。
    active_img = cv2.resize(active.astype(np.uint8), (width, height),
                            interpolation=cv2.INTER_NEAREST) > 0
    inner_edge = allowed & (cv2.erode(allowed.astype(np.uint8), np.ones((5, 5), np.uint8)) == 0)
    outer_collar = np.zeros((height, width), bool)
    if has_real_omit and np.any(active_img & inner_edge):
        border_halo_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (73, 73))
        outer_collar = (
            cv2.dilate((active_img & inner_edge).astype(np.uint8), border_halo_kernel) > 0
        ) & ~allowed
        if np.any(outer_collar & (dust > 0)) and np.count_nonzero(outer_collar & (dust == 0)) >= 16:
            bg_out = residual[outer_collar & (dust == 0)]
            med_out = float(np.median(bg_out))
            noise_out = 1.4826 * float(np.median(np.abs(bg_out - med_out)))
            thr_out = max(2.0, 4.0 * noise_out)
            ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            for sign in (-1, 1):
                bin_out = (sign * residual >= thr_out).astype(np.uint8)
                o_cnt, o_labels, o_stats, _ = cv2.connectedComponentsWithStats(bin_out, connectivity=8)
                for fid in np.unique(o_labels[outer_collar]):
                    if fid <= 0 or fid >= o_cnt:
                        continue
                    x, y, w, h, area = (int(v) for v in o_stats[fid])
                    if area < 3 or x < 2 or y < 2 or x + w > width - 2 or y + h > height - 2:
                        continue
                    comp = o_labels == fid
                    if np.any(comp & allowed):
                        continue
                    ring_out = (cv2.dilate(comp.astype(np.uint8), ring_kernel) > 0) & ~comp & ~allowed
                    if not np.any(ring_out):
                        continue
                    lc = sign * (float(np.mean(smooth[comp])) - float(np.median(smooth[ring_out])))
                    if lc < thr_out * 0.5:
                        continue
                    ov = int(np.count_nonzero(comp & (dust > 0)))
                    if ov / area >= 0.80:
                        res_out = comp & (dust == 0)
                        rc, _, rs, _ = cv2.connectedComponentsWithStats(res_out.astype(np.uint8), connectivity=8)
                        if not any(rs[idx, cv2.CC_STAT_AREA] >= 3 for idx in range(1, rc)):
                            outer_selected.update(int(v) for v in np.unique(dust_labels[comp]) if v > 0)
                            edit |= comp & (dust > 0)

    if not selected and not outer_selected:
        return None, features, "missing_surface_evidence"
    if selected:
        surface = np.isin(dust_labels, list(selected - outer_selected))
        if np.any(surface) and (np.any(surface & ~allowed) or np.any(surface[:2]) or np.any(surface[-2:])
                or np.any(surface[:, :2]) or np.any(surface[:, -2:])):
            return None, features, "surface_mask_clipped"
    if outer_selected:
        outer_surface = np.isin(dust_labels, list(outer_selected))
        if (np.any(outer_surface[:2]) or np.any(outer_surface[-2:])
                or np.any(outer_surface[:, :2]) or np.any(outer_surface[:, -2:])):
            return None, features, "surface_mask_clipped"
    ring = (cv2.dilate(edit.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0) & ~edit & (allowed | outer_collar)
    if np.count_nonzero(ring) < 16:
        return None, features, "insufficient_background"
    return edit.astype(np.uint8) * 255, features, "confirmed_surface"


def surface_removed_images(image, edit_mask):
    """Create two independent background fills; preserve every other input pixel."""
    source = np.asarray(image)
    mask = np.asarray(edit_mask, dtype=np.uint8)
    if source.shape[:2] != mask.shape or not np.any(mask):
        raise ValueError("invalid surface edit mask")
    for name, method in (("telea", cv2.INPAINT_TELEA), ("navier_stokes", cv2.INPAINT_NS)):
        channels = [source] if source.ndim == 2 else cv2.split(source)
        filled = [cv2.inpaint(channel.astype(np.float32), mask, 3, method) for channel in channels]
        candidate = filled[0] if source.ndim == 2 else np.stack(filled, axis=-1)
        if not np.isfinite(candidate).all():
            raise ValueError("non-finite background fill")
        if np.issubdtype(source.dtype, np.integer):
            bounds = np.iinfo(source.dtype)
            candidate = np.clip(np.rint(candidate), bounds.min, bounds.max)
        result = source.copy()
        result[mask > 0] = candidate.astype(source.dtype)[mask > 0]
        yield name, result
