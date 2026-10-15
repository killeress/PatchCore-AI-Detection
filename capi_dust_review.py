"""Verify low-coverage AOI heat regions against physical image features.

Unlike the legacy two-stage check, this review starts from an NG decision.
An empty or incomplete observation must therefore never suppress a region.
"""

import cv2
import numpy as np


def review_low_coverage_regions(
    original_image,
    precise_dust_mask,
    anomaly_map,
    region_labels,
    region_details,
    *,
    coverage_threshold,
    evaluation_mask=None,
):
    """Return copied region decisions and feature evidence in image coordinates.

    Coverage remains the original heatmap metric. Only low-coverage NG regions
    with direct OMIT evidence can be reconsidered, and every visible feature
    in such a region must match the unextended OMIT core by at least 80%.
    """
    regions = [dict(region) for region in region_details]
    candidates = [region for region in regions if (
        not region["is_dust"]
        and 0 < region.get("dust_overlap", 0)
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

    # Suppress the LCD pixel lattice before measuring local dark/bright contrast.
    # A robust noise estimate avoids letting one strong dust spot hide a weaker
    # independent defect, as a percentile of positive contrast would do.
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    residual = smooth - cv2.GaussianBlur(smooth, (31, 31), 0)
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    features = []
    for region in candidates:
        zone = (labels_tile == region["label_id"]) & allowed
        region["feature_reviewed"] = True
        region["feature_review_reason"] = "no_matched_features"
        if not np.any(zone) or not np.any(zone & dust):
            continue
        background = residual[zone & ~dust]
        if background.size < 16:
            continue
        median = float(np.median(background))
        noise = 1.4826 * float(np.median(np.abs(background - median)))
        contrast_threshold = max(2.0, 4.0 * noise)
        region_features = []
        for sign, spot_type in ((-1, "dark"), (1, "bright")):
            # Components are extracted before ROI clipping: a cut component is
            # unresolved, rather than a smaller artificial match at the ROI edge.
            binary = (sign * residual >= contrast_threshold).astype(np.uint8)
            count, component_labels, stats, _ = cv2.connectedComponentsWithStats(
                binary, connectivity=8)
            intersecting = np.unique(component_labels[zone])
            for feature_id in intersecting:
                if feature_id == 0 or feature_id >= count:
                    continue
                x, y, w, h, area = (int(value) for value in stats[feature_id])
                if area < 3:
                    continue
                component = component_labels == feature_id
                clipped = bool(x < 2 or y < 2 or x + w > width - 2
                               or y + h > height - 2 or np.any(component & ~allowed))
                ring = (cv2.dilate(component.astype(np.uint8), ring_kernel) > 0) & ~component
                ring &= allowed
                # Gaussian background subtraction can produce an opposite-sign
                # halo around a spot. Require contrast in the image itself too.
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
                ys, xs = np.where(component)
                contours, _ = cv2.findContours(component.astype(np.uint8),
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour = max(contours, key=cv2.contourArea).reshape(-1, 2).tolist()
                feature = {
                    "zone_id": region["label_id"],
                    "abs_pos": (int(np.mean(xs)), int(np.mean(ys))),
                    "area": area,
                    "type": spot_type,
                    "feature_bbox": (x, y, w, h),
                    "feature_contour": contour,
                    "dust_overlap": overlap,
                    "dust_ratio": ratio,
                    "is_dust": matched,
                    "dust_reason": "precise_overlap" if matched else "clipped" if clipped else "unmatched",
                    "unmatched_area": int(np.count_nonzero(significant_residual)),
                    "contrast": local_contrast,
                    "contrast_threshold": contrast_threshold,
                }
                if not matched:
                    real_support = significant_residual & zone
                    if not np.any(real_support):
                        real_support = component & zone
                    ry, rx = np.unravel_index(
                        np.argmax(np.where(real_support, sign * residual, -np.inf)), gray.shape)
                    feature["real_pos"] = (int(rx), int(ry))
                region_features.append(feature)
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
