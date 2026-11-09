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
):
    """Return copied region decisions and feature evidence in image coordinates.

    Coverage remains the original Top% metric. The pre-core, post-exclusion map
    can associate a displaced core with its connected half-peak response inside
    the ROI. Every observed feature must still match the unextended OMIT mask;
    a nearby dust patch or an empty observation cannot clear a region.
    """
    regions = [dict(region) for region in region_details]
    candidates = [region for region in regions if (
        not region["is_dust"]
        and (0 < region.get("dust_overlap", 0) or (
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
    features = []
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
                small_boundary_match = False
                if not matched and not clipped and small_labels is not None and ratio >= 0.5:
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
                    "dust_reason": ("small_core_boundary" if small_boundary_match else
                                    "precise_overlap" if matched else "clipped" if clipped else "unmatched"),
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


def confirmed_roi_surface_mask(original_image, precise_dust_mask, roi_mask, support_map, *, small_core_mask=None):
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
    allowed = cv2.resize((roi > 0).astype(np.uint8), (width, height),
                         interpolation=cv2.INTER_NEAREST) > 0
    active = (support > 0) & (cv2.resize((roi > 0).astype(np.uint8),
              (support.shape[1], support.shape[0]), interpolation=cv2.INTER_NEAREST) > 0)
    if not np.any(active) or not np.any((dust > 0) & allowed):
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
    )
    if not reviewed[0].get("is_dust"):
        return None, features, reviewed[0].get("feature_review_reason", "missing_evidence")
    _, dust_labels = cv2.connectedComponents((dust > 0).astype(np.uint8), connectivity=8)
    selected = set()
    edit = np.zeros((height, width), bool)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    residual = smooth - cv2.GaussianBlur(smooth, (31, 31), 0)
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
        selected.update(int(value) for value in np.unique(dust_labels[component]) if value > 0)
        # Preserve unverified pixels even inside an oversized OMIT component.
        # In particular, do not fill a contour's holes or a neighboring faint spot.
        edit |= component & (dust > 0)
    if not selected:
        return None, features, "missing_surface_evidence"
    surface = np.isin(dust_labels, list(selected))
    if (np.any(surface & ~allowed) or np.any(surface[:2]) or np.any(surface[-2:])
            or np.any(surface[:, :2]) or np.any(surface[:, -2:])):
        return None, features, "surface_mask_clipped"
    ring = (cv2.dilate(edit.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0) & ~edit & allowed
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
