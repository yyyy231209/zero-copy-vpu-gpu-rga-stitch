"""Four-camera offline/live panorama with cylindrical projection and feather blending.

Offline test: python stitch_360.py --images cam_images --image-index 11
Live preview:  python stitch_360.py --live
"""
from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np

FLIP_HORIZONTAL = False  # Preserve raw orientation; calibrations use the same convention.


def load_pair_homography(source: int, target: int, directory: Path) -> np.ndarray:
    direct = directory / f"stitch_params_{source}{target}.npz"
    reverse = directory / f"stitch_params_{target}{source}.npz"
    if direct.exists():
        return np.load(direct)["H"].astype(np.float64)
    if reverse.exists():
        return np.linalg.inv(np.load(reverse)["H"].astype(np.float64))
    raise FileNotFoundError(f"Need {direct.name} (or the reverse matrix) in {directory}")


def build_warps(order: list[int], image_size: tuple[int, int], params: Path):
    width, height = image_size
    anchor_index = len(order) // 2
    transforms = [np.eye(3, dtype=np.float64) for _ in order]
    # Build every camera transform into the middle camera coordinate system.
    for i in range(anchor_index - 1, -1, -1):
        transforms[i] = transforms[i + 1] @ load_pair_homography(order[i], order[i + 1], params)
    for i in range(anchor_index + 1, len(order)):
        transforms[i] = transforms[i - 1] @ np.linalg.inv(load_pair_homography(order[i - 1], order[i], params))
    corners = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]).reshape(-1, 1, 2)
    projected = np.concatenate([cv2.perspectiveTransform(corners, H) for H in transforms])
    xmin, ymin = np.floor(projected.reshape(-1, 2).min(axis=0)).astype(int)
    xmax, ymax = np.ceil(projected.reshape(-1, 2).max(axis=0)).astype(int)
    canvas = (int(xmax - xmin + 1), int(ymax - ymin + 1))
    if not (0 < canvas[0] <= 8000 and 0 < canvas[1] <= 4000):
        raise ValueError(f"Unreasonable canvas {canvas}; recalibrate the adjacent camera matrices.")
    translate = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float64)
    return [translate @ H for H in transforms], canvas


def feather_blend(images: list[np.ndarray], warps: list[np.ndarray], canvas: tuple[int, int]) -> np.ndarray:
    accum = np.zeros((canvas[1], canvas[0], 3), np.float32)
    weights = np.zeros((canvas[1], canvas[0]), np.float32)
    source_mask = np.full(images[0].shape[:2], 255, np.uint8)
    for image, H in zip(images, warps):
        warped = cv2.warpPerspective(image, H, canvas, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mask = cv2.warpPerspective(source_mask, H, canvas, flags=cv2.INTER_NEAREST) > 0
        # Distance-to-edge weights hide seams without treating genuine black pixels as holes.
        distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        weight = np.minimum(distance, 80.0) * mask
        accum += warped.astype(np.float32) * weight[..., None]
        weights += weight
    result = accum / np.maximum(weights[..., None], 1e-6)
    valid = weights > 0
    ys, xs = np.where(valid)
    return result[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)


def build_fisheye_maps(images: list[np.ndarray], fov_degrees: float, output_width: int,
                       yaw_offsets_degrees: list[float], calibrations: list[dict]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Map panorama pixels straight into raw fisheye pixels; no crop is applied."""
    height, width = images[0].shape[:2]
    focal_output = output_width / (2.0 * np.pi)
    theta = (np.arange(output_width, dtype=np.float64) / focal_output) - np.pi
    vertical = (np.arange(height, dtype=np.float64) - (height - 1) / 2.0) / focal_output
    theta_grid, vertical_grid = np.meshgrid(theta, vertical)
    half_fov, blend_width = np.deg2rad(fov_degrees) / 2.0, np.deg2rad(6.0)
    maps = []
    for index, calibration in enumerate(calibrations):
        yaw = -np.pi + index * (np.pi / 2.0) + np.deg2rad(yaw_offsets_degrees[index])
        delta = np.arctan2(np.sin(theta_grid - yaw), np.cos(theta_grid - yaw))
        # Unit-ray coordinates in a virtual pinhole view, transformed directly
        # to the distorted RAW fisheye image by the measured K/D model.
        points = np.dstack((np.tan(delta), vertical_grid / np.cos(delta))).reshape(-1, 1, 2)
        raw = cv2.fisheye.distortPoints(points, calibration["K"], calibration["D"]).reshape(height, output_width, 2)
        map_x, map_y = raw[..., 0].astype(np.float32), raw[..., 1].astype(np.float32)
        valid = (np.abs(delta) < half_fov) & (map_x >= 0) & (map_x < width - 1) & (map_y >= 0) & (map_y < height - 1)
        weight = np.clip((np.pi / 4.0 + blend_width / 2.0 - np.abs(delta)) / blend_width, 0.0, 1.0) * valid
        maps.append((map_x, map_y, weight.astype(np.float32)))
    return maps


def render_precomputed_maps(images: list[np.ndarray], maps: list[tuple[np.ndarray, np.ndarray, np.ndarray]], multiband: bool = False) -> np.ndarray:
    """Blend images using maps of any output width (used by the open chain)."""
    height, width = maps[0][0].shape
    accumulated = np.zeros((height, width, 3), np.float32)
    total_weight = np.zeros((height, width), np.float32)
    warped_images, weights = [], []
    for image, (map_x, map_y, weight) in zip(images, maps):
        warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        warped_images.append(warped.astype(np.float32) / 255.0)
        weights.append(weight.astype(np.float32))
        accumulated += warped.astype(np.float32) * weight[..., None]
        total_weight += weight
    if multiband:
        # Pyramid blending smooths low-frequency exposure transitions while
        # retaining high-frequency lines and texture near a narrow seam.
        levels = 4
        image_pyramids, weight_pyramids = [], []
        for image, weight in zip(warped_images, weights):
            gp_image, gp_weight = [image], [weight]
            for _ in range(levels - 1):
                gp_image.append(cv2.pyrDown(gp_image[-1]))
                gp_weight.append(cv2.pyrDown(gp_weight[-1]))
            laplacian = []
            for level in range(levels - 1):
                up = cv2.pyrUp(gp_image[level + 1], dstsize=(gp_image[level].shape[1], gp_image[level].shape[0]))
                laplacian.append(gp_image[level] - up)
            laplacian.append(gp_image[-1])
            image_pyramids.append(laplacian)
            weight_pyramids.append(gp_weight)
        blended_levels = []
        for level in range(levels):
            total = sum(pyramid[level] for pyramid in weight_pyramids)
            blended = sum(image_pyramids[i][level] * (weight_pyramids[i][level] / np.maximum(total, 1e-6))[..., None] for i in range(len(images)))
            blended_levels.append(blended)
        result = blended_levels[-1]
        for level in range(levels - 2, -1, -1):
            result = cv2.pyrUp(result, dstsize=(blended_levels[level].shape[1], blended_levels[level].shape[0])) + blended_levels[level]
        return np.clip(result * 255.0, 0, 255).astype(np.uint8)
    return (accumulated / np.maximum(total_weight[..., None], 1e-6)).astype(np.uint8)


def estimate_exposure_gains(images: list[np.ndarray], maps: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    """Estimate fixed BGR gains from the three already-aligned overlap bands."""
    warped, masks = [], []
    for image, (map_x, map_y, weight) in zip(images, maps):
        warped.append(cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(np.float32))
        masks.append(weight > 0)
    gains = np.ones((len(images), 3), np.float32)
    for index in range(len(images) - 1):
        overlap = masks[index] & masks[index + 1]
        if overlap.sum() < 200:
            continue
        reference = np.median(warped[index][overlap] * gains[index], axis=0)
        measured = np.median(warped[index + 1][overlap], axis=0)
        # Do not propagate a bad overlap estimate through the whole chain.
        # A modest correction is preferable to clipping highlights in live use.
        gains[index + 1] = np.clip(gains[index] * np.clip(reference / np.maximum(measured, 1.0), 0.85, 1.18), 0.80, 1.25)
    return gains


def apply_exposure_gains(images: list[np.ndarray], gains: np.ndarray) -> list[np.ndarray]:
    return [np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8) for image, gain in zip(images, gains)]


def annotate_open_chain_sources(image: np.ndarray, order: list[int]) -> np.ndarray:
    """Diagnostic labels only; proves where each raw camera contributes."""
    marked = image.copy()
    # Open-chain span is 420 degrees: camera centres are at 75, 165, 255, 345.
    for index, camera_id in enumerate(order):
        x = int((75.0 + index * 90.0) / 420.0 * marked.shape[1])
        cv2.putText(marked, f"cam{camera_id}", (x - 38, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.line(marked, (x, 42), (x, min(70, marked.shape[0] - 1)), (0, 255, 0), 2)
    return marked


def load_open_chain_rotations(directory: Path, camera_order: list[int]) -> list[np.ndarray]:
    """Return camera-to-cam2 rotations for the calibrated 2->1->4->3 chain."""
    if camera_order != [2, 1, 4, 3]:
        raise ValueError("Full-R open-chain projection currently requires --order 2 1 4 3")
    r12 = np.load(directory / "extrinsic_cam12.npz")["R"]  # cam1 -> cam2
    r14 = np.load(directory / "extrinsic_cam14.npz")["R"]  # cam1 -> cam4
    r34 = np.load(directory / "extrinsic_cam34.npz")["R"]  # cam3 -> cam4
    r32 = np.load(directory / "extrinsic_cam32.npz")["R"]  # cam3 -> cam2
    rotations = [np.eye(3), r12, r12 @ r14.T, r12 @ r14.T @ r34]
    # The fourth independent pair closes the chain.  A large disagreement
    # means the rig moved between captures and must be recalibrated.
    closure_error = np.rad2deg(np.arccos(np.clip((np.trace(rotations[3].T @ r32) - 1.0) / 2.0, -1.0, 1.0)))
    print(f"Full-R chain closure error: {closure_error:.3f} deg")
    return rotations


def build_open_chain_maps(images: list[np.ndarray], fov_degrees: float, panorama_width: int,
                          calibrations: list[dict], rotations: list[np.ndarray] | None = None) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Open 2->1->4->3 projection; full extrinsics are optional experimental input."""
    height, width = images[0].shape[:2]
    half_fov = np.deg2rad(fov_degrees) / 2.0
    focal_output = panorama_width / (2.0 * np.pi)  # retain the 360-mode pixel density
    if rotations is None:
        # Stable baseline: exact 90-degree yaw-only camera arrangement.
        rotations = []
        for angle in np.arange(len(images)) * (np.pi / 2.0):
            c, s = np.cos(angle), np.sin(angle)
            rotations.append(np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]))
    forward = np.array([0.0, 0.0, 1.0])
    headings = np.array([np.arctan2((G @ forward)[0], (G @ forward)[2]) for G in rotations])
    # Unwrap headings in the declared open-chain order.
    for i in range(1, len(headings)):
        while headings[i] <= headings[i - 1]:
            headings[i] += 2.0 * np.pi
    start, end = headings[0] - half_fov, headings[-1] + half_fov
    span = end - start
    output_width = int(round(span * focal_output))
    theta = np.arange(output_width, dtype=np.float64) / focal_output + start
    vertical = (np.arange(height, dtype=np.float64) - (height - 1) / 2.0) / focal_output
    theta_grid, vertical_grid = np.meshgrid(theta, vertical)
    feather = np.deg2rad(6.0)
    maps = []
    for index, (calibration, G) in enumerate(zip(calibrations, rotations)):
        world_rays = np.dstack((np.sin(theta_grid), vertical_grid, np.cos(theta_grid)))
        local_rays = np.einsum("ij,...j->...i", G.T, world_rays)
        z = local_rays[..., 2]
        safe_z = np.where(np.abs(z) > 1e-5, z, 1.0)
        normalized = np.dstack((local_rays[..., 0] / safe_z, local_rays[..., 1] / safe_z))
        normalized = np.clip(normalized, -1e4, 1e4)
        points = normalized.reshape(-1, 1, 2)
        raw = cv2.fisheye.distortPoints(points, calibration["K"], calibration["D"]).reshape(height, output_width, 2)
        map_x, map_y = raw[..., 0].astype(np.float32), raw[..., 1].astype(np.float32)
        local_heading = np.arctan2(local_rays[..., 0], z)
        valid = (z > 1e-5) & (np.abs(local_heading) < half_fov) & (map_x >= 0) & (map_x < width - 1) & (map_y >= 0) & (map_y < height - 1)
        # Keep the complete valid map.  Seam weights are applied afterwards so
        # a seam can move anywhere inside the true 60-degree overlap.
        maps.append((map_x, map_y, valid.astype(np.float32)))
    return maps


def default_open_chain_seams(width: int, count: int) -> list[int]:
    """Default internal seams for a 150-degree, 90-degree-spaced open chain."""
    return [int(round((120.0 + index * 90.0) / 420.0 * width)) for index in range(count - 1)]


def apply_open_chain_seams(maps: list[tuple[np.ndarray, np.ndarray, np.ndarray]], seams: list[int], feather_px: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Turn valid masks into mutually complementary narrow seam weights."""
    height, width = maps[0][0].shape
    x = np.arange(width, dtype=np.float32)[None, :]
    weighted = []
    for index, (map_x, map_y, valid) in enumerate(maps):
        weight = valid.copy()
        if index > 0:
            weight *= np.clip((x - seams[index - 1] + feather_px / 2.0) / max(feather_px, 1), 0.0, 1.0)
        if index < len(maps) - 1:
            weight *= np.clip((seams[index] + feather_px / 2.0 - x) / max(feather_px, 1), 0.0, 1.0)
        weighted.append((map_x, map_y, weight.astype(np.float32)))
    return weighted


def find_open_chain_seams(images: list[np.ndarray], maps: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> list[int]:
    """Find a stable low-difference vertical seam near each nominal overlap centre."""
    height, width = maps[0][0].shape
    warped, masks = [], []
    for image, (map_x, map_y, valid) in zip(images, maps):
        warped.append(cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
        masks.append(valid > 0)
    nominal = default_open_chain_seams(width, len(images))
    search_half_width = max(8, int(round(20.0 / 420.0 * width)))  # search ±20 degrees
    seams = []
    for index, center in enumerate(nominal):
        left_gray = cv2.GaussianBlur(cv2.cvtColor(warped[index], cv2.COLOR_BGR2GRAY), (5, 5), 0)
        right_gray = cv2.GaussianBlur(cv2.cvtColor(warped[index + 1], cv2.COLOR_BGR2GRAY), (5, 5), 0)
        difference = cv2.absdiff(left_gray, right_gray).astype(np.float32)
        candidates = range(max(0, center - search_half_width), min(width, center + search_half_width + 1))
        best_x, best_score = center, float("inf")
        for x in candidates:
            overlap = masks[index][:, x] & masks[index + 1][:, x]
            if overlap.sum() < max(30, height // 8):
                continue
            # Median makes a single person or chair crossing the seam less dominant.
            score = float(np.median(difference[:, x][overlap]))
            if score < best_score:
                best_x, best_score = x, score
        seams.append(best_x)
    return seams


def estimate_local_seam_shifts(images: list[np.ndarray], maps: list[tuple[np.ndarray, np.ndarray, np.ndarray]], seams: list[int]) -> list[tuple[float, float]]:
    """Estimate a small output-space translation for the right image at each seam.

    This is deliberately limited to a narrow overlap strip.  It corrects the
    residual projection error without trying to force nearby objects (which
    have real stereo parallax because the cameras have different centres) to
    line up across the complete panorama.
    """
    height, width = maps[0][0].shape
    warped, masks = [], []
    for image, (map_x, map_y, valid) in zip(images, maps):
        warped.append(cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
        masks.append(valid > 0)
    half_width = max(20, int(round(12.0 / 420.0 * width)))
    y0, y1 = int(height * 0.16), int(height * 0.84)
    shifts = []
    for index, seam in enumerate(seams):
        x0, x1 = max(0, seam - half_width), min(width, seam + half_width)
        left = cv2.cvtColor(warped[index][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        right = cv2.cvtColor(warped[index + 1][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        overlap = masks[index][y0:y1, x0:x1] & masks[index + 1][y0:y1, x0:x1]
        if overlap.sum() < (x1 - x0) * max(12, (y1 - y0) // 5):
            shifts.append((0.0, 0.0)); continue
        # Gradient magnitude is much less sensitive than brightness to the
        # different auto-exposure settings of inexpensive USB cameras.
        left = cv2.magnitude(cv2.Sobel(left, cv2.CV_32F, 1, 0), cv2.Sobel(left, cv2.CV_32F, 0, 1)) * overlap
        right = cv2.magnitude(cv2.Sobel(right, cv2.CV_32F, 1, 0), cv2.Sobel(right, cv2.CV_32F, 0, 1)) * overlap
        window = cv2.createHanningWindow((x1 - x0, y1 - y0), cv2.CV_32F)
        proposed, response = cv2.phaseCorrelate(left, right, window)
        proposed = np.asarray(proposed, np.float32)
        # A correction larger than four panorama pixels is not a residual
        # calibration error here; it is almost always parallax or a moving
        # object.  Reject it instead of clipping it into a misleading shift.
        if np.max(np.abs(proposed)) > 4.0:
            shifts.append((0.0, 0.0)); continue

        # phaseCorrelate's sign convention is easy to confuse.  Evaluate both
        # signs in image space and retain only a measurable improvement.
        def score(shift: np.ndarray) -> float:
            matrix = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
            moved = cv2.warpAffine(right, matrix, (right.shape[1], right.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            moved_mask = cv2.warpAffine(overlap.astype(np.uint8), matrix, (right.shape[1], right.shape[0]), flags=cv2.INTER_NEAREST) > 0
            valid = overlap & moved_mask
            return float(np.mean(np.abs(left[valid] - moved[valid]))) if valid.sum() > 300 else float("inf")
        baseline = score(np.zeros(2, np.float32))
        candidates = (proposed, -proposed)
        best = min(candidates, key=score)
        improved = score(best) < baseline * 0.95
        shifts.append((float(best[0]), float(best[1])) if response >= 0.08 and improved else (0.0, 0.0))
    return shifts


def apply_local_seam_shifts(maps: list[tuple[np.ndarray, np.ndarray, np.ndarray]], seams: list[int], shifts: list[tuple[float, float]]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Apply each estimated shift only in a 12-degree band beside its seam."""
    height, width = maps[0][0].shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    half_width = max(20, int(round(12.0 / 420.0 * width)))
    adjusted = [(mx.copy(), my.copy(), valid.copy()) for mx, my, valid in maps]
    for index, (seam, shift) in enumerate(zip(seams, shifts)):
        dx, dy = shift
        if abs(dx) < 0.05 and abs(dy) < 0.05:
            continue
        # The right-hand camera owns this correction.  A cosine-like ramp
        # makes it disappear before reaching that camera's next seam.
        alpha = np.clip(1.0 - np.abs(grid_x - seam) / half_width, 0.0, 1.0).astype(np.float32)
        sample_x, sample_y = grid_x - alpha * dx, grid_y - alpha * dy
        map_x, map_y, valid = adjusted[index + 1]
        shifted_x = cv2.remap(map_x, sample_x, sample_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        shifted_y = cv2.remap(map_y, sample_x, sample_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        shifted_valid = cv2.remap(valid, sample_x, sample_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        region = alpha > 0
        map_x[region], map_y[region], valid[region] = shifted_x[region], shifted_y[region], shifted_valid[region]
        adjusted[index + 1] = (map_x, map_y, valid)
    return adjusted


def load_fisheye_calibrations(directory: Path, image_size: tuple[int, int], camera_order: list[int]) -> list[dict] | None:
    calibrations = []
    for cam in camera_order:
        path = directory / f"calib_cam{cam}.npz"
        if not path.exists():
            return None
        data = np.load(path)
        if tuple(data["image_size"]) != image_size:
            raise ValueError(f"{path.name} expects {tuple(data['image_size'])}, input is {image_size}")
        calibrations.append({"K": data["K"], "D": data["D"]})
    return calibrations


def cylindrical_panorama(
    images: list[np.ndarray],
    fov_degrees: float,
    output_width: int,
    yaw_offsets_degrees: list[float] | None = None,
    blend_degrees: float = 6.0,
    fisheye_maps: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    crop_common_vertical: bool = True,
    vertical_coverage: float = 0.995,
) -> np.ndarray:
    """Project four outward-facing cameras, 90 degrees apart, onto a 360-degree cylinder.

    This model is stable for a 150-degree lens.  A planar homography is only a
    local approximation and becomes singular near the very wide image edges.
    """
    height, width = images[0].shape[:2]
    focal_input = width / (2.0 * np.tan(np.deg2rad(fov_degrees) / 2.0))
    focal_output = output_width / (2.0 * np.pi)
    theta = (np.arange(output_width, dtype=np.float32) / focal_output) - np.pi
    y_out = np.arange(height, dtype=np.float32) - (height - 1) / 2.0
    theta_grid, y_grid = np.meshgrid(theta, y_out)
    accumulated = np.zeros((height, output_width, 3), np.float32)
    total_weight = np.zeros((height, output_width), np.float32)
    half_fov = np.deg2rad(fov_degrees) / 2.0
    # Four cameras 90 degrees apart have 60 degrees of overlap for 150-degree
    # lenses. The overlap contains the same scene twice. With cameras mounted
    # 15 cm apart, blending all 60 degrees creates severe double images from
    # parallax, so use a narrow transition by default and let one camera own
    # the rest of the redundant overlap. This does *not* remove any direction
    # from the 360-degree panorama; it merely chooses the sharper copy.
    available_overlap = np.deg2rad(fov_degrees - 90.0)
    if available_overlap <= 0:
        raise ValueError("Camera FOV must be greater than 90 degrees for four-camera coverage.")
    blend_width = np.deg2rad(blend_degrees)
    if not 0 < blend_width <= available_overlap + 1e-6:
        raise ValueError(f"--blend-width must be in (0, {np.rad2deg(available_overlap):.1f}] degrees.")
    yaw_offsets = yaw_offsets_degrees or [0.0] * len(images)
    if len(yaw_offsets) != len(images):
        raise ValueError("--yaw-offsets needs one value for each camera.")
    for camera_index, image in enumerate(images):
        if fisheye_maps is not None:
            map_x, map_y, weight = fisheye_maps[camera_index]
            warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            accumulated += warped.astype(np.float32) * weight[..., None]
            total_weight += weight
            continue
        yaw = -np.pi + camera_index * (np.pi / 2.0) + np.deg2rad(yaw_offsets[camera_index])
        delta = np.arctan2(np.sin(theta_grid - yaw), np.cos(theta_grid - yaw))
        tangent = np.tan(delta)
        map_x = (width - 1) / 2.0 + focal_input * tangent
        # Inverse cylindrical projection; keeps vertical lines straight.
        map_y = (height - 1) / 2.0 + y_grid * np.sqrt(1.0 + tangent * tangent) * focal_input / focal_output
        valid = (np.abs(delta) < half_fov) & (map_x >= 0) & (map_x < width - 1) & (map_y >= 0) & (map_y < height - 1)
        warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # Each camera owns its central 90 degrees. Around each 45-degree seam,
        # a small feather transition avoids a hard cut while limiting ghosting.
        ownership = np.abs(delta)
        seam_half_width = blend_width / 2.0
        weight = np.clip((np.pi / 4.0 + seam_half_width - ownership) / blend_width, 0.0, 1.0) * valid
        accumulated += warped.astype(np.float32) * weight[..., None]
        total_weight += weight
    result = (accumulated / np.maximum(total_weight[..., None], 1e-6)).astype(np.uint8)
    # A rectangular 360 panorama can only retain the vertical field shared by
    # every azimuth. Remove rows that are mostly outside a circular fisheye's
    # valid image area; horizontal 360-degree coverage is left untouched.
    complete_rows = np.count_nonzero(total_weight > 0, axis=1) >= int(vertical_coverage * output_width)
    if crop_common_vertical and np.any(complete_rows):
        first, last = np.flatnonzero(complete_rows)[[0, -1]]
        result = result[first:last + 1]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="cam_images")
    parser.add_argument("--params", default=".")
    parser.add_argument("--order", nargs=4, type=int, default=[2, 1, 4, 3], help="physical adjacent camera order, confirmed as 2 -> 1 -> 4 -> 3")
    parser.add_argument("--devices", nargs=4, type=int, metavar=("DEV_CAM2", "DEV_CAM1", "DEV_CAM4", "DEV_CAM3"), help="capture-device indices in --order sequence; use this on RK3588/Linux, e.g. --devices 0 1 2 3")
    parser.add_argument("--image-index", type=int, default=11, help="offline image number, e.g. 11 -> 0011.jpg")
    parser.add_argument("--output", default="panorama.jpg")
    parser.add_argument("--fov", type=float, default=150.0, help="horizontal camera FOV in degrees")
    parser.add_argument("--panorama-width", type=int, default=2048)
    parser.add_argument("--start-angle", type=float, default=0.0, help="global angle placed at the left edge, in degrees; changes only the 360 panorama cut position")
    parser.add_argument("--keep-vertical", action="store_true", help="keep the complete vertical lens view; uncovered top/bottom regions remain black")
    parser.add_argument("--open-chain", action="store_true", help="open 2->1->4->3 panorama: do not stitch camera 3 back to camera 2; preserves the full endpoints")
    parser.add_argument("--auto-seams", action="store_true", help="choose low-difference seams in the three open-chain overlap regions at startup")
    parser.add_argument("--local-align", action="store_true", help="apply small, reference-derived local alignment corrections at the three open-chain seams")
    parser.add_argument("--local-align-file", help="load checkerboard-derived seam shifts from calibrate_local_pairs.py")
    parser.add_argument("--alignment-images", help="static reference root for --local-align (for live mode, normally cam_images)")
    parser.add_argument("--alignment-index", type=int, help="static reference number for --local-align, e.g. 50 -> 0050.jpg")
    parser.add_argument("--exposure-compensation", action="store_true", help="estimate fixed BGR gains from overlap bands at startup")
    parser.add_argument("--multiband", action="store_true", help="use multi-band blending in open-chain mode")
    parser.add_argument("--full-rotation", action="store_true", help="experimental: use complete extrinsic R matrices; default uses the visually steadier yaw-only model")
    parser.add_argument("--show-sources", action="store_true", help="in live mode, display the four labelled raw camera feeds for verification")
    parser.add_argument("--show-camera-bands", action="store_true", help="label the primary source regions in open-chain output (diagnostic only)")
    parser.add_argument("--display-width", type=int, default=1600, help="maximum live preview width; does not change saved output resolution")
    parser.add_argument("--vertical-coverage", type=float, default=0.995, help="fraction of panorama width that must contain data to keep a row (0-1; lower value crops less)")
    parser.add_argument("--no-fisheye-calib", action="store_true", help="ignore calib_cam*.npz and use the approximate lens model")
    parser.add_argument("--blend-width", type=float, default=6.0, help="cross-fade width in degrees (default: 6; do not use the full 60-degree overlap with a 15 cm camera baseline)")
    parser.add_argument("--yaw-offsets", nargs=4, type=float, default=[0.0, -1.54, -0.06, 0.26], metavar=("CAM2", "CAM1", "CAM4", "CAM3"), help="per-camera yaw corrections in degrees, in --order order; measured from the final 2->1->4->3 extrinsic calibration")
    parser.add_argument("--homography", action="store_true", help="use planar H matrices (only for narrow-FOV cameras)")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if not 0 < args.vertical_coverage <= 1:
        raise SystemExit("--vertical-coverage must be in (0, 1]")
    if args.display_width < 320:
        raise SystemExit("--display-width must be at least 320")
    if not -180 <= args.start_angle < 180:
        raise SystemExit("--start-angle must be in [-180, 180)")
    index_name = f"{args.image_index:04d}.jpg"
    if args.live:
        device_indices = args.devices if args.devices is not None else args.order
        captures = [cv2.VideoCapture(device) for device in device_indices]
        for cap in captures:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ok, initial = zip(*(cap.read() for cap in captures))
        if not all(ok): raise SystemExit("Could not open all four cameras")
        print(f"Capture-device indices (cam2, cam1, cam4, cam3): {device_indices}")
    else:
        initial = [cv2.imread(str(Path(args.images) / f"cam{cam}" / index_name)) for cam in args.order]
        if any(frame is None for frame in initial): raise SystemExit(f"Missing image set {index_name}")
    if FLIP_HORIZONTAL:
        initial = [cv2.flip(frame, 1) for frame in initial]
    reference = list(initial)
    if args.alignment_images:
        reference_index = args.alignment_index if args.alignment_index is not None else args.image_index
        reference_name = f"{reference_index:04d}.jpg"
        reference = [cv2.imread(str(Path(args.alignment_images) / f"cam{cam}" / reference_name)) for cam in args.order]
        if any(frame is None for frame in reference):
            raise SystemExit(f"Missing local-alignment reference set {reference_name}")
        if FLIP_HORIZONTAL:
            reference = [cv2.flip(frame, 1) for frame in reference]
    h, w = initial[0].shape[:2]
    if any(frame.shape[:2] != (h, w) for frame in reference):
        raise SystemExit("Local-alignment reference images must have the same resolution as the live/offline input")
    color_gains = None
    if args.homography:
        warps, canvas = build_warps(args.order, (w, h), Path(args.params))
        print(f"Planar canvas: {canvas[0]}x{canvas[1]}; camera order: {args.order}")
    else:
        warps, canvas = None, (args.panorama_width, h)
        calibrations = None if args.no_fisheye_calib else load_fisheye_calibrations(Path(args.params), (w, h), args.order)
        if args.open_chain and calibrations is None:
            raise SystemExit("--open-chain requires calib_cam*.npz")
        rotations = load_open_chain_rotations(Path(args.params), args.order) if args.open_chain and args.full_rotation else None
        if calibrations is None:
            fisheye_maps = None
        elif args.open_chain:
            base_maps = build_open_chain_maps(list(initial), args.fov, args.panorama_width, calibrations, rotations)
            seams = find_open_chain_seams(reference, base_maps) if args.auto_seams else default_open_chain_seams(base_maps[0][0].shape[1], len(initial))
            if args.local_align_file:
                data = np.load(args.local_align_file)
                shifts = [tuple(float(v) for v in shift) for shift in data["shifts"]]
                if len(shifts) != len(seams):
                    raise SystemExit("Local-alignment file must contain exactly three seam shifts")
                base_maps = apply_local_seam_shifts(base_maps, seams, shifts)
                print("Open-chain checkerboard seam shifts (px): " + str([tuple(round(v, 2) for v in shift) for shift in shifts]))
            elif args.local_align:
                shifts = estimate_local_seam_shifts(reference, base_maps, seams)
                base_maps = apply_local_seam_shifts(base_maps, seams, shifts)
                print("Open-chain local seam shifts (px): " + str([tuple(round(v, 2) for v in shift) for shift in shifts]))
            feather_px = max(2, int(round(args.blend_width / 420.0 * base_maps[0][0].shape[1])))
            fisheye_maps = apply_open_chain_seams(base_maps, seams, feather_px)
            seam_degrees = [round(x / base_maps[0][0].shape[1] * 420.0 - 75.0, 1) for x in seams]
            print(f"Open-chain seams ({'auto' if args.auto_seams else 'default'}): {seam_degrees} deg")
            if args.exposure_compensation:
                color_gains = estimate_exposure_gains(list(initial), fisheye_maps)
                print(f"Open-chain BGR exposure gains: {np.round(color_gains, 3).tolist()}")
        else:
            fisheye_maps = build_fisheye_maps(list(initial), args.fov, args.panorama_width, args.yaw_offsets, calibrations)
        mode = "calibrated raw-fisheye" if fisheye_maps is not None else "approximate FOV"
        label = "Open-chain panorama" if args.open_chain else "Cylindrical 360 panorama"
        print(f"{label} ({mode}): {canvas[0]}x{canvas[1]}; camera order: {args.order}")
    if not args.live:
        blend_input = apply_exposure_gains(initial, color_gains) if color_gains is not None else initial
        result = feather_blend(blend_input, warps, canvas) if args.homography else (render_precomputed_maps(blend_input, fisheye_maps, args.multiband) if args.open_chain else cylindrical_panorama(blend_input, args.fov, args.panorama_width, args.yaw_offsets, args.blend_width, fisheye_maps, not args.keep_vertical, args.vertical_coverage))
        # The panorama is circular. Rotate its flat representation so the
        # left/right window cut does not split the 2-3 region.
        if not args.open_chain:
            roll = int(round((args.start_angle + 180.0) / 360.0 * result.shape[1]))
            result = np.roll(result, -roll, axis=1)
        elif args.show_camera_bands:
            result = annotate_open_chain_sources(result, args.order)
        cv2.imwrite(args.output, result)
        print(f"Saved {args.output}")
        return
    try:
        while True:
            ok, frames = zip(*(cap.read() for cap in captures))
            if all(ok):
                if FLIP_HORIZONTAL:
                    frames = tuple(cv2.flip(frame, 1) for frame in frames)
                if args.show_sources:
                    tiles = []
                    for camera_id, frame in zip(args.order, frames):
                        tile = cv2.resize(frame, (320, 180))
                        cv2.putText(tile, f"cam{camera_id}", (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        tiles.append(tile)
                    cv2.imshow("Raw camera sources", np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:]))))
                blend_input = apply_exposure_gains(list(frames), color_gains) if color_gains is not None else list(frames)
                result = feather_blend(blend_input, warps, canvas) if args.homography else (render_precomputed_maps(blend_input, fisheye_maps, args.multiband) if args.open_chain else cylindrical_panorama(blend_input, args.fov, args.panorama_width, args.yaw_offsets, args.blend_width, fisheye_maps, not args.keep_vertical, args.vertical_coverage))
                if not args.open_chain:
                    roll = int(round((args.start_angle + 180.0) / 360.0 * result.shape[1]))
                    result = np.roll(result, -roll, axis=1)
                elif args.show_camera_bands:
                    result = annotate_open_chain_sources(result, args.order)
                display = result
                if display.shape[1] > args.display_width:
                    scale = args.display_width / display.shape[1]
                    display = cv2.resize(display, (args.display_width, int(round(display.shape[0] * scale))), interpolation=cv2.INTER_AREA)
                cv2.imshow("Four-camera panorama", display)
            if cv2.waitKey(1) & 0xFF == 27: break
    finally:
        for cap in captures: cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
