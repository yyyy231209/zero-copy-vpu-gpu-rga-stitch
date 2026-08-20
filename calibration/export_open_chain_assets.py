"""Export deterministic RK3588 assets for the frozen open-chain panorama.

The exporter uses the same fisheye projection functions as stitch_360.py, but
stores only the even-aligned ROI needed by each camera.  It also exports
normalized Y and UV seam weights for the three frozen feather bands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from stitch_360 import (
    apply_open_chain_seams,
    build_open_chain_maps,
    load_fisheye_calibrations,
    render_precomputed_maps,
)


LOGICAL_ORDER = [2, 1, 4, 3]
INPUT_WIDTH = 1280
INPUT_HEIGHT = 720
FOV_DEGREES = 150.0
PANORAMA_DENSITY_WIDTH = 2048
VISIBLE_WIDTH = 2389
LOGICAL_NV12_WIDTH = 2390
PANORAMA_HEIGHT = 720
PANORAMA_STRIDE_ALIGN = 64
PANORAMA_STRIDE = 2432
FROZEN_SEAMS = [754, 1156, 1711]
FEATHER_PX = 34
FLOAT_DTYPE = np.dtype("<f4")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def align_down(value: int, alignment: int) -> int:
    return value // alignment * alignment


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def write_array(path: Path, array: np.ndarray, dtype: np.dtype) -> None:
    contiguous = np.ascontiguousarray(array, dtype=dtype)
    with path.open("wb") as stream:
        contiguous.tofile(stream)


def read_array(path: Path, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    array = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(shape))
    if array.size != expected:
        raise RuntimeError(
            f"{path.name}: expected {expected} elements, found {array.size}"
        )
    return array.reshape(shape)


def load_golden_frames(
    golden_directory: Path,
) -> tuple[list[np.ndarray], np.ndarray, dict]:
    manifest_path = golden_directory / "golden_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["algorithm"]["order"] != LOGICAL_ORDER:
        raise RuntimeError(
            f"golden order {manifest['algorithm']['order']} != {LOGICAL_ORDER}"
        )
    if manifest["output"]["seams_x"] != FROZEN_SEAMS:
        raise RuntimeError(
            f"golden seams {manifest['output']['seams_x']} != {FROZEN_SEAMS}"
        )
    if (
        manifest["output"]["width"],
        manifest["output"]["height"],
    ) != (VISIBLE_WIDTH, PANORAMA_HEIGHT):
        raise RuntimeError("golden panorama geometry is not 2389x720")

    devices = manifest["capture"]["devices_in_logical_order"]
    frames = []
    for logical_camera, device in zip(LOGICAL_ORDER, devices):
        path = golden_directory / f"cam{logical_camera}_video{device}.png"
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None or frame.shape != (
            INPUT_HEIGHT,
            INPUT_WIDTH,
            3,
        ):
            raise RuntimeError(f"invalid golden source image: {path}")
        frames.append(frame)

    panorama_path = golden_directory / "python_open_chain.png"
    panorama = cv2.imread(str(panorama_path), cv2.IMREAD_COLOR)
    if panorama is None or panorama.shape != (
        PANORAMA_HEIGHT,
        VISIBLE_WIDTH,
        3,
    ):
        raise RuntimeError(f"invalid golden panorama: {panorama_path}")
    return frames, panorama, manifest


def compute_seam_bands() -> list[dict]:
    half = FEATHER_PX // 2
    bands = []
    for index, seam in enumerate(FROZEN_SEAMS):
        x0 = align_down(seam - half, 2)
        x1 = align_up(seam + half, 2)
        if x0 < 0 or x1 > LOGICAL_NV12_WIDTH or x1 <= x0:
            raise RuntimeError(f"invalid seam band for seam {seam}")
        bands.append(
            {
                "index": index,
                "left_camera_index": index,
                "right_camera_index": index + 1,
                "seam_x": seam,
                "x": x0,
                "width": x1 - x0,
            }
        )
    return bands


def normalize_pair(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = left + right
    covered = total > 1e-8
    left_normalized = np.divide(
        left,
        total,
        out=np.zeros_like(left, dtype=np.float32),
        where=covered,
    )
    right_normalized = np.divide(
        right,
        total,
        out=np.zeros_like(right, dtype=np.float32),
        where=covered,
    )
    return left_normalized, right_normalized, covered


def downsample_weight_2x2(weight: np.ndarray) -> np.ndarray:
    height, width = weight.shape
    if height % 2 or width % 2:
        raise RuntimeError(f"weight shape must be even, got {weight.shape}")
    return weight.reshape(height // 2, 2, width // 2, 2).mean(
        axis=(1, 3),
        dtype=np.float32,
    )


def rebuild_weight(
    camera_index: int,
    valid: np.ndarray,
) -> np.ndarray:
    x = np.arange(VISIBLE_WIDTH, dtype=np.float32)[None, :]
    weight = valid.astype(np.float32)
    if camera_index > 0:
        weight *= np.clip(
            (
                x
                - FROZEN_SEAMS[camera_index - 1]
                + FEATHER_PX / 2.0
            )
            / FEATHER_PX,
            0.0,
            1.0,
        )
    if camera_index < len(LOGICAL_ORDER) - 1:
        weight *= np.clip(
            (
                FROZEN_SEAMS[camera_index]
                + FEATHER_PX / 2.0
                - x
            )
            / FEATHER_PX,
            0.0,
            1.0,
        )
    return weight


def image_metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    difference = actual.astype(np.int16) - expected.astype(np.int16)
    absolute = np.abs(difference)
    mse = float(np.mean(difference.astype(np.float64) ** 2))
    psnr = None if mse == 0.0 else float(10.0 * np.log10(255.0**2 / mse))
    return {
        "mae": float(np.mean(absolute)),
        "mse": mse,
        "psnr_db": psnr,
        "max_abs_error": int(np.max(absolute)),
        "different_values": int(np.count_nonzero(difference)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--camera-order",
        type=Path,
        help="confirmed camera_order.json; defaults to ../assets/camera_order.json",
    )
    args = parser.parse_args()

    golden_directory = args.golden_dir.resolve()
    output_directory = args.output_dir.resolve()
    reference_directory = args.reference_dir.resolve()
    camera_order_path = (
        args.camera_order.resolve()
        if args.camera_order is not None
        else reference_directory.parent / "assets" / "camera_order.json"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    if any(output_directory.iterdir()):
        raise RuntimeError(f"output directory must be empty: {output_directory}")

    if align_up(LOGICAL_NV12_WIDTH, PANORAMA_STRIDE_ALIGN) != PANORAMA_STRIDE:
        raise RuntimeError("panorama stride constants are inconsistent")
    camera_order = json.loads(camera_order_path.read_text(encoding="utf-8"))
    if camera_order["logical_order"] != LOGICAL_ORDER:
        raise RuntimeError("camera_order.json logical order mismatch")
    confirmed_ports = [
        item["hub_downstream_port"] for item in camera_order["cameras"]
    ]
    if confirmed_ports != [4, 3, 2, 5]:
        raise RuntimeError(
            f"camera_order.json physical ports {confirmed_ports} != [4, 3, 2, 5]"
        )

    frames, golden_panorama, golden_manifest = load_golden_frames(
        golden_directory
    )
    calibrations = load_fisheye_calibrations(
        reference_directory,
        (INPUT_WIDTH, INPUT_HEIGHT),
        LOGICAL_ORDER,
    )
    if calibrations is None:
        raise RuntimeError("missing fisheye calibration files")

    base_maps = build_open_chain_maps(
        frames,
        FOV_DEGREES,
        PANORAMA_DENSITY_WIDTH,
        calibrations,
        rotations=None,
    )
    if any(item[0].shape != (PANORAMA_HEIGHT, VISIBLE_WIDTH) for item in base_maps):
        raise RuntimeError("unexpected full map geometry")
    weighted_maps = apply_open_chain_seams(
        base_maps,
        FROZEN_SEAMS,
        FEATHER_PX,
    )

    original_render = render_precomputed_maps(
        frames,
        weighted_maps,
        multiband=False,
    )
    original_metrics = image_metrics(original_render, golden_panorama)
    if original_metrics["different_values"] != 0:
        raise RuntimeError(
            "frozen reference does not reproduce golden exactly: "
            f"{original_metrics}"
        )

    camera_assets = []
    for camera_index, (logical_camera, base_map, weighted_map) in enumerate(
        zip(LOGICAL_ORDER, base_maps, weighted_maps)
    ):
        map_x_full, map_y_full, base_valid_full = base_map
        weight_full = weighted_map[2]
        support = weight_full > 0.0
        support_columns = np.flatnonzero(np.any(support, axis=0))
        if support_columns.size == 0:
            raise RuntimeError(f"cam{logical_camera}: empty support")

        x0 = align_down(int(support_columns[0]), 2)
        x1 = align_up(int(support_columns[-1]) + 1, 2)
        x1 = min(x1, LOGICAL_NV12_WIDTH)
        roi_width = x1 - x0
        if x0 % 2 or roi_width % 2 or roi_width <= 0:
            raise RuntimeError(f"cam{logical_camera}: invalid NV12 ROI")

        map_x = np.zeros((PANORAMA_HEIGHT, roi_width), dtype=np.float32)
        map_y = np.zeros_like(map_x)
        valid = np.zeros((PANORAMA_HEIGHT, roi_width), dtype=np.uint8)
        copy_x1 = min(x1, VISIBLE_WIDTH)
        copy_width = copy_x1 - x0
        roi_support = support[:, x0:copy_x1]
        roi_valid = (
            (base_valid_full[:, x0:copy_x1] > 0.0)
            & roi_support
        )
        map_x_slice = map_x_full[:, x0:copy_x1]
        map_y_slice = map_y_full[:, x0:copy_x1]
        map_x[:, :copy_width] = np.where(roi_valid, map_x_slice, 0.0)
        map_y[:, :copy_width] = np.where(roi_valid, map_y_slice, 0.0)
        valid[:, :copy_width] = roi_valid.astype(np.uint8)

        if not np.isfinite(map_x).all() or not np.isfinite(map_y).all():
            raise RuntimeError(f"cam{logical_camera}: non-finite map value")
        selected_x = map_x[valid > 0]
        selected_y = map_y[valid > 0]
        if selected_x.size == 0:
            raise RuntimeError(f"cam{logical_camera}: no valid coordinates")
        if (
            selected_x.min() < 0.0
            or selected_x.max() >= INPUT_WIDTH - 1
            or selected_y.min() < 0.0
            or selected_y.max() >= INPUT_HEIGHT - 1
        ):
            raise RuntimeError(f"cam{logical_camera}: coordinate out of range")

        prefix = f"cam{logical_camera}"
        map_x_name = f"{prefix}_map_x.f32"
        map_y_name = f"{prefix}_map_y.f32"
        valid_name = f"{prefix}_valid.u8"
        write_array(output_directory / map_x_name, map_x, FLOAT_DTYPE)
        write_array(output_directory / map_y_name, map_y, FLOAT_DTYPE)
        write_array(output_directory / valid_name, valid, np.dtype("u1"))

        camera_assets.append(
            {
                "camera_index": camera_index,
                "logical_camera": logical_camera,
                "x_offset": x0,
                "width": roi_width,
                "height": PANORAMA_HEIGHT,
                "map_x": map_x_name,
                "map_y": map_y_name,
                "valid": valid_name,
                "valid_pixels": int(np.count_nonzero(valid)),
                "valid_ratio": float(np.mean(valid)),
                "map_x_valid_range": [
                    float(selected_x.min()),
                    float(selected_x.max()),
                ],
                "map_y_valid_range": [
                    float(selected_y.min()),
                    float(selected_y.max()),
                ],
            }
        )

    seam_assets = []
    seam_bands = compute_seam_bands()
    for band in seam_bands:
        seam_index = band["index"]
        x0 = band["x"]
        x1 = x0 + band["width"]
        left_raw = weighted_maps[seam_index][2][:, x0:x1].astype(np.float32)
        right_raw = weighted_maps[seam_index + 1][2][:, x0:x1].astype(
            np.float32
        )
        left_y, right_y, covered_y = normalize_pair(left_raw, right_raw)
        left_uv_raw = downsample_weight_2x2(left_raw)
        right_uv_raw = downsample_weight_2x2(right_raw)
        left_uv, right_uv, covered_uv = normalize_pair(
            left_uv_raw,
            right_uv_raw,
        )

        y_sum = left_y + right_y
        uv_sum = left_uv + right_uv
        if not np.allclose(y_sum[covered_y], 1.0, atol=1e-6):
            raise RuntimeError(f"seam{seam_index}: Y weights not complementary")
        if not np.allclose(uv_sum[covered_uv], 1.0, atol=1e-6):
            raise RuntimeError(f"seam{seam_index}: UV weights not complementary")

        prefix = f"seam{seam_index}"
        file_names = {
            "left_weight_y": f"{prefix}_left_y.f32",
            "right_weight_y": f"{prefix}_right_y.f32",
            "coverage_y": f"{prefix}_coverage_y.u8",
            "left_weight_uv": f"{prefix}_left_uv.f32",
            "right_weight_uv": f"{prefix}_right_uv.f32",
            "coverage_uv": f"{prefix}_coverage_uv.u8",
        }
        write_array(
            output_directory / file_names["left_weight_y"],
            left_y,
            FLOAT_DTYPE,
        )
        write_array(
            output_directory / file_names["right_weight_y"],
            right_y,
            FLOAT_DTYPE,
        )
        write_array(
            output_directory / file_names["coverage_y"],
            covered_y,
            np.dtype("u1"),
        )
        write_array(
            output_directory / file_names["left_weight_uv"],
            left_uv,
            FLOAT_DTYPE,
        )
        write_array(
            output_directory / file_names["right_weight_uv"],
            right_uv,
            FLOAT_DTYPE,
        )
        write_array(
            output_directory / file_names["coverage_uv"],
            covered_uv,
            np.dtype("u1"),
        )
        seam_assets.append(
            {
                **band,
                "height_y": PANORAMA_HEIGHT,
                "height_uv": PANORAMA_HEIGHT // 2,
                "width_uv": band["width"] // 2,
                **file_names,
                "covered_y_pixels": int(np.count_nonzero(covered_y)),
                "covered_uv_pixels": int(np.count_nonzero(covered_uv)),
                "y_weight_sum_error_max": float(
                    np.max(np.abs(y_sum[covered_y] - 1.0))
                ),
                "uv_weight_sum_error_max": float(
                    np.max(np.abs(uv_sum[covered_uv] - 1.0))
                ),
            }
        )

    body_boundaries = [
        (0, seam_bands[0]["x"]),
        (
            seam_bands[0]["x"] + seam_bands[0]["width"],
            seam_bands[1]["x"],
        ),
        (
            seam_bands[1]["x"] + seam_bands[1]["width"],
            seam_bands[2]["x"],
        ),
        (
            seam_bands[2]["x"] + seam_bands[2]["width"],
            LOGICAL_NV12_WIDTH,
        ),
    ]
    body_segments = []
    for camera_index, (nominal_x0, nominal_x1) in enumerate(body_boundaries):
        camera_roi = camera_assets[camera_index]
        camera_x0 = camera_roi["x_offset"]
        camera_x1 = camera_x0 + camera_roi["width"]
        x0 = max(nominal_x0, camera_x0)
        x1 = min(nominal_x1, camera_x1)
        width = x1 - x0
        source_x = x0 - camera_roi["x_offset"]
        if (
            width <= 0
            or x0 % 2
            or width % 2
            or source_x % 2
            or source_x < 0
            or source_x + width > camera_roi["width"]
        ):
            raise RuntimeError(f"invalid body segment {camera_index}")
        body_segments.append(
            {
                "camera_index": camera_index,
                "logical_camera": LOGICAL_ORDER[camera_index],
                "dst_x": x0,
                "src_x": source_x,
                "width": width,
                "height": PANORAMA_HEIGHT,
                "nominal_dst_x": nominal_x0,
                "nominal_width": nominal_x1 - nominal_x0,
            }
        )

    reconstructed_maps = []
    for camera_index, camera_asset in enumerate(camera_assets):
        x0 = camera_asset["x_offset"]
        width = camera_asset["width"]
        visible_copy_width = min(width, VISIBLE_WIDTH - x0)
        map_x_roi = read_array(
            output_directory / camera_asset["map_x"],
            FLOAT_DTYPE,
            (PANORAMA_HEIGHT, width),
        )
        map_y_roi = read_array(
            output_directory / camera_asset["map_y"],
            FLOAT_DTYPE,
            (PANORAMA_HEIGHT, width),
        )
        valid_roi = read_array(
            output_directory / camera_asset["valid"],
            np.dtype("u1"),
            (PANORAMA_HEIGHT, width),
        )
        map_x_full = np.zeros(
            (PANORAMA_HEIGHT, VISIBLE_WIDTH),
            dtype=np.float32,
        )
        map_y_full = np.zeros_like(map_x_full)
        valid_full = np.zeros_like(map_x_full)
        map_x_full[:, x0 : x0 + visible_copy_width] = map_x_roi[
            :, :visible_copy_width
        ]
        map_y_full[:, x0 : x0 + visible_copy_width] = map_y_roi[
            :, :visible_copy_width
        ]
        valid_full[:, x0 : x0 + visible_copy_width] = valid_roi[
            :, :visible_copy_width
        ]
        reconstructed_maps.append(
            (
                map_x_full,
                map_y_full,
                rebuild_weight(camera_index, valid_full > 0),
            )
        )

    reconstructed_render = render_precomputed_maps(
        frames,
        reconstructed_maps,
        multiband=False,
    )
    reconstruction_metrics = image_metrics(
        reconstructed_render,
        original_render,
    )
    golden_reconstruction_metrics = image_metrics(
        reconstructed_render,
        golden_panorama,
    )
    if reconstruction_metrics["different_values"] != 0:
        raise RuntimeError(
            "ROI reconstruction differs from full maps: "
            f"{reconstruction_metrics}"
        )
    if golden_reconstruction_metrics["different_values"] != 0:
        raise RuntimeError(
            "ROI reconstruction differs from golden: "
            f"{golden_reconstruction_metrics}"
        )

    validation = {
        "original_render_vs_golden": original_metrics,
        "roi_reconstruction_vs_original": reconstruction_metrics,
        "roi_reconstruction_vs_golden": golden_reconstruction_metrics,
        "all_maps_finite": True,
        "all_valid_coordinates_in_source_bounds": True,
        "all_roi_offsets_and_widths_even": True,
        "all_body_rectangles_nv12_even": True,
        "all_seam_weight_pairs_normalized": True,
    }
    validation_path = output_directory / "validation.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    source_files = [
        reference_directory / "stitch_360.py",
        reference_directory / "export_open_chain_assets.py",
        camera_order_path,
        *[
            reference_directory / f"calib_cam{camera}.npz"
            for camera in LOGICAL_ORDER
        ],
        golden_directory / "golden_manifest.json",
        golden_directory / "python_open_chain.png",
    ]
    sources = {
        path.name: {"sha256": sha256(path), "size": path.stat().st_size}
        for path in source_files
    }

    resource_files = sorted(
        path for path in output_directory.iterdir() if path.is_file()
    )
    resources = {
        path.name: {"sha256": sha256(path), "size": path.stat().st_size}
        for path in resource_files
    }
    manifest = {
        "schema_version": 1,
        "algorithm": {
            "projection": "fisheye_open_chain_yaw_only",
            "logical_order": LOGICAL_ORDER,
            "fov_degrees": FOV_DEGREES,
            "panorama_density_width": PANORAMA_DENSITY_WIDTH,
            "seams_x": FROZEN_SEAMS,
            "feather_px": FEATHER_PX,
            "seam_source": "user_accepted_fixed_full_resolution",
        },
        "input": {
            "width": INPUT_WIDTH,
            "height": INPUT_HEIGHT,
            "pixel_format": "NV12",
        },
        "output": {
            "visible_width": VISIBLE_WIDTH,
            "logical_nv12_width": LOGICAL_NV12_WIDTH,
            "height": PANORAMA_HEIGHT,
            "stride_align": PANORAMA_STRIDE_ALIGN,
            "stride": PANORAMA_STRIDE,
            "pixel_format": "NV12",
            "padding_columns": LOGICAL_NV12_WIDTH - VISIBLE_WIDTH,
        },
        "cameras": camera_assets,
        "seams": seam_assets,
        "body_segments": body_segments,
        "camera_identity": camera_order,
        "validation": validation,
        "sources": sources,
        "resources": resources,
        "golden_capture": {
            "devices_in_logical_order": golden_manifest["capture"][
                "devices_in_logical_order"
            ],
        },
    }
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"Exported {len(camera_assets)} camera ROIs and "
        f"{len(seam_assets)} seam bands to {output_directory}"
    )
    print(
        "ROI reconstruction is pixel-identical to the frozen full-map "
        "reference and golden panorama"
    )
    print(f"manifest_sha256={sha256(manifest_path)}")


if __name__ == "__main__":
    main()
