"""Data model and hydraulic calculations for the manual trace application."""

from __future__ import annotations

import bisect
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_VERSION = 1


@dataclass
class ImageGeometry:
    image_width: int
    image_height: int
    plot_left: float
    plot_top: float
    plot_width: float
    plot_height: float
    vertical_scale: float

    @property
    def source_height(self) -> float:
        return self.plot_height / self.vertical_scale

    def image_to_data(self, image_x: float, image_y: float) -> tuple[float, float]:
        return image_x - self.plot_left, (image_y - self.plot_top) / self.vertical_scale

    def data_to_image(self, data_x: float, data_y: float) -> tuple[float, float]:
        return self.plot_left + data_x, self.plot_top + data_y * self.vertical_scale

    def contains_image_point(self, image_x: float, image_y: float) -> bool:
        return (
            self.plot_left <= image_x <= self.plot_left + self.plot_width
            and self.plot_top <= image_y <= self.plot_top + self.plot_height
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "plot_left": self.plot_left,
            "plot_top": self.plot_top,
            "plot_width": self.plot_width,
            "plot_height": self.plot_height,
            "vertical_scale": self.vertical_scale,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ImageGeometry":
        return cls(**value)


def detect_geometry(image_path: Path, width: int, height: int) -> ImageGeometry:
    name = image_path.stem.lower()
    if "_axes_y2" in name:
        return ImageGeometry(width, height, 210, 30, width - 245, height - 200, 2.0)
    if "_axes" in name:
        return ImageGeometry(width, height, 210, 30, width - 245, height - 200, 1.0)
    return ImageGeometry(width, height, 0, 0, width, height, 1.0)


def default_project(image_path: Path, geometry: ImageGeometry) -> dict[str, Any]:
    return {
        "version": PROJECT_VERSION,
        "image_path": str(image_path.resolve()),
        "geometry": geometry.as_dict(),
        "calibration": {
            "fps": 240.0,
            "strip_width_px": 3.0,
            "pixels_per_cm": 93.4,
            "ground_reference_time_s": None,
        },
        "traces": {"water": [], "ground": []},
        "water_outside": [],
        "velocity": {
            "csv_path": "",
            "time_column": "time_s",
            "velocity_column": "velocity",
            "unit": "cm/s",
        },
        "manual_velocity": {
            "csv_path": "",
            "time_column": "time_s",
            "velocity_column": "velocity",
            "unit": "cm/s",
        },
        "physics": {
            "gravity_m_s2": 9.80665,
            "water_density_kg_m3": 1000.0,
            "sediment_density_kg_m3": 2650.0,
            "d50_mm": 0.2,
            "shields_method": "darcy",
            "energy_slope": 0.01,
            "darcy_friction_factor": 0.03,
            "manning_n": 0.012,
        },
    }


def save_project(project: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    temporary.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_project(path: Path) -> dict[str, Any]:
    project = json.loads(path.read_text(encoding="utf-8"))
    if project.get("version") != PROJECT_VERSION:
        raise ValueError(f"Unsupported project version: {project.get('version')}")
    project.setdefault("manual_velocity", {
        "csv_path": "",
        "time_column": "time_s",
        "velocity_column": "velocity",
        "unit": "cm/s",
    })
    return project


def interpolate_segments(
    segments: list[list[list[float]]], sample_x: list[float]
) -> list[float | None]:
    output: list[float | None] = [None] * len(sample_x)
    for segment in segments:
        if not segment:
            continue
        points = sorted((float(point[0]), float(point[1])) for point in segment)
        xs = [point[0] for point in points]
        if len(points) == 1:
            nearest = min(range(len(sample_x)), key=lambda index: abs(sample_x[index] - xs[0]))
            output[nearest] = points[0][1]
            continue
        start = bisect.bisect_left(sample_x, xs[0])
        end = bisect.bisect_right(sample_x, xs[-1])
        for index in range(start, end):
            x = sample_x[index]
            right = bisect.bisect_right(xs, x)
            if right == 0 or right >= len(points):
                if math.isclose(x, xs[-1]):
                    output[index] = points[-1][1]
                continue
            x0, y0 = points[right - 1]
            x1, y1 = points[right]
            ratio = 0.0 if math.isclose(x0, x1) else (x - x0) / (x1 - x0)
            output[index] = y0 + ratio * (y1 - y0)
    return output


def outside_state(intervals: list[dict[str, Any]], x: float) -> str:
    state = ""
    for interval in intervals:
        start, end = sorted((float(interval["x0"]), float(interval["x1"])))
        if start <= x <= end:
            state = str(interval.get("direction", "top"))
    return state


def read_velocity_csv(
    path: Path, time_column: str, velocity_column: str, unit: str
) -> tuple[list[float], list[float]]:
    values: dict[float, list[float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError("The velocity CSV has no header")
        if time_column not in reader.fieldnames or velocity_column not in reader.fieldnames:
            raise ValueError("The selected time or velocity column is missing")
        for row in reader:
            try:
                time_value = float(row[time_column])
                velocity = float(row[velocity_column])
            except (TypeError, ValueError):
                continue
            if unit == "cm/s":
                velocity /= 100.0
            elif unit == "mm/s":
                velocity /= 1000.0
            values.setdefault(time_value, []).append(velocity)
    times = sorted(values)
    velocities = [sum(values[time]) / len(values[time]) for time in times]
    return times, velocities


def interpolate_velocity(times: list[float], values: list[float], time_s: float) -> float | None:
    if not times or time_s < times[0] or time_s > times[-1]:
        return None
    right = bisect.bisect_left(times, time_s)
    if right < len(times) and math.isclose(times[right], time_s, abs_tol=1e-12):
        return values[right]
    if right == 0 or right >= len(times):
        return None
    t0, t1 = times[right - 1], times[right]
    ratio = (time_s - t0) / (t1 - t0)
    return values[right - 1] + ratio * (values[right] - values[right - 1])


def compute_shields(
    velocity_m_s: float,
    depth_m: float,
    physics: dict[str, Any],
) -> tuple[float | None, float | None]:
    rho = float(physics["water_density_kg_m3"])
    rho_s = float(physics["sediment_density_kg_m3"])
    gravity = float(physics["gravity_m_s2"])
    diameter = float(physics["d50_mm"]) / 1000.0
    if diameter <= 0 or rho_s <= rho:
        return None, None
    method = physics.get("shields_method", "darcy")
    if method == "slope":
        shear = rho * gravity * depth_m * float(physics["energy_slope"])
    elif method == "darcy":
        shear = rho * float(physics["darcy_friction_factor"]) * velocity_m_s**2 / 8.0
    elif method == "manning":
        if depth_m <= 0:
            return None, None
        n = float(physics["manning_n"])
        shear = rho * gravity * n**2 * velocity_m_s**2 / depth_m ** (1.0 / 3.0)
    else:
        raise ValueError(f"Unknown Shields method: {method}")
    denominator = (rho_s - rho) * gravity * diameter
    return shear, shear / denominator


def calculate_rows(project: dict[str, Any]) -> list[dict[str, Any]]:
    geometry = ImageGeometry.from_dict(project["geometry"])
    calibration = project["calibration"]
    fps = float(calibration["fps"])
    strip_width = float(calibration["strip_width_px"])
    pixels_per_cm = float(calibration["pixels_per_cm"])
    if fps <= 0 or strip_width <= 0 or pixels_per_cm <= 0:
        raise ValueError("fps, strip width, and pixels/cm must be positive")

    frame_count = int(round(geometry.plot_width / strip_width))
    sample_x = [index * strip_width + (strip_width - 1) / 2 for index in range(frame_count)]
    water_y = interpolate_segments(project["traces"]["water"], sample_x)
    ground_y = interpolate_segments(project["traces"]["ground"], sample_x)

    velocity_times: list[float] = []
    velocity_values: list[float] = []
    velocity_config = project.get("velocity", {})
    velocity_path = str(velocity_config.get("csv_path", "")).strip()
    if velocity_path:
        velocity_times, velocity_values = read_velocity_csv(
            Path(velocity_path),
            str(velocity_config["time_column"]),
            str(velocity_config["velocity_column"]),
            str(velocity_config["unit"]),
        )

    reference_time = calibration.get("ground_reference_time_s")
    valid_ground_indices = [index for index, value in enumerate(ground_y) if value is not None]
    reference_ground_y: float | None = None
    if valid_ground_indices:
        if reference_time in (None, ""):
            reference_index = valid_ground_indices[0]
        else:
            target = float(reference_time) * fps
            reference_index = min(valid_ground_indices, key=lambda index: abs(index - target))
        reference_ground_y = ground_y[reference_index]

    physics = project["physics"]
    gravity = float(physics["gravity_m_s2"])
    rows: list[dict[str, Any]] = []
    for index, x in enumerate(sample_x):
        time_s = index / fps
        outside = outside_state(project.get("water_outside", []), x)
        wy = water_y[index]
        gy = ground_y[index]
        water_height = (
            None
            if wy is None or outside in ("top", "bottom")
            else (geometry.source_height - 1 - wy) / pixels_per_cm
        )
        ground_height = None if gy is None else (geometry.source_height - 1 - gy) / pixels_per_cm
        ground_change = (
            None
            if gy is None or reference_ground_y is None
            else (reference_ground_y - gy) / pixels_per_cm
        )

        depth_cm: float | None = None
        depth_lower_bound_cm: float | None = None
        reference_depth_cm: float | None = None
        reference_flag = ""
        quality = ""
        if outside == "top" and gy is not None:
            depth_lower_bound_cm = gy / pixels_per_cm
            reference_depth_cm = depth_lower_bound_cm
            reference_flag = "water_edge_top_reference"
            quality = "water_above_image"
        elif outside == "bottom":
            if gy is not None:
                reference_depth_cm = (gy - (geometry.source_height - 1)) / pixels_per_cm
                if reference_depth_cm <= 0:
                    reference_depth_cm = None
                    reference_flag = "water_edge_below_ground"
                else:
                    reference_flag = "water_edge_bottom_reference"
            quality = "water_below_image"
        elif wy is not None and gy is not None:
            raw_depth = (gy - wy) / pixels_per_cm
            if raw_depth >= 0:
                depth_cm = raw_depth
            else:
                quality = "water_below_ground"
        elif wy is None or gy is None:
            quality = "trace_missing"

        velocity = interpolate_velocity(velocity_times, velocity_values, time_s)
        froude: float | None = None
        shear: float | None = None
        shields: float | None = None
        if velocity is not None and depth_cm is not None and depth_cm > 0:
            depth_m = depth_cm / 100.0
            froude = abs(velocity) / math.sqrt(gravity * depth_m)
            shear, shields = compute_shields(velocity, depth_m, physics)

        reference_froude: float | None = None
        reference_shear: float | None = None
        reference_shields: float | None = None
        if velocity is not None and reference_depth_cm is not None and reference_depth_cm > 0:
            reference_depth_m = reference_depth_cm / 100.0
            reference_froude = abs(velocity) / math.sqrt(gravity * reference_depth_m)
            reference_shear, reference_shields = compute_shields(velocity, reference_depth_m, physics)

        rows.append(
            {
                "frame_index": index,
                "time_s": time_s,
                "water_y_source_px": wy,
                "ground_y_source_px": gy,
                "water_surface_height_cm": water_height,
                "ground_surface_height_cm": ground_height,
                "water_depth_cm": depth_cm,
                "water_depth_lower_bound_cm": depth_lower_bound_cm,
                "reference_water_depth_cm": reference_depth_cm,
                "reference_flag": reference_flag,
                "ground_elevation_change_cm": ground_change,
                "water_outside": outside,
                "quality_flag": quality,
                "velocity_m_s": velocity,
                "froude_number": froude,
                "bed_shear_stress_pa": shear,
                "shields_number": shields,
                "reference_froude_number": reference_froude,
                "reference_bed_shear_stress_pa": reference_shear,
                "reference_shields_number": reference_shields,
            }
        )
    return rows


def export_results_csv(project: dict[str, Any], path: Path) -> int:
    rows = calculate_rows(project)
    if not rows:
        raise ValueError("No rows to export")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _interpolate_result_value(
    rows: list[dict[str, Any]], key: str, time_s: float
) -> float | None:
    """Interpolate a trace-derived value only across two valid adjacent frames."""
    times = [float(row["time_s"]) for row in rows]
    if not times or time_s < times[0] or time_s > times[-1]:
        return None
    right = bisect.bisect_left(times, time_s)
    if right < len(times) and math.isclose(times[right], time_s, abs_tol=1e-12):
        value = rows[right].get(key)
        return None if value is None else float(value)
    if right == 0 or right >= len(times):
        return None
    first = rows[right - 1].get(key)
    second = rows[right].get(key)
    if first is None or second is None:
        return None
    ratio = (time_s - times[right - 1]) / (times[right] - times[right - 1])
    return float(first) + ratio * (float(second) - float(first))


def _interpolate_result_rate(
    rows: list[dict[str, Any]], key: str, time_s: float
) -> float | None:
    """Return the local time rate of a trace-derived value without crossing gaps."""
    times = [float(row["time_s"]) for row in rows]
    if not times or time_s < times[0] or time_s > times[-1]:
        return None
    right = bisect.bisect_left(times, time_s)
    if right < len(times) and math.isclose(times[right], time_s, abs_tol=1e-12):
        current = rows[right].get(key)
        if current is None:
            return None
        previous = rows[right - 1].get(key) if right > 0 else None
        following = rows[right + 1].get(key) if right + 1 < len(rows) else None
        if previous is not None and following is not None:
            return (float(following) - float(previous)) / (times[right + 1] - times[right - 1])
        if previous is not None:
            return (float(current) - float(previous)) / (times[right] - times[right - 1])
        if following is not None:
            return (float(following) - float(current)) / (times[right + 1] - times[right])
        return None
    if right == 0 or right >= len(times):
        return None
    first = rows[right - 1].get(key)
    second = rows[right].get(key)
    if first is None or second is None:
        return None
    return (float(second) - float(first)) / (times[right] - times[right - 1])


def calculate_manual_velocity_rows(project: dict[str, Any]) -> list[dict[str, Any]]:
    """Calculate dimensionless numbers at the discrete manual-velocity samples."""
    config = project.get("manual_velocity", {})
    csv_path = str(config.get("csv_path", "")).strip()
    if not csv_path:
        return []
    manual_times, manual_velocities = read_velocity_csv(
        Path(csv_path),
        str(config.get("time_column", "time_s")),
        str(config.get("velocity_column", "velocity")),
        str(config.get("unit", "cm/s")),
    )
    primary_rows = calculate_rows(project)
    physics = project["physics"]
    gravity = float(physics["gravity_m_s2"])
    results: list[dict[str, Any]] = []
    for sequence, (time_s, velocity) in enumerate(zip(manual_times, manual_velocities), start=1):
        depth_cm = _interpolate_result_value(primary_rows, "water_depth_cm", time_s)
        ground_change_cm = _interpolate_result_value(
            primary_rows, "ground_elevation_change_cm", time_s
        )
        ground_change_rate_cm_s = _interpolate_result_rate(
            primary_rows, "ground_elevation_change_cm", time_s
        )
        reference_depth_cm = None
        reference_flag = ""
        if depth_cm is None:
            reference_depth_cm = _interpolate_result_value(primary_rows, "reference_water_depth_cm", time_s)
            if reference_depth_cm is not None:
                reference_flag = "water_edge_reference"
        depth_for_calculation = depth_cm if depth_cm is not None else reference_depth_cm
        froude: float | None = None
        shear: float | None = None
        shields: float | None = None
        status = "water_edge_reference" if reference_flag else ""
        if depth_for_calculation is not None and depth_for_calculation > 0:
            depth_m = depth_for_calculation / 100.0
            froude = abs(velocity) / math.sqrt(gravity * depth_m)
            shear, shields = compute_shields(velocity, depth_m, physics)
        else:
            status = "water_depth_missing"
        results.append({
            "sequence": sequence,
            "time_s": time_s,
            "manual_velocity_m_s": velocity,
            "water_depth_cm": depth_cm,
            "reference_water_depth_cm": reference_depth_cm,
            "ground_elevation_change_cm": ground_change_cm,
            "ground_elevation_change_rate_cm_s": ground_change_rate_cm_s,
            "manual_froude_number": froude,
            "manual_bed_shear_stress_pa": shear,
            "manual_shields_number": shields,
            "quality_flag": status,
            "ground_rate_quality_flag": "" if ground_change_rate_cm_s is not None else "ground_trace_missing",
        })
    return results


def export_manual_velocity_results_csv(project: dict[str, Any], path: Path) -> int:
    rows = calculate_manual_velocity_rows(project)
    if not rows:
        return 0
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
