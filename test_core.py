from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

from core import (
    ImageGeometry,
    calculate_manual_velocity_rows,
    calculate_rows,
    default_project,
    detect_geometry,
    export_manual_velocity_results_csv,
    export_results_csv,
)
from phase_plot import build_phase_figure, prepare_phase_rows


class CoreTests(unittest.TestCase):
    def make_project(self, directory: Path) -> dict:
        image_path = directory / "test_spacetime.png"
        image_path.touch()
        geometry = ImageGeometry(30, 100, 0, 0, 30, 100, 1.0)
        project = default_project(image_path, geometry)
        project["calibration"].update({"fps": 2.0, "strip_width_px": 3.0, "pixels_per_cm": 10.0})
        project["traces"]["ground"] = [[[1, 60], [28, 50]]]
        project["traces"]["water"] = [[[1, 20], [7, 20]], [[19, 30], [28, 30]]]
        project["water_outside"] = [{"x0": 10, "x1": 14, "direction": "top"}]
        return project

    def test_axes_y2_geometry_returns_source_scale(self) -> None:
        geometry = detect_geometry(Path("sample_axes_y2.png"), 20087, 3240)
        self.assertEqual(geometry.plot_width, 19842)
        self.assertEqual(geometry.source_height, 1520)
        self.assertEqual(geometry.image_to_data(210, 2030), (0, 1000))

    def test_discontinuous_trace_and_water_outside(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            project = self.make_project(Path(directory_name))
            rows = calculate_rows(project)
        self.assertEqual(len(rows), 10)
        self.assertAlmostEqual(rows[0]["water_depth_cm"], 4.0)
        self.assertEqual(rows[3]["quality_flag"], "water_above_image")
        self.assertIsNone(rows[3]["water_depth_cm"])
        self.assertIsNone(rows[3]["water_surface_height_cm"])
        self.assertAlmostEqual(rows[3]["water_depth_lower_bound_cm"], rows[3]["ground_y_source_px"] / 10.0)
        self.assertEqual(rows[5]["quality_flag"], "trace_missing")
        self.assertGreater(rows[-1]["ground_elevation_change_cm"], 0)

    def test_velocity_froude_shields_and_csv_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            velocity_path = directory / "velocity.csv"
            with velocity_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["time_s", "velocity"])
                writer.writerow([0, 50])
                writer.writerow([5, 50])
            project = self.make_project(directory)
            project["traces"]["water"] = [[[1, 20], [28, 20]]]
            project["water_outside"] = []
            project["velocity"]["csv_path"] = str(velocity_path)
            output = directory / "results.csv"
            count = export_results_csv(project, output)
            rows = calculate_rows(project)
            self.assertEqual(count, 10)
            self.assertAlmostEqual(rows[0]["velocity_m_s"], 0.5)
            expected_froude = 0.5 / math.sqrt(9.80665 * 0.04)
            self.assertAlmostEqual(rows[0]["froude_number"], expected_froude)
            self.assertGreater(rows[0]["shields_number"], 0)
            self.assertTrue(output.read_bytes().startswith(b"\xef\xbb\xbf"))

    def test_top_edge_reference_dimensionless_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            velocity_path = directory / "velocity.csv"
            velocity_path.write_text("time_s,velocity\n0,50\n5,50\n", encoding="utf-8")
            project = self.make_project(directory)
            project["traces"]["water"] = [[[1, 20], [28, 20]]]
            project["water_outside"] = [{"x0": 1, "x1": 4, "direction": "top"}]
            project["velocity"]["csv_path"] = str(velocity_path)
            row = calculate_rows(project)[0]
        self.assertIsNone(row["water_depth_cm"])
        self.assertEqual(row["reference_flag"], "water_edge_top_reference")
        self.assertAlmostEqual(row["reference_water_depth_cm"], row["ground_y_source_px"] / 10.0)
        self.assertGreater(row["reference_froude_number"], 0)
        self.assertGreater(row["reference_shields_number"], 0)

    def test_manual_velocity_points_are_added_to_phase_plot_without_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            velocity_path = directory / "manual_velocity.csv"
            velocity_path.write_text("time_s,velocity\n0,45\n2,50\n4,55\n", encoding="utf-8")
            project = self.make_project(directory)
            project["traces"]["water"] = [[[1, 20], [28, 20]]]
            project["water_outside"] = []
            project["manual_velocity"].update({"csv_path": str(velocity_path)})
            rows = prepare_phase_rows(project)
            figure, metadata = build_phase_figure(rows, plot_mode="line", frame_rate_fps=2.0, x_limits=(-3.0, -1.0), y_limits=(-2.0, 0.0))
        self.assertEqual(metadata["source_point_count"]["optical_flow"], 0)
        self.assertEqual(metadata["source_point_count"]["manual"], 3)
        self.assertEqual(metadata["axis_limits"], {"x": [-3.0, -1.0], "y": [-2.0, 0.0]})
        self.assertEqual(tuple(round(value, 6) for value in figure.axes[0].get_xlim()), (-3.0, -1.0))
        self.assertEqual(len(figure.axes[0].collections), 1)
        self.assertEqual(figure.axes[0].collections[0].get_label(), "手動計測流速（点のみ）")

    def test_manual_velocity_summary_includes_ground_elevation_rate(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            velocity_path = directory / "manual_velocity.csv"
            velocity_path.write_text("time_s,velocity\n1,45\n2,50\n", encoding="utf-8")
            project = self.make_project(directory)
            project["traces"]["water"] = [[[1, 20], [28, 20]]]
            project["water_outside"] = []
            project["manual_velocity"].update({"csv_path": str(velocity_path)})
            output = directory / "manual_summary.csv"
            rows = calculate_manual_velocity_rows(project)
            count = export_manual_velocity_results_csv(project, output)
            exported = output.read_text(encoding="utf-8-sig")
        self.assertEqual(count, 2)
        self.assertGreater(rows[0]["ground_elevation_change_rate_cm_s"], 0)
        self.assertEqual(rows[0]["ground_rate_quality_flag"], "")
        self.assertIn("ground_elevation_change_rate_cm_s", exported)


if __name__ == "__main__":
    unittest.main()
