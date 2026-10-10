"""CPU integration regressions; GPU/color parity is checked by the main comparison.

Run: python scripts/test_triangle_splat_backends.py
"""

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import triangle_splat_renderer as renderer
from compare_triangle_splat_backends import compare_buffers, compare_frames, compare_backends, diagnose_frames


class BackendIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vertices = np.array([[-.8, -.8, 1.], [.8, -.8, 1.], [0., .8, 1.],
                                 [-1.6, -1.6, 2.], [1.6, -1.6, 2.], [0., 1.6, 2.]], dtype=np.float32)
        cls.faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        cls.geometry = cls.vertices, cls.faces
        cls.mesh = SimpleNamespace(vertices=cls.vertices, faces=cls.faces)
        cls.numba_args = cls.args("numba")
        cls.startup = renderer.prepare_raster_backend(cls.numba_args, cls.geometry)
        if cls.numba_args.raster_backend != "numba":
            raise AssertionError("Numba must compile for these integration tests")

    @staticmethod
    def args(backend="numpy"):
        return argparse.Namespace(
            raster_backend=backend, camera_position=[0., 0., 0.], camera_look_at=[0., 0., 1.],
            image_width=32, image_height=32, fov_y=90., visibility_mode="raster",
            compare_visibility=False, splat_raster_mode="reuse", batch_size=512,
            experiment=Path("synthetic"), kernel_face_cache=None, cutoff=.005,
            benchmark_runs=2, raster_backend_report=None,
        )

    def setUp(self):
        self.torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), __version__="mock")
        self.torch_patch = patch.dict(sys.modules, {"torch": self.torch})
        self.torch_patch.start()
        self.stdout = redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)
        self.addCleanup(self.torch_patch.stop)

    def visibility(self, args):
        return renderer.raster_visible_face_mask(
            self.mesh, args, retain_buffers=True, raster_geometry=self.geometry,
            profiling=renderer.RasterVisibilityTiming(enabled=False))

    def test_default_numpy_needs_no_numba_import(self):
        with patch.object(sys, "argv", ["triangle_splat_renderer.py"]):
            args = renderer.parse_args()
        self.assertEqual(args.raster_backend, "numpy")
        with patch("builtins.__import__", side_effect=AssertionError("unexpected optional import")):
            self.assertEqual(renderer.prepare_raster_backend(args, self.geometry)["total_s"], 0.)

    def test_visibility_and_kernel_face_filtering(self):
        nmask, nbuffers = self.visibility(self.args())
        cmask, cbuffers = self.visibility(self.numba_args)
        np.testing.assert_array_equal(cmask, nmask)
        compare_buffers(nbuffers, cbuffers)
        np.testing.assert_array_equal(cmask, [True, False])
        affected = [np.array([0, 1]), np.array([1]), np.array([], dtype=np.int64)]
        a = renderer.filter_cached_kernel_visibility(self.mesh, affected, self.args(), self.geometry, benchmark=True)
        b = renderer.filter_cached_kernel_visibility(self.mesh, affected, self.numba_args, self.geometry, benchmark=True)
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[0], [0])
        np.testing.assert_array_equal(b[1][0], [0])

    def test_import_failure_falls_back_to_numpy(self):
        args = self.args("numba")
        with patch("builtins.__import__", side_effect=ImportError("Numba unavailable")):
            renderer.prepare_raster_backend(args, self.geometry)
        self.assertEqual(args.raster_backend, "numpy")
        self.assertIn("Numba unavailable", args._raster_backend_fallback)
        compare_buffers(self.visibility(self.args())[1], self.visibility(args)[1])

    def test_runtime_failure_falls_back_to_numpy(self):
        args = self.args("numba")
        args._numba_raster = unittest.mock.Mock(side_effect=RuntimeError("kernel failure"))
        result = self.visibility(args)
        self.assertEqual(args.raster_backend, "numpy")
        compare_buffers(self.visibility(self.args())[1], result[1])

    def frame(self, args, affected, force_separate=False):
        # Isolate raster/filter/reuse wiring. The real GPU evaluator is exercised
        # by --compare-raster-backends on the trained experiment, not this mock.
        def colors(trainer, face_ids, barys, kernels, batch_size, masks, timing, **kwargs):
            self.assertIs(kwargs["gating_lookup"], self.gating)
            self.assertFalse(kwargs["manage_knn_cache"])
            np.testing.assert_array_equal(kwargs["covered_mask"], face_ids >= 0)
            return np.nan_to_num(barys).astype(np.float32)

        with patch.object(renderer, "profiled_selected_kernel_colors", side_effect=colors):
            return renderer.render_splat_frame(
                self.mesh, affected, args, self.geometry, None, self.gating,
                renderer.KernelEvaluationTiming(lambda: None, enabled=False), force_separate)

    def test_reuse_separate_and_unsupported_pixels(self):
        self.gating = object()
        affected = [np.array([0, 1])]
        original = self.frame(self.args(), affected)
        accelerated = self.frame(self.numba_args, affected)
        metrics = compare_frames(original, accelerated)
        self.assertTrue(metrics["raster_buffers_reused"])
        self.assertEqual(metrics["rendered_rgb"]["exact_channel_mismatches"], 0)
        separate = self.frame(self.numba_args, affected, force_separate=True)
        self.assertFalse(separate["reused"])
        compare_buffers(original["buffers"][1:], separate["buffers"][1:])
        unsupported = self.frame(self.numba_args, [])
        self.assertFalse(unsupported["reused"])
        self.assertTrue((unsupported["buffers"][2] == -1).all())
        self.assertFalse(unsupported["colors"].any())

    def test_comparison_rejects_face_or_rgb_changes(self):
        self.gating = object()
        original = self.frame(self.args(), [np.array([0])])
        changed = self.frame(self.numba_args, [np.array([0])])
        changed["colors"][0, 0] = 0.5
        with self.assertRaises(AssertionError):
            compare_frames(original, changed)
        depth, faces, barys = original["visibility"]
        cfaces = faces.copy()
        cfaces[0, 0] = 1
        with self.assertRaises(AssertionError):
            compare_buffers((depth, faces, barys), (depth, cfaces, barys))

    def test_full_frame_comparison_records_both_warmed_paths(self):
        self.gating = object()
        affected = [np.array([0, 1])]
        with patch.object(renderer, "profiled_selected_kernel_colors",
                          side_effect=lambda trainer, ids, barys, *a, **kw: np.nan_to_num(barys).astype(np.float32)):
            report = compare_backends(
                self.mesh, affected, self.args(), self.numba_args, self.geometry,
                None, self.gating, renderer.KernelEvaluationTiming(lambda: None, enabled=False),
                lambda: None, force_separate=False, startup={"raster_backend": self.startup})
        self.assertEqual(report["correctness"]["full_mesh_visibility"]["face_id_mismatches"], 0)
        for backend in ("numpy", "numba"):
            self.assertEqual(len(report["warmed_full_frame"][backend]["samples_s"]), 2)
            self.assertGreater(report["validation_warmup_s"][backend], 0.)
        self.assertIn("jit_compilation_s", report["startup"]["raster_backend"])

    def test_diagnostics_distinguish_exact_from_tolerant_depth_equality(self):
        self.gating = object()
        original = self.frame(self.args(), [np.array([0])])
        changed = dict(original)
        changed["visibility"] = tuple(array.copy() for array in original["visibility"])
        y, x = np.argwhere(original["visibility"][1] >= 0)[0]
        changed["visibility"][0][y, x] = np.nextafter(changed["visibility"][0][y, x], np.inf)
        compare_frames(original, changed)  # Existing float64 tolerance still passes.
        self.assertFalse(diagnose_frames(original, changed)["exact_full_mesh_visibility"]["depth"])

    def test_rgb_failure_saves_repeat_diagnostics_before_raising(self):
        self.gating = object()
        calls = 0

        def colors(trainer, ids, barys, *a, **kw):
            nonlocal calls
            calls += 1
            result = np.nan_to_num(barys).astype(np.float32)
            if calls % 2 == 0:  # Both Numba frames: stable backend-specific error.
                result[16, 16, 0] += 0.00012624
            return result

        with tempfile.TemporaryDirectory() as directory:
            args = self.args()
            args.raster_backend_report = Path(directory) / "failed_diagnostic.json"
            with patch.object(renderer, "profiled_selected_kernel_colors", side_effect=colors):
                with self.assertRaises(AssertionError):
                    compare_backends(
                        self.mesh, [np.array([0, 1])], args, self.numba_args, self.geometry,
                        None, self.gating, renderer.KernelEvaluationTiming(lambda: None, enabled=False),
                        lambda: None, force_separate=False, startup={"raster_backend": self.startup})
            report = json.loads(args.raster_backend_report.read_text())
        self.assertEqual(calls, 4)  # No warmed performance claims after failure.
        self.assertNotIn("measured_numpy_over_numba_frame_ratio", report)
        for name in ("numpy_vs_numpy", "numba_vs_numba"):
            self.assertEqual(report["repeat_diagnostics"][name]["differing_rgb_channels"], 0)
        cross = report["repeat_diagnostics"]["numpy_vs_numba"]
        self.assertEqual(cross["rgb_channels_above_1e_6"], 1)
        self.assertEqual(cross["differing_rgb_channels"], 1)
        self.assertGreater(cross["max_rgb_absolute_error"], 1e-6)
        location = cross["largest_rgb_differences"][0]
        self.assertEqual((location["x"], location["y"], location["channel"]), (16, 16, 0))
        self.assertTrue(all(cross["exact_full_mesh_visibility"].values()))
        self.assertTrue(cross["selected_kernel_ids_exactly_equal"])
        self.assertTrue(cross["kernel_to_face_support_exactly_equal"])
        self.assertIn("numpy_vs_numba", report["strict_assertion_failures"])

    def test_allow_rgb_mismatch_completes_timings_and_marks_correctness_failed(self):
        calls = 0

        def colors(trainer, ids, barys, *a, **kw):
            nonlocal calls
            calls += 1
            result = np.nan_to_num(barys).astype(np.float32)
            if calls % 2 == 0:
                result[16, 16, 0] += 0.00012624
            return result

        args = self.args()
        args.allow_rgb_mismatch = True
        with tempfile.TemporaryDirectory() as directory:
            args.raster_backend_report = Path(directory) / "completed_benchmark.json"
            with patch.object(renderer, "profiled_selected_kernel_colors", side_effect=colors):
                report = compare_backends(
                    self.mesh, [np.array([0, 1])], args, self.numba_args, self.geometry,
                    None, object(), renderer.KernelEvaluationTiming(lambda: None, enabled=False),
                    lambda: None, force_separate=False, startup={"raster_backend": self.startup})
            saved = json.loads(args.raster_backend_report.read_text())
        self.assertEqual(calls, 4 + 2 * args.benchmark_runs)
        self.assertEqual(report["raster_support_parity"], "PASSED")
        self.assertEqual(report["rgb_parity"], "FAILED")
        self.assertEqual(report["full_correctness"], "FAILED")
        self.assertEqual(saved["full_correctness"], "FAILED")
        self.assertEqual(report["rgb_absolute_tolerance"], 1e-6)
        self.assertEqual(report["correctness"]["rendered_rgb"]["parity"], "FAILED")
        self.assertIn("numpy_vs_numba", report["strict_assertion_failures"])
        for backend in ("numpy", "numba"):
            self.assertEqual(len(report["warmed_full_frame"][backend]["samples_s"]), args.benchmark_runs)

    def test_nonfatal_rgb_mode_retains_all_surface_and_support_checks(self):
        self.gating = object()
        original = self.frame(self.args(), [np.array([0])])
        for field in ("faces", "depth", "barycentrics", "coverage", "kernel_ids", "support"):
            with self.subTest(field=field):
                changed = self.frame(self.numba_args, [np.array([0])])
                if field == "faces":
                    changed["visibility"][1][16, 16] = 1
                elif field == "depth":
                    changed["visibility"][0][16, 16] += 0.01
                elif field == "barycentrics":
                    changed["visibility"][2][16, 16, 0] += 0.01
                elif field == "coverage":
                    changed["buffers"][0][0, 0, 1] = 255
                elif field == "kernel_ids":
                    changed["kernel_ids"][0] = 1
                else:
                    changed["visible_sets"][0][0] = 1
                with self.assertRaises(AssertionError):
                    compare_frames(original, changed, strict_rgb=False)

    def test_allow_rgb_mismatch_keeps_raster_failure_fatal_in_harness(self):
        from triangle_splat_renderer import render_splat_frame

        args = self.args()
        args.allow_rgb_mismatch = True
        calls = 0

        def corrupted(*a, **kw):
            nonlocal calls
            calls += 1
            result = render_splat_frame(*a, **kw)
            if calls == 2:
                result["visibility"][0][16, 16] += 0.01
            return result

        with tempfile.TemporaryDirectory() as directory:
            args.raster_backend_report = Path(directory) / "raster_failure.json"
            with patch.object(renderer, "render_splat_frame", side_effect=corrupted), \
                 patch.object(renderer, "profiled_selected_kernel_colors",
                              side_effect=lambda trainer, ids, barys, *a, **kw: np.nan_to_num(barys).astype(np.float32)):
                with self.assertRaises(AssertionError):
                    compare_backends(
                        self.mesh, [np.array([0, 1])], args, self.numba_args, self.geometry,
                        None, object(), renderer.KernelEvaluationTiming(lambda: None, enabled=False),
                        lambda: None, force_separate=False, startup={"raster_backend": self.startup})
            report = json.loads(args.raster_backend_report.read_text())
        self.assertEqual(calls, 4)
        self.assertEqual(report["raster_support_parity"], "FAILED")
        self.assertNotIn("warmed_full_frame", report)


if __name__ == "__main__":
    unittest.main()
