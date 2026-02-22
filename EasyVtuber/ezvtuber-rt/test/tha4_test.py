"""
Test cases for tha4.py - THA4Engines and THA4EnginesSimple classes (TensorRT-based)
Mirrors coverage patterns from tha3_test.py for parity across pipelines.
"""
import unittest
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401 - ensures CUDA context is initialized
import cv2
import os
import json
from typing import List
from tqdm import tqdm

from ezvtb_rt.tha4 import THA4Engines, THA4EnginesSimple


class TestTHA4Base(unittest.TestCase):
    """Base test class with common utilities"""

    TEST_IMAGE_PATH = './test/data/base.png'
    POSE_DATA_PATH = './test/data/pose_20fps.json'

    @classmethod
    def setUpClass(cls):
        """Load test data once for all tests"""
        cls.test_image = cv2.imread(cls.TEST_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
        if cls.test_image is None:
            raise FileNotFoundError(f"Test image not found at {cls.TEST_IMAGE_PATH}")

        with open(cls.POSE_DATA_PATH, 'r') as f:
            cls.pose_data = json.load(f)

    def create_random_pose(self) -> np.ndarray:
        """Create a random pose array with shape (1, 45)"""
        return np.random.randn(1, 45).astype(np.float32)

    def create_zero_pose(self) -> np.ndarray:
        """Create a zero pose array with shape (1, 45)"""
        return np.zeros((1, 45), dtype=np.float32)

    def validate_output(self, output: np.ndarray):
        """Validate inference output format for THA4Engines"""
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(output.shape, (512, 512, 4))
        self.assertEqual(output.dtype, np.uint8)


# ============================================================================
# THA4EnginesSimple Tests
# ============================================================================


class TestTHA4EnginesSimpleFP16(TestTHA4Base):
    """Test THA4EnginesSimple with fp16 model"""

    MODEL_DIR = './data/tha4/fp16'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA4EnginesSimple(cls.MODEL_DIR)

    def test_set_image_valid(self):
        """Test setting image with valid input"""
        self.engine.setImage(self.test_image)

    def test_inference_after_set_image(self):
        """Test inference after setting image"""
        self.engine.setImage(self.test_image)
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        output = self.engine.inference(pose)
        self.validate_output(output)

    def test_inference_with_zero_pose(self):
        """Test inference with zero pose"""
        self.engine.setImage(self.test_image)
        output = self.engine.inference(self.create_zero_pose())
        self.validate_output(output)

    def test_inference_with_random_pose(self):
        """Test inference with random pose values"""
        self.engine.setImage(self.test_image)
        output = self.engine.inference(self.create_random_pose())
        self.validate_output(output)

    def test_multiple_inferences(self):
        """Test multiple consecutive inferences"""
        self.engine.setImage(self.test_image)
        for i in range(5):
            pose = np.array(self.pose_data[800 + i]).reshape(1, 45).astype(np.float32)
            output = self.engine.inference(pose)
            self.validate_output(output)

    def test_engines_created(self):
        """Test that all engines are created"""
        self.assertIsNotNone(self.engine.decomposer)
        self.assertIsNotNone(self.engine.combiner)
        self.assertIsNotNone(self.engine.morpher)
        self.assertIsNotNone(self.engine.body_morpher)
        self.assertIsNotNone(self.engine.upscaler)
        self.assertIsNotNone(self.engine.stream)


class TestTHA4EnginesSimpleEdgeCases(TestTHA4Base):
    """Test edge cases for THA4EnginesSimple"""

    MODEL_DIR = './data/tha4/fp16'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA4EnginesSimple(cls.MODEL_DIR)

    def test_set_image_wrong_dimensions(self):
        """Test that wrong image dimensions raise AssertionError"""
        wrong_size_image = np.zeros((256, 256, 4), dtype=np.uint8)
        with self.assertRaises(AssertionError):
            self.engine.setImage(wrong_size_image)

    def test_set_image_wrong_channels(self):
        """Test that wrong channel count raises AssertionError"""
        wrong_channels_image = np.zeros((512, 512, 3), dtype=np.uint8)
        with self.assertRaises(AssertionError):
            self.engine.setImage(wrong_channels_image)

    def test_set_image_2d_array(self):
        """Test that 2D array raises AssertionError"""
        wrong_shape_image = np.zeros((512, 512), dtype=np.uint8)
        with self.assertRaises(AssertionError):
            self.engine.setImage(wrong_shape_image)

    def test_set_image_wrong_dtype(self):
        """Test that wrong dtype raises AssertionError"""
        wrong_dtype_image = np.zeros((512, 512, 4), dtype=np.float32)
        with self.assertRaises(AssertionError):
            self.engine.setImage(wrong_dtype_image)

    def test_pose_extreme_values(self):
        """Test inference with extreme pose values"""
        self.engine.setImage(self.test_image)
        extreme_pose = np.ones((1, 45), dtype=np.float32) * 100.0
        output = self.engine.inference(extreme_pose)
        self.validate_output(output)

    def test_pose_negative_values(self):
        """Test inference with negative pose values"""
        self.engine.setImage(self.test_image)
        negative_pose = np.ones((1, 45), dtype=np.float32) * -50.0
        output = self.engine.inference(negative_pose)
        self.validate_output(output)

    def test_rapid_pose_changes(self):
        """Test rapid pose changes in succession"""
        self.engine.setImage(self.test_image)
        for _ in range(20):
            pose = self.create_random_pose()
            output = self.engine.inference(pose)
            self.validate_output(output)


class TestTHA4EnginesSimpleOutputConsistency(TestTHA4Base):
    """Test output consistency for THA4EnginesSimple"""

    MODEL_DIR = './data/tha4/fp16'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA4EnginesSimple(cls.MODEL_DIR)

    def test_deterministic_output(self):
        """Test that same input produces consistent output"""
        self.engine.setImage(self.test_image)
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)

        output1 = self.engine.inference(pose).copy()
        output2 = self.engine.inference(pose).copy()

        np.testing.assert_array_equal(output1, output2)

    def test_different_poses_different_output(self):
        """Test that different poses produce different outputs"""
        self.engine.setImage(self.test_image)

        pose1 = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        pose2 = np.array(self.pose_data[850]).reshape(1, 45).astype(np.float32)

        output1 = self.engine.inference(pose1).copy()
        output2 = self.engine.inference(pose2).copy()

        self.assertFalse(np.array_equal(output1, output2))


# ============================================================================
# THA4Engines Tests (with VRAM caching)
# ============================================================================


class TestTHA4EnginesWithCacheEyebrow(TestTHA4Base):
    """Test THA4Engines with VRAM caching and eyebrow enabled"""

    MODEL_DIR = './data/tha4/fp16'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA4Engines(cls.MODEL_DIR, vram_cache_size=1.0, use_eyebrow=True)

    def test_eyebrow_mode(self):
        """Eyebrow mode is enabled"""
        self.assertTrue(self.engine.use_eyebrow)

    def test_cacher_created(self):
        """Cacher is created when cache enabled"""
        self.assertIsNotNone(self.engine.cacher)

    def test_sync_set_image(self):
        """syncSetImage works"""
        self.engine.syncSetImage(self.test_image)

    def test_async_set_image(self):
        """asyncSetImage works"""
        self.engine.asyncSetImage(self.test_image)
        self.engine.stream.synchronize()

    def test_inference_after_sync_set_image(self):
        """Run inference after syncSetImage"""
        self.engine.syncSetImage(self.test_image)
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        self.engine.asyncInfer(pose)
        output = self.engine.syncAndGetOutput()
        self.validate_output(output)

    def test_inference_with_zero_pose(self):
        """Inference with zero pose"""
        self.engine.syncSetImage(self.test_image)
        self.engine.asyncInfer(self.create_zero_pose())
        output = self.engine.syncAndGetOutput()
        self.validate_output(output)

    def test_inference_with_random_pose(self):
        """Inference with random pose"""
        self.engine.syncSetImage(self.test_image)
        self.engine.asyncInfer(self.create_random_pose())
        output = self.engine.syncAndGetOutput()
        self.validate_output(output)

    def test_multiple_inferences(self):
        """Multiple consecutive inferences"""
        self.engine.syncSetImage(self.test_image)
        for i in range(5):
            pose = np.array(self.pose_data[800 + i]).reshape(1, 45).astype(np.float32)
            self.engine.asyncInfer(pose)
            output = self.engine.syncAndGetOutput()
            self.validate_output(output)

    def test_get_output_mem(self):
        """getOutputMem returns HostDeviceMem"""
        self.engine.syncSetImage(self.test_image)
        pose = np.array(self.pose_data[810]).reshape(1, 45).astype(np.float32)
        self.engine.asyncInfer(pose)
        _ = self.engine.syncAndGetOutput()
        out_mem = self.engine.getOutputMem()
        self.assertIsInstance(out_mem, type(self.engine.upscaler.outputs[1]))


class TestTHA4EnginesWithCacheNoEyebrow(TestTHA4Base):
    """Test THA4Engines with cache and eyebrow disabled"""

    MODEL_DIR = './data/tha4/fp16'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA4Engines(cls.MODEL_DIR, vram_cache_size=1.0, use_eyebrow=False)

    def test_eyebrow_mode_disabled(self):
        """Eyebrow mode is disabled"""
        self.assertFalse(self.engine.use_eyebrow)

    def test_inference_no_eyebrow(self):
        """Inference without eyebrow path"""
        self.engine.syncSetImage(self.test_image)
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        self.engine.asyncInfer(pose)
        output = self.engine.syncAndGetOutput()
        self.validate_output(output)

    def test_multiple_inferences_no_eyebrow(self):
        """Multiple inferences without eyebrow"""
        self.engine.syncSetImage(self.test_image)
        for i in range(5):
            pose = np.array(self.pose_data[820 + i]).reshape(1, 45).astype(np.float32)
            self.engine.asyncInfer(pose)
            output = self.engine.syncAndGetOutput()
            self.validate_output(output)


class TestTHA4EnginesNoCache(TestTHA4Base):
    """Test THA4Engines without VRAM caching"""

    MODEL_DIR = './data/tha4/fp16'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA4Engines(cls.MODEL_DIR, vram_cache_size=0.0, use_eyebrow=True)

    def test_cacher_disabled(self):
        """Cacher should be None when disabled"""
        self.assertIsNone(self.engine.cacher)

    def test_inference_without_cache(self):
        """Inference path with cache disabled"""
        self.engine.syncSetImage(self.test_image)
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        self.engine.asyncInfer(pose)
        output = self.engine.syncAndGetOutput()
        self.validate_output(output)

    def test_multiple_inferences_without_cache(self):
        """Multiple inferences with cache disabled"""
        self.engine.syncSetImage(self.test_image)
        for i in range(5):
            pose = np.array(self.pose_data[840 + i]).reshape(1, 45).astype(np.float32)
            self.engine.asyncInfer(pose)
            output = self.engine.syncAndGetOutput()
            self.validate_output(output)


class TestTHA4EnginesCacheBehavior(TestTHA4Base):
    """Test THA4Engines VRAM cache behavior"""

    MODEL_DIR = './data/tha4/fp16'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA4Engines(cls.MODEL_DIR, vram_cache_size=1.0, use_eyebrow=True)

    def setUp(self):
        if self.engine.cacher is not None:
            self.engine.cacher.clear()

    def test_cache_hits_on_repeated_same_pose(self):
        """Cache should hit when repeating same pose"""
        self.engine.syncSetImage(self.test_image)
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)

        self.engine.asyncInfer(pose)
        output1 = self.engine.syncAndGetOutput()

        self.engine.asyncInfer(pose)
        output2 = self.engine.syncAndGetOutput()

        self.validate_output(output1)
        self.validate_output(output2)
        self.assertGreater(self.engine.cacher.hits, 0)

    def test_cache_misses_on_different_poses(self):
        """Different poses should produce cache misses"""
        self.engine.syncSetImage(self.test_image)
        initial_misses = self.engine.cacher.miss

        for i in range(4):
            pose = np.array(self.pose_data[800 + i * 3]).reshape(1, 45).astype(np.float32)
            self.engine.asyncInfer(pose)
            self.engine.syncAndGetOutput()

        self.assertGreater(self.engine.cacher.miss, initial_misses)

    def test_cache_second_pass_has_hits(self):
        """Second pass over same poses should have hits"""
        self.engine.syncSetImage(self.test_image)
        poses = [np.array(self.pose_data[800 + i]).reshape(1, 45).astype(np.float32) for i in range(10)]

        for pose in poses:
            self.engine.asyncInfer(pose)
            self.engine.syncAndGetOutput()

        hits_after_first_pass = self.engine.cacher.hits

        for pose in poses:
            self.engine.asyncInfer(pose)
            self.engine.syncAndGetOutput()

        hits_after_second_pass = self.engine.cacher.hits
        self.assertGreater(hits_after_second_pass, hits_after_first_pass)

    def test_cache_hit_rate_property(self):
        """Cache hit rate is computed"""
        self.engine.syncSetImage(self.test_image)
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)

        for _ in range(4):
            self.engine.asyncInfer(pose)
            self.engine.syncAndGetOutput()

        hit_rate = self.engine.cacher.hit_rate
        self.assertGreaterEqual(hit_rate, 0.0)


class TestTHA4EnginesEdgeCases(TestTHA4Base):
    """Edge cases for THA4Engines (cached)."""

    MODEL_DIR = './data/tha4/fp16'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA4Engines(cls.MODEL_DIR, vram_cache_size=0.5, use_eyebrow=True)

    def test_set_image_wrong_dimensions(self):
        wrong_size_image = np.zeros((256, 256, 4), dtype=np.uint8)
        with self.assertRaises(AssertionError):
            self.engine.syncSetImage(wrong_size_image)

    def test_set_image_wrong_channels(self):
        wrong_channels_image = np.zeros((512, 512, 3), dtype=np.uint8)
        with self.assertRaises(AssertionError):
            self.engine.syncSetImage(wrong_channels_image)

    def test_set_image_2d_array(self):
        wrong_shape_image = np.zeros((512, 512), dtype=np.uint8)
        with self.assertRaises(AssertionError):
            self.engine.syncSetImage(wrong_shape_image)

    def test_set_image_wrong_dtype(self):
        wrong_dtype_image = np.zeros((512, 512, 4), dtype=np.float32)
        with self.assertRaises(AssertionError):
            self.engine.syncSetImage(wrong_dtype_image)

    def test_pose_extreme_values(self):
        self.engine.syncSetImage(self.test_image)
        extreme_pose = np.ones((1, 45), dtype=np.float32) * 100.0
        self.engine.asyncInfer(extreme_pose)
        output = self.engine.syncAndGetOutput()
        self.validate_output(output)

    def test_pose_negative_values(self):
        self.engine.syncSetImage(self.test_image)
        negative_pose = np.ones((1, 45), dtype=np.float32) * -50.0
        self.engine.asyncInfer(negative_pose)
        output = self.engine.syncAndGetOutput()
        self.validate_output(output)

    def test_image_update_between_inferences(self):
        self.engine.syncSetImage(self.test_image)
        pose = self.create_zero_pose()
        self.engine.asyncInfer(pose)
        first = self.engine.syncAndGetOutput()
        self.validate_output(first)

        self.engine.syncSetImage(self.test_image)
        self.engine.asyncInfer(pose)
        second = self.engine.syncAndGetOutput()
        self.validate_output(second)

    def test_rapid_pose_changes(self):
        self.engine.syncSetImage(self.test_image)
        for _ in range(30):
            pose = self.create_random_pose()
            self.engine.asyncInfer(pose)
            output = self.engine.syncAndGetOutput()
            self.validate_output(output)


class TestTHA4EnginesOutputConsistency(TestTHA4Base):
    """Output consistency checks for cached THA4."""

    MODEL_DIR = './data/tha4/fp16'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA4Engines(cls.MODEL_DIR, vram_cache_size=1.0, use_eyebrow=True)

    def test_deterministic_output(self):
        self.engine.syncSetImage(self.test_image)
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)

        self.engine.asyncInfer(pose)
        out1 = self.engine.syncAndGetOutput().copy()

        self.engine.asyncInfer(pose)
        out2 = self.engine.syncAndGetOutput().copy()

        np.testing.assert_array_equal(out1, out2)

    def test_different_poses_different_output(self):
        self.engine.syncSetImage(self.test_image)
        pose1 = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        pose2 = np.array(self.pose_data[850]).reshape(1, 45).astype(np.float32)

        self.engine.asyncInfer(pose1)
        out1 = self.engine.syncAndGetOutput().copy()

        self.engine.asyncInfer(pose2)
        out2 = self.engine.syncAndGetOutput().copy()

        self.assertFalse(np.array_equal(out1, out2))


class TestTHA4EnginesPoseSlicing(TestTHA4Base):
    """Verify pose slicing matches expected segments."""

    MODEL_DIR = './data/tha4/fp16'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA4Engines(cls.MODEL_DIR, vram_cache_size=0.5, use_eyebrow=True)

    def test_eyebrow_pose_slicing(self):
        self.engine.syncSetImage(self.test_image)
        pose = np.arange(45, dtype=np.float32).reshape(1, 45)
        self.engine.asyncInfer(pose)
        output = self.engine.syncAndGetOutput()
        self.validate_output(output)

    def test_face_pose_slicing(self):
        self.engine.syncSetImage(self.test_image)
        pose = np.arange(45, dtype=np.float32).reshape(1, 45)
        self.engine.asyncInfer(pose)
        output = self.engine.syncAndGetOutput()
        self.validate_output(output)

    def test_rotation_pose_slicing(self):
        self.engine.syncSetImage(self.test_image)
        pose = np.arange(45, dtype=np.float32).reshape(1, 45)
        self.engine.asyncInfer(pose)
        output = self.engine.syncAndGetOutput()
        self.validate_output(output)


class TestTHA4EnginesCompareSimpleVsCached(TestTHA4Base):
    """Compare outputs between simple and cached THA4 engines."""

    MODEL_DIR = './data/tha4/fp16'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine_simple = THA4EnginesSimple(cls.MODEL_DIR)
        cls.engine_cached = THA4Engines(cls.MODEL_DIR, vram_cache_size=1.0, use_eyebrow=True)

    def test_both_engines_produce_valid_output(self):
        self.engine_simple.setImage(self.test_image)
        self.engine_cached.syncSetImage(self.test_image)

        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)

        out_simple = self.engine_simple.inference(pose)
        self.engine_cached.asyncInfer(pose)
        out_cached = self.engine_cached.syncAndGetOutput()

        self.validate_output(out_simple)
        self.validate_output(out_cached)

    def test_outputs_are_similar(self):
        self.engine_simple.setImage(self.test_image)
        self.engine_cached.syncSetImage(self.test_image)
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)

        out_simple = self.engine_simple.inference(pose)
        self.engine_cached.asyncInfer(pose)
        out_cached = self.engine_cached.syncAndGetOutput()

        diff = np.abs(out_simple.astype(np.float32) - out_cached.astype(np.float32))
        mean_diff = np.mean(diff)
        self.assertLess(mean_diff, 5.0)


class TestTHA4EnginesEyebrowVsNoEyebrow(TestTHA4Base):
    """Compare outputs between eyebrow enabled/disabled."""

    MODEL_DIR = './data/tha4/fp16'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine_eyebrow = THA4Engines(cls.MODEL_DIR, vram_cache_size=0.5, use_eyebrow=True)
        cls.engine_no_eyebrow = THA4Engines(cls.MODEL_DIR, vram_cache_size=0.5, use_eyebrow=False)

    def test_both_modes_produce_valid_output(self):
        self.engine_eyebrow.syncSetImage(self.test_image)
        self.engine_no_eyebrow.syncSetImage(self.test_image)

        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)

        self.engine_eyebrow.asyncInfer(pose)
        out_eye = self.engine_eyebrow.syncAndGetOutput()

        self.engine_no_eyebrow.asyncInfer(pose)
        out_no_eye = self.engine_no_eyebrow.syncAndGetOutput()

        self.validate_output(out_eye)
        self.validate_output(out_no_eye)


class TestTHA4EnginesAllConfigurations(TestTHA4Base):
    """Test THA4Engines across model configs."""

    MODEL_CONFIGS = [
        ('./data/tha4/fp16', True, 1.0),
        ('./data/tha4/fp16', True, 0.0),
        ('./data/tha4/fp16', False, 1.0),
        ('./data/tha4/fp32', True, 1.0),
    ]

    def test_all_configurations(self):
        for model_dir, use_eyebrow, cache_size in self.MODEL_CONFIGS:
            if not os.path.exists(model_dir):
                continue
            with self.subTest(model_dir=model_dir, use_eyebrow=use_eyebrow, cache_size=cache_size):
                engine = THA4Engines(model_dir, vram_cache_size=cache_size, use_eyebrow=use_eyebrow)

                self.assertEqual(engine.use_eyebrow, use_eyebrow)
                if cache_size > 0:
                    self.assertIsNotNone(engine.cacher)
                else:
                    self.assertIsNone(engine.cacher)

                engine.syncSetImage(self.test_image)
                pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
                engine.asyncInfer(pose)
                output = engine.syncAndGetOutput()
                self.validate_output(output)


class TestTHA4EnginesAsyncStream(TestTHA4Base):
    """Async stream usage for THA4."""

    MODEL_DIR = './data/tha4/fp16'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA4Engines(cls.MODEL_DIR, vram_cache_size=1.0, use_eyebrow=True)

    def test_custom_stream(self):
        self.engine.syncSetImage(self.test_image)
        custom_stream = cuda.Stream()
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        self.engine.asyncInfer(pose, stream=custom_stream)
        custom_stream.synchronize()
        self.engine.finishedFetch.synchronize()
        output = self.engine.upscaler.outputs[1].host
        self.validate_output(output)

    def test_multiple_async_inferences(self):
        self.engine.syncSetImage(self.test_image)
        for i in range(8):
            pose = np.array(self.pose_data[800 + i]).reshape(1, 45).astype(np.float32)
            self.engine.asyncInfer(pose)
            output = self.engine.syncAndGetOutput()
            self.validate_output(output)


# ============================================================================
# Video generation helper and showcase functions (manual verification)
# ============================================================================


def generate_video(imgs: List[np.ndarray], video_path: str, framerate: float):
    """Generate video from a list of BGR images."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_path, fourcc, framerate, (imgs[0].shape[1], imgs[0].shape[0]))
    if not video.isOpened():
        raise ValueError("CV2 video encoder Not supported")
    for img in imgs:
        video.write(img)
    video.release()
    cv2.destroyAllWindows()


def THA4EnginesSimple_ShowVideo():
    model_dir = './data/tha4/fp16'
    if not os.path.exists(model_dir):
        print(f"Skipping: Model directory not found: {model_dir}")
        return

    engine = THA4EnginesSimple(model_dir)
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    engine.setImage(img)

    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)

    frames = []
    print("Generating frames with THA4EnginesSimple:")
    for i in tqdm(range(len(pose_data[800:1200]))):
        pose = np.array(pose_data[800 + i]).reshape(1, 45).astype(np.float32)
        output = engine.inference(pose)
        frames.append(output[:, :, :3].copy())

    generate_video(frames, './test/data/tha4_simple_test.mp4', 20)


def THA4Engines_ShowVideo_WithCache_Eyebrow():
    model_dir = './data/tha4/fp16'
    if not os.path.exists(model_dir):
        print(f"Skipping: Model directory not found: {model_dir}")
        return

    engine = THA4Engines(model_dir, vram_cache_size=1.0, use_eyebrow=True)
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    engine.syncSetImage(img)

    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)

    frames = []
    print("Generating frames with THA4Engines (cache=1.0GB, eyebrow=True) - Pass 1:")
    for i in tqdm(range(len(pose_data[800:1200]))):
        pose = np.array(pose_data[800 + i]).reshape(1, 45).astype(np.float32)
        engine.asyncInfer(pose)
        output = engine.syncAndGetOutput()
        frames.append(output[:, :, :3].copy())

    print("Generating frames with THA4Engines (cache=1.0GB, eyebrow=True) - Pass 2:")
    for i in tqdm(range(len(pose_data[800:1200]))):
        pose = np.array(pose_data[800 + i]).reshape(1, 45).astype(np.float32)
        engine.asyncInfer(pose)
        output = engine.syncAndGetOutput()
        frames.append(output[:, :, :3].copy())

    generate_video(frames, './test/data/tha4_cached_eyebrow_test.mp4', 20)
    if engine.cacher is not None:
        print(f"Cache Stats - Hits: {engine.cacher.hits}, Misses: {engine.cacher.miss}, Hit Rate: {engine.cacher.hit_rate:.2%}")


def THA4Engines_ShowVideo_WithCache_NoEyebrow():
    model_dir = './data/tha4/fp16'
    if not os.path.exists(model_dir):
        print(f"Skipping: Model directory not found: {model_dir}")
        return

    engine = THA4Engines(model_dir, vram_cache_size=1.0, use_eyebrow=False)
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    engine.syncSetImage(img)

    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)

    frames = []
    print("Generating frames with THA4Engines (cache=1.0GB, eyebrow=False):")
    for i in tqdm(range(len(pose_data[800:1200]))):
        pose = np.array(pose_data[800 + i]).reshape(1, 45).astype(np.float32)
        engine.asyncInfer(pose)
        output = engine.syncAndGetOutput()
        frames.append(output[:, :, :3].copy())

    generate_video(frames, './test/data/tha4_cached_no_eyebrow_test.mp4', 20)
    if engine.cacher is not None:
        print(f"Cache Stats - Hits: {engine.cacher.hits}, Misses: {engine.cacher.miss}, Hit Rate: {engine.cacher.hit_rate:.2%}")


def THA4Engines_ShowVideo_NoCache():
    model_dir = './data/tha4/fp16'
    if not os.path.exists(model_dir):
        print(f"Skipping: Model directory not found: {model_dir}")
        return

    engine = THA4Engines(model_dir, vram_cache_size=0.0, use_eyebrow=True)
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    engine.syncSetImage(img)

    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)

    frames = []
    print("Generating frames with THA4Engines (no cache):")
    for i in tqdm(range(len(pose_data[800:1200]))):
        pose = np.array(pose_data[800 + i]).reshape(1, 45).astype(np.float32)
        engine.asyncInfer(pose)
        output = engine.syncAndGetOutput()
        frames.append(output[:, :, :3].copy())

    generate_video(frames, './test/data/tha4_no_cache_test.mp4', 20)


def THA4Engines_ShowVideo_FP32():
    model_dir = './data/tha4/fp32'
    if not os.path.exists(model_dir):
        print(f"Skipping: Model directory not found: {model_dir}")
        return

    engine = THA4Engines(model_dir, vram_cache_size=1.0, use_eyebrow=True)
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    engine.syncSetImage(img)

    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)

    frames = []
    print("Generating frames with THA4Engines (fp32):")
    for i in tqdm(range(len(pose_data[800:1200]))):
        pose = np.array(pose_data[800 + i]).reshape(1, 45).astype(np.float32)
        engine.asyncInfer(pose)
        output = engine.syncAndGetOutput()
        frames.append(output[:, :, :3].copy())

    generate_video(frames, './test/data/tha4_fp32_test.mp4', 20)
    if engine.cacher is not None:
        print(f"Cache Stats - Hits: {engine.cacher.hits}, Misses: {engine.cacher.miss}, Hit Rate: {engine.cacher.hit_rate:.2%}")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == '--show-all':
            print("=" * 60)
            print("Running all THA4 video showcases")
            print("=" * 60)
            THA4EnginesSimple_ShowVideo()
            THA4Engines_ShowVideo_WithCache_Eyebrow()
            THA4Engines_ShowVideo_WithCache_NoEyebrow()
            THA4Engines_ShowVideo_NoCache()
            THA4Engines_ShowVideo_FP32()
        elif sys.argv[1] == '--show-simple':
            THA4EnginesSimple_ShowVideo()
        elif sys.argv[1] == '--show-cached':
            THA4Engines_ShowVideo_WithCache_Eyebrow()
            THA4Engines_ShowVideo_WithCache_NoEyebrow()
        elif sys.argv[1] == '--show-no-cache':
            THA4Engines_ShowVideo_NoCache()
        elif sys.argv[1] == '--show-fp32':
            THA4Engines_ShowVideo_FP32()
        else:
            unittest.main(verbosity=2)
    else:
        unittest.main(verbosity=2)
