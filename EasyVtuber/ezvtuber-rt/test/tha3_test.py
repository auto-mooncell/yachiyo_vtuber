"""
Test cases for tha3.py - THA3Engines and THA3EnginesSimple classes (TensorRT-based)
Tests different model configurations, caching behavior, edge cases, and video generation
"""
import unittest
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import os
import json
from typing import List
from tqdm import tqdm

from ezvtb_rt.tha3 import THA3Engines, THA3EnginesSimple


class TestTHA3Base(unittest.TestCase):
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
        """Validate inference output format for THA3Engines (returns raw array, not list)"""
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(output.shape, (512, 512, 4))
        self.assertEqual(output.dtype, np.uint8)


# ============================================================================
# THA3EnginesSimple Tests
# ============================================================================

class TestTHA3EnginesSimpleSeperableFP16(TestTHA3Base):
    """Test THA3EnginesSimple with seperable/fp16 model"""
    
    MODEL_DIR = './data/tha3/seperable/fp16'
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA3EnginesSimple(cls.MODEL_DIR)
    
    def test_set_image_valid(self):
        """Test setting image with valid input"""
        self.engine.setImage(self.test_image)
        # No exception means success
    
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
        for i in range(10):
            pose = np.array(self.pose_data[800 + i]).reshape(1, 45).astype(np.float32)
            output = self.engine.inference(pose)
            self.validate_output(output)
    
    def test_engines_created(self):
        """Test that all engines are created"""
        self.assertIsNotNone(self.engine.decomposer)
        self.assertIsNotNone(self.engine.combiner)
        self.assertIsNotNone(self.engine.morpher)
        self.assertIsNotNone(self.engine.rotator)
        self.assertIsNotNone(self.engine.editor)
        self.assertIsNotNone(self.engine.stream)


class TestTHA3EnginesSimpleEdgeCases(TestTHA3Base):
    """Test edge cases for THA3EnginesSimple"""
    
    MODEL_DIR = './data/tha3/seperable/fp16'
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA3EnginesSimple(cls.MODEL_DIR)
    
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
        for _ in range(50):
            pose = self.create_random_pose()
            output = self.engine.inference(pose)
            self.validate_output(output)


class TestTHA3EnginesSimpleOutputConsistency(TestTHA3Base):
    """Test output consistency for THA3EnginesSimple"""
    
    MODEL_DIR = './data/tha3/seperable/fp16'
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA3EnginesSimple(cls.MODEL_DIR)
    
    def test_deterministic_output(self):
        """Test that same input produces consistent output"""
        self.engine.setImage(self.test_image)
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        
        output1 = self.engine.inference(pose).copy()
        output2 = self.engine.inference(pose).copy()
        
        # Same pose should produce same output
        np.testing.assert_array_equal(output1, output2)
    
    def test_different_poses_different_output(self):
        """Test that different poses produce different outputs"""
        self.engine.setImage(self.test_image)
        
        pose1 = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        pose2 = np.array(self.pose_data[850]).reshape(1, 45).astype(np.float32)
        
        output1 = self.engine.inference(pose1).copy()
        output2 = self.engine.inference(pose2).copy()
        
        # Different poses should produce different outputs
        self.assertFalse(np.array_equal(output1, output2))


# ============================================================================
# THA3Engines Tests (with VRAM caching)
# ============================================================================

class TestTHA3EnginesWithCacheEyebrow(TestTHA3Base):
    """Test THA3Engines with VRAM caching and eyebrow enabled"""
    
    MODEL_DIR = './data/tha3/seperable/fp16'
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA3Engines(cls.MODEL_DIR, vram_cache_size=1.0, use_eyebrow=True)
    
    def test_eyebrow_mode(self):
        """Test eyebrow mode is enabled"""
        self.assertTrue(self.engine.use_eyebrow)
    
    def test_cacher_created(self):
        """Test that cacher is created"""
        self.assertIsNotNone(self.engine.cacher)
    
    def test_sync_set_image(self):
        """Test syncSetImage"""
        self.engine.syncSetImage(self.test_image)
        # No exception means success
    
    def test_async_set_image(self):
        """Test asyncSetImage"""
        self.engine.asyncSetImage(self.test_image)
        self.engine.stream.synchronize()
        # No exception means success
    
    def test_inference_after_sync_set_image(self):
        """Test inference after syncSetImage"""
        self.engine.syncSetImage(self.test_image)
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        self.engine.asyncInfer(pose)
        output = self.engine.syncAndGetOutput()
        self.validate_output(output)
    
    def test_inference_with_zero_pose(self):
        """Test inference with zero pose"""
        self.engine.syncSetImage(self.test_image)
        self.engine.asyncInfer(self.create_zero_pose())
        output = self.engine.syncAndGetOutput()
        self.validate_output(output)
    
    def test_inference_with_random_pose(self):
        """Test inference with random pose values"""
        self.engine.syncSetImage(self.test_image)
        self.engine.asyncInfer(self.create_random_pose())
        output = self.engine.syncAndGetOutput()
        self.validate_output(output)
    
    def test_multiple_inferences(self):
        """Test multiple consecutive inferences"""
        self.engine.syncSetImage(self.test_image)
        for i in range(10):
            pose = np.array(self.pose_data[800 + i]).reshape(1, 45).astype(np.float32)
            self.engine.asyncInfer(pose)
            output = self.engine.syncAndGetOutput()
            self.validate_output(output)
    
    def test_get_output_mem(self):
        """Test getOutputMem returns valid memory"""
        self.engine.syncSetImage(self.test_image)
        self.engine.asyncInfer(self.create_zero_pose())
        output_mem = self.engine.getOutputMem()
        self.assertIsNotNone(output_mem)
        self.assertIsNotNone(output_mem.host)
        self.assertIsNotNone(output_mem.device)


class TestTHA3EnginesWithCacheNoEyebrow(TestTHA3Base):
    """Test THA3Engines with VRAM caching and eyebrow disabled"""
    
    MODEL_DIR = './data/tha3/seperable/fp16'
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA3Engines(cls.MODEL_DIR, vram_cache_size=1.0, use_eyebrow=False)
    
    def test_eyebrow_mode_disabled(self):
        """Test eyebrow mode is disabled"""
        self.assertFalse(self.engine.use_eyebrow)
    
    def test_inference_no_eyebrow(self):
        """Test inference without eyebrow"""
        self.engine.syncSetImage(self.test_image)
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        self.engine.asyncInfer(pose)
        output = self.engine.syncAndGetOutput()
        self.validate_output(output)
    
    def test_multiple_inferences_no_eyebrow(self):
        """Test multiple inferences without eyebrow"""
        self.engine.syncSetImage(self.test_image)
        for i in range(10):
            pose = np.array(self.pose_data[800 + i]).reshape(1, 45).astype(np.float32)
            self.engine.asyncInfer(pose)
            output = self.engine.syncAndGetOutput()
            self.validate_output(output)


class TestTHA3EnginesNoCache(TestTHA3Base):
    """Test THA3Engines without VRAM caching"""
    
    MODEL_DIR = './data/tha3/seperable/fp16'
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA3Engines(cls.MODEL_DIR, vram_cache_size=0.0, use_eyebrow=True)
    
    def test_cacher_disabled(self):
        """Test that cacher is disabled when vram_cache_size=0"""
        self.assertIsNone(self.engine.cacher)
    
    def test_inference_without_cache(self):
        """Test inference without caching"""
        self.engine.syncSetImage(self.test_image)
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        self.engine.asyncInfer(pose)
        output = self.engine.syncAndGetOutput()
        self.validate_output(output)
    
    def test_multiple_inferences_without_cache(self):
        """Test multiple inferences without caching"""
        self.engine.syncSetImage(self.test_image)
        for i in range(10):
            pose = np.array(self.pose_data[800 + i]).reshape(1, 45).astype(np.float32)
            self.engine.asyncInfer(pose)
            output = self.engine.syncAndGetOutput()
            self.validate_output(output)


class TestTHA3EnginesCacheBehavior(TestTHA3Base):
    """Test THA3Engines VRAM cache behavior"""
    
    MODEL_DIR = './data/tha3/seperable/fp16'
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        # Use fresh engine for cache testing
        cls.engine = THA3Engines(cls.MODEL_DIR, vram_cache_size=1.0, use_eyebrow=True)
    
    def setUp(self):
        """Reset cache before each test"""
        if self.engine.cacher is not None:
            self.engine.cacher.clear()
    
    def test_cache_hits_on_repeated_same_pose(self):
        """Test that cache hits when using same pose repeatedly"""
        self.engine.syncSetImage(self.test_image)
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        
        # First inference - cache miss
        self.engine.asyncInfer(pose)
        output1 = self.engine.syncAndGetOutput()
        
        # Second inference with same pose - cache hit
        self.engine.asyncInfer(pose)
        output2 = self.engine.syncAndGetOutput()
        
        self.validate_output(output1)
        self.validate_output(output2)
        
        # Check cache statistics
        self.assertGreater(self.engine.cacher.hits, 0)
    
    def test_cache_misses_on_different_poses(self):
        """Test that cache misses when using different poses"""
        self.engine.syncSetImage(self.test_image)
        
        initial_misses = self.engine.cacher.miss
        
        # Multiple different poses - all should miss
        for i in range(5):
            pose = np.array(self.pose_data[800 + i * 10]).reshape(1, 45).astype(np.float32)
            self.engine.asyncInfer(pose)
            self.engine.syncAndGetOutput()
        
        # Should have cache misses
        self.assertGreater(self.engine.cacher.miss, initial_misses)
    
    def test_cache_second_pass_has_hits(self):
        """Test that second pass over same data has cache hits"""
        self.engine.syncSetImage(self.test_image)
        poses = [np.array(self.pose_data[800 + i]).reshape(1, 45).astype(np.float32) for i in range(20)]
        
        # First pass - populates cache
        for pose in poses:
            self.engine.asyncInfer(pose)
            self.engine.syncAndGetOutput()
        
        hits_after_first_pass = self.engine.cacher.hits
        
        # Second pass - should have more hits
        for pose in poses:
            self.engine.asyncInfer(pose)
            self.engine.syncAndGetOutput()
        
        hits_after_second_pass = self.engine.cacher.hits
        self.assertGreater(hits_after_second_pass, hits_after_first_pass)
    
    def test_cache_hit_rate_property(self):
        """Test cache hit rate property"""
        self.engine.syncSetImage(self.test_image)
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        
        # Do some inferences
        for _ in range(5):
            self.engine.asyncInfer(pose)
            self.engine.syncAndGetOutput()
        
        hit_rate = self.engine.cacher.hit_rate
        self.assertGreaterEqual(hit_rate, 0.0)
        self.assertLessEqual(hit_rate, 1.0)


class TestTHA3EnginesEdgeCases(TestTHA3Base):
    """Test edge cases for THA3Engines"""
    
    MODEL_DIR = './data/tha3/seperable/fp16'
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA3Engines(cls.MODEL_DIR, vram_cache_size=0.5, use_eyebrow=True)
    
    def test_set_image_wrong_dimensions(self):
        """Test that wrong image dimensions raise AssertionError"""
        wrong_size_image = np.zeros((256, 256, 4), dtype=np.uint8)
        with self.assertRaises(AssertionError):
            self.engine.syncSetImage(wrong_size_image)
    
    def test_set_image_wrong_channels(self):
        """Test that wrong channel count raises AssertionError"""
        wrong_channels_image = np.zeros((512, 512, 3), dtype=np.uint8)
        with self.assertRaises(AssertionError):
            self.engine.syncSetImage(wrong_channels_image)
    
    def test_set_image_2d_array(self):
        """Test that 2D array raises AssertionError"""
        wrong_shape_image = np.zeros((512, 512), dtype=np.uint8)
        with self.assertRaises(AssertionError):
            self.engine.syncSetImage(wrong_shape_image)
    
    def test_set_image_wrong_dtype(self):
        """Test that wrong dtype raises AssertionError"""
        wrong_dtype_image = np.zeros((512, 512, 4), dtype=np.float32)
        with self.assertRaises(AssertionError):
            self.engine.syncSetImage(wrong_dtype_image)
    
    def test_pose_extreme_values(self):
        """Test inference with extreme pose values"""
        self.engine.syncSetImage(self.test_image)
        extreme_pose = np.ones((1, 45), dtype=np.float32) * 100.0
        self.engine.asyncInfer(extreme_pose)
        output = self.engine.syncAndGetOutput()
        self.validate_output(output)
    
    def test_pose_negative_values(self):
        """Test inference with negative pose values"""
        self.engine.syncSetImage(self.test_image)
        negative_pose = np.ones((1, 45), dtype=np.float32) * -50.0
        self.engine.asyncInfer(negative_pose)
        output = self.engine.syncAndGetOutput()
        self.validate_output(output)
    
    def test_image_update_between_inferences(self):
        """Test updating image between inferences"""
        self.engine.syncSetImage(self.test_image)
        pose = self.create_zero_pose()
        self.engine.asyncInfer(pose)
        output1 = self.engine.syncAndGetOutput()
        self.validate_output(output1)
        
        # Update with same image (simulating image change)
        self.engine.syncSetImage(self.test_image)
        self.engine.asyncInfer(pose)
        output2 = self.engine.syncAndGetOutput()
        self.validate_output(output2)
    
    def test_rapid_pose_changes(self):
        """Test rapid pose changes in succession"""
        self.engine.syncSetImage(self.test_image)
        for _ in range(50):
            pose = self.create_random_pose()
            self.engine.asyncInfer(pose)
            output = self.engine.syncAndGetOutput()
            self.validate_output(output)


class TestTHA3EnginesOutputConsistency(TestTHA3Base):
    """Test output consistency for THA3Engines"""
    
    MODEL_DIR = './data/tha3/seperable/fp16'
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA3Engines(cls.MODEL_DIR, vram_cache_size=1.0, use_eyebrow=True)
    
    def test_deterministic_output(self):
        """Test that same input produces consistent output"""
        self.engine.syncSetImage(self.test_image)
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        
        self.engine.asyncInfer(pose)
        output1 = self.engine.syncAndGetOutput().copy()
        
        self.engine.asyncInfer(pose)
        output2 = self.engine.syncAndGetOutput().copy()
        
        # Same pose should produce same output
        np.testing.assert_array_equal(output1, output2)
    
    def test_different_poses_different_output(self):
        """Test that different poses produce different outputs"""
        self.engine.syncSetImage(self.test_image)
        
        pose1 = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        pose2 = np.array(self.pose_data[850]).reshape(1, 45).astype(np.float32)
        
        self.engine.asyncInfer(pose1)
        output1 = self.engine.syncAndGetOutput().copy()
        
        self.engine.asyncInfer(pose2)
        output2 = self.engine.syncAndGetOutput().copy()
        
        # Different poses should produce different outputs
        self.assertFalse(np.array_equal(output1, output2))


class TestTHA3EnginesPoseSlicing(TestTHA3Base):
    """Test that pose slicing works correctly for THA3Engines"""
    
    MODEL_DIR = './data/tha3/seperable/fp16'
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA3Engines(cls.MODEL_DIR, vram_cache_size=0.5, use_eyebrow=True)
    
    def test_eyebrow_pose_slicing(self):
        """Test that eyebrow pose (first 12 values) is correctly used"""
        self.engine.syncSetImage(self.test_image)
        
        pose = np.arange(45, dtype=np.float32).reshape(1, 45)
        # This should use pose[:, :12] for eyebrow
        self.engine.asyncInfer(pose)
        output = self.engine.syncAndGetOutput()
        self.validate_output(output)
    
    def test_face_pose_slicing(self):
        """Test that face pose (values 12-39) is correctly used"""
        self.engine.syncSetImage(self.test_image)
        
        pose = np.arange(45, dtype=np.float32).reshape(1, 45)
        # This should use pose[:,12:12+27] for face pose
        self.engine.asyncInfer(pose)
        output = self.engine.syncAndGetOutput()
        self.validate_output(output)
    
    def test_rotation_pose_slicing(self):
        """Test that rotation pose (last 6 values) is correctly used"""
        self.engine.syncSetImage(self.test_image)
        
        pose = np.arange(45, dtype=np.float32).reshape(1, 45)
        # This should use pose[:,12+27:] for rotation pose
        self.engine.asyncInfer(pose)
        output = self.engine.syncAndGetOutput()
        self.validate_output(output)


class TestTHA3EnginesCompareSimpleVsCached(TestTHA3Base):
    """Compare outputs between THA3EnginesSimple and THA3Engines"""
    
    MODEL_DIR = './data/tha3/seperable/fp16'
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine_simple = THA3EnginesSimple(cls.MODEL_DIR)
        cls.engine_cached = THA3Engines(cls.MODEL_DIR, vram_cache_size=1.0, use_eyebrow=True)
    
    def test_both_engines_produce_valid_output(self):
        """Test that both engines produce valid outputs"""
        self.engine_simple.setImage(self.test_image)
        self.engine_cached.syncSetImage(self.test_image)
        
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        
        output_simple = self.engine_simple.inference(pose)
        
        self.engine_cached.asyncInfer(pose)
        output_cached = self.engine_cached.syncAndGetOutput()
        
        self.validate_output(output_simple)
        self.validate_output(output_cached)
    
    def test_outputs_are_similar(self):
        """Test that both engines produce similar outputs"""
        self.engine_simple.setImage(self.test_image)
        self.engine_cached.syncSetImage(self.test_image)
        
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        
        output_simple = self.engine_simple.inference(pose)
        
        self.engine_cached.asyncInfer(pose)
        output_cached = self.engine_cached.syncAndGetOutput()
        
        # Allow small differences due to potential numerical variations
        diff = np.abs(output_simple.astype(np.float32) - output_cached.astype(np.float32))
        mean_diff = np.mean(diff)
        
        # Mean difference should be small
        self.assertLess(mean_diff, 5.0, "Outputs differ significantly between simple and cached engines")


class TestTHA3EnginesEyebrowVsNoEyebrow(TestTHA3Base):
    """Compare outputs between eyebrow and no-eyebrow modes"""
    
    MODEL_DIR = './data/tha3/seperable/fp16'
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine_eyebrow = THA3Engines(cls.MODEL_DIR, vram_cache_size=0.5, use_eyebrow=True)
        cls.engine_no_eyebrow = THA3Engines(cls.MODEL_DIR, vram_cache_size=0.5, use_eyebrow=False)
    
    def test_both_modes_produce_valid_output(self):
        """Test that both eyebrow modes produce valid outputs"""
        self.engine_eyebrow.syncSetImage(self.test_image)
        self.engine_no_eyebrow.syncSetImage(self.test_image)
        
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        
        self.engine_eyebrow.asyncInfer(pose)
        output_eyebrow = self.engine_eyebrow.syncAndGetOutput()
        
        self.engine_no_eyebrow.asyncInfer(pose)
        output_no_eyebrow = self.engine_no_eyebrow.syncAndGetOutput()
        
        self.validate_output(output_eyebrow)
        self.validate_output(output_no_eyebrow)


class TestTHA3EnginesAllConfigurations(TestTHA3Base):
    """Test THA3Engines across available model configurations"""
    
    MODEL_CONFIGS = [
        ('./data/tha3/seperable/fp16', True, 1.0),   # seperable, fp16, with cache
        ('./data/tha3/seperable/fp16', True, 0.0),   # seperable, fp16, no cache
        ('./data/tha3/seperable/fp16', False, 1.0),  # seperable, fp16, no eyebrow
        ('./data/tha3/seperable/fp32', True, 1.0),   # seperable, fp32, with cache
        ('./data/tha3/standard/fp16', True, 1.0),    # standard, fp16, with cache
        ('./data/tha3/standard/fp32', True, 1.0),    # standard, fp32, with cache
    ]
    
    def test_all_configurations(self):
        """Test THA3Engines with all available configurations"""
        for model_dir, use_eyebrow, cache_size in self.MODEL_CONFIGS:
            if not os.path.exists(model_dir):
                continue
            
            with self.subTest(model_dir=model_dir, use_eyebrow=use_eyebrow, cache_size=cache_size):
                engine = THA3Engines(model_dir, vram_cache_size=cache_size, use_eyebrow=use_eyebrow)
                
                # Verify configuration
                self.assertEqual(engine.use_eyebrow, use_eyebrow)
                if cache_size > 0:
                    self.assertIsNotNone(engine.cacher)
                else:
                    self.assertIsNone(engine.cacher)
                
                # Test inference
                engine.syncSetImage(self.test_image)
                pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
                engine.asyncInfer(pose)
                output = engine.syncAndGetOutput()
                self.validate_output(output)


class TestTHA3EnginesAsyncStream(TestTHA3Base):
    """Test THA3Engines async stream functionality"""
    
    MODEL_DIR = './data/tha3/seperable/fp16'
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA3Engines(cls.MODEL_DIR, vram_cache_size=1.0, use_eyebrow=True)
    
    def test_custom_stream(self):
        """Test inference with custom CUDA stream"""
        self.engine.syncSetImage(self.test_image)
        custom_stream = cuda.Stream()
        
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        self.engine.asyncInfer(pose, stream=custom_stream)
        
        # Synchronize custom stream
        custom_stream.synchronize()
        self.engine.finishedFetch.synchronize()
        
        output = self.engine.editor.outputs[1].host
        self.validate_output(output)
    
    def test_multiple_async_inferences(self):
        """Test multiple async inferences"""
        self.engine.syncSetImage(self.test_image)
        
        for i in range(10):
            pose = np.array(self.pose_data[800 + i]).reshape(1, 45).astype(np.float32)
            self.engine.asyncInfer(pose)
            output = self.engine.syncAndGetOutput()
            self.validate_output(output)


# ============================================================================
# Video Generation Showcase Functions
# ============================================================================

def generate_video(imgs: List[np.ndarray], video_path: str, framerate: float):
    """Generate video from a list of images
    Args:
        imgs: List of images in opencv format (BGR)
        video_path: Output video path
        framerate: Video framerate
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_path, fourcc, framerate, (imgs[0].shape[1], imgs[0].shape[0]))
    if not video.isOpened():
        raise ValueError("CV2 video encoder Not supported")

    for img in imgs:
        video.write(img)

    video.release()
    cv2.destroyAllWindows()
    print(f"Video generated successfully at {video_path}!")


def THA3EnginesSimple_ShowVideo():
    """Generate a test video using THA3EnginesSimple (no caching)"""
    model_dir = './data/tha3/seperable/fp16'
    if not os.path.exists(model_dir):
        print(f"Skipping: Model directory not found: {model_dir}")
        return
    
    engine = THA3EnginesSimple(model_dir)
    
    # Load test image (512x512 RGBA)
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    engine.setImage(img)
    
    # Load pose data
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    
    frames = []
    print("Generating frames with THA3EnginesSimple:")
    for i in tqdm(range(len(pose_data[800:1200]))):
        pose = np.array(pose_data[800 + i]).reshape(1, 45).astype(np.float32)
        output = engine.inference(pose)
        # Convert RGBA to BGR for opencv
        frames.append(output[:, :, :3].copy())
    
    generate_video(frames, './test/data/tha3_simple_test.mp4', 20)


def THA3Engines_ShowVideo_WithCache_Eyebrow():
    """Generate a test video using THA3Engines with cache and eyebrow enabled"""
    model_dir = './data/tha3/seperable/fp16'
    if not os.path.exists(model_dir):
        print(f"Skipping: Model directory not found: {model_dir}")
        return
    
    engine = THA3Engines(model_dir, vram_cache_size=1.0, use_eyebrow=True)
    
    # Load test image (512x512 RGBA)
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    engine.syncSetImage(img)
    
    # Load pose data
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    
    frames = []
    print("Generating frames with THA3Engines (cache=1.0GB, eyebrow=True) - Pass 1:")
    for i in tqdm(range(len(pose_data[800:1200]))):
        pose = np.array(pose_data[800 + i]).reshape(1, 45).astype(np.float32)
        engine.asyncInfer(pose)
        output = engine.syncAndGetOutput()
        frames.append(output[:, :, :3].copy())
    
    # Second pass to show cache effectiveness
    print("Generating frames with THA3Engines (cache=1.0GB, eyebrow=True) - Pass 2:")
    for i in tqdm(range(len(pose_data[800:1200]))):
        pose = np.array(pose_data[800 + i]).reshape(1, 45).astype(np.float32)
        engine.asyncInfer(pose)
        output = engine.syncAndGetOutput()
        frames.append(output[:, :, :3].copy())
    
    generate_video(frames, './test/data/tha3_cached_eyebrow_test.mp4', 20)
    
    # Print cache statistics
    if engine.cacher is not None:
        print(f"\nCache Stats - Hits: {engine.cacher.hits}, Misses: {engine.cacher.miss}, Hit Rate: {engine.cacher.hit_rate:.2%}")


def THA3Engines_ShowVideo_WithCache_NoEyebrow():
    """Generate a test video using THA3Engines with cache and eyebrow disabled"""
    model_dir = './data/tha3/seperable/fp16'
    if not os.path.exists(model_dir):
        print(f"Skipping: Model directory not found: {model_dir}")
        return
    
    engine = THA3Engines(model_dir, vram_cache_size=1.0, use_eyebrow=False)
    
    # Load test image (512x512 RGBA)
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    engine.syncSetImage(img)
    
    # Load pose data
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    
    frames = []
    print("Generating frames with THA3Engines (cache=1.0GB, eyebrow=False):")
    for i in tqdm(range(len(pose_data[800:1200]))):
        pose = np.array(pose_data[800 + i]).reshape(1, 45).astype(np.float32)
        engine.asyncInfer(pose)
        output = engine.syncAndGetOutput()
        frames.append(output[:, :, :3].copy())
    
    generate_video(frames, './test/data/tha3_cached_no_eyebrow_test.mp4', 20)
    
    # Print cache statistics
    if engine.cacher is not None:
        print(f"\nCache Stats - Hits: {engine.cacher.hits}, Misses: {engine.cacher.miss}, Hit Rate: {engine.cacher.hit_rate:.2%}")


def THA3Engines_ShowVideo_NoCache():
    """Generate a test video using THA3Engines without cache"""
    model_dir = './data/tha3/seperable/fp16'
    if not os.path.exists(model_dir):
        print(f"Skipping: Model directory not found: {model_dir}")
        return
    
    engine = THA3Engines(model_dir, vram_cache_size=0.0, use_eyebrow=True)
    
    # Load test image (512x512 RGBA)
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    engine.syncSetImage(img)
    
    # Load pose data
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    
    frames = []
    print("Generating frames with THA3Engines (no cache):")
    for i in tqdm(range(len(pose_data[800:1200]))):
        pose = np.array(pose_data[800 + i]).reshape(1, 45).astype(np.float32)
        engine.asyncInfer(pose)
        output = engine.syncAndGetOutput()
        frames.append(output[:, :, :3].copy())
    
    generate_video(frames, './test/data/tha3_no_cache_test.mp4', 20)


def THA3Engines_ShowVideo_StandardFP16():
    """Generate a test video using THA3Engines with standard/fp16 model"""
    model_dir = './data/tha3/standard/fp16'
    if not os.path.exists(model_dir):
        print(f"Skipping: Model directory not found: {model_dir}")
        return
    
    engine = THA3Engines(model_dir, vram_cache_size=1.0, use_eyebrow=True)
    
    # Load test image (512x512 RGBA)
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    engine.syncSetImage(img)
    
    # Load pose data
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    
    frames = []
    print("Generating frames with THA3Engines (standard/fp16):")
    for i in tqdm(range(len(pose_data[800:1200]))):
        pose = np.array(pose_data[800 + i]).reshape(1, 45).astype(np.float32)
        engine.asyncInfer(pose)
        output = engine.syncAndGetOutput()
        frames.append(output[:, :, :3].copy())
    
    generate_video(frames, './test/data/tha3_standard_fp16_test.mp4', 20)
    
    # Print cache statistics
    if engine.cacher is not None:
        print(f"\nCache Stats - Hits: {engine.cacher.hits}, Misses: {engine.cacher.miss}, Hit Rate: {engine.cacher.hit_rate:.2%}")


def THA3Engines_ShowVideo_SeperableFP32():
    """Generate a test video using THA3Engines with seperable/fp32 model"""
    model_dir = './data/tha3/seperable/fp32'
    if not os.path.exists(model_dir):
        print(f"Skipping: Model directory not found: {model_dir}")
        return
    
    engine = THA3Engines(model_dir, vram_cache_size=1.0, use_eyebrow=True)
    
    # Load test image (512x512 RGBA)
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    engine.syncSetImage(img)
    
    # Load pose data
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    
    frames = []
    print("Generating frames with THA3Engines (seperable/fp32):")
    for i in tqdm(range(len(pose_data[800:1200]))):
        pose = np.array(pose_data[800 + i]).reshape(1, 45).astype(np.float32)
        engine.asyncInfer(pose)
        output = engine.syncAndGetOutput()
        frames.append(output[:, :, :3].copy())
    
    generate_video(frames, './test/data/tha3_seperable_fp32_test.mp4', 20)
    
    # Print cache statistics
    if engine.cacher is not None:
        print(f"\nCache Stats - Hits: {engine.cacher.hits}, Misses: {engine.cacher.miss}, Hit Rate: {engine.cacher.hit_rate:.2%}")


def THA3Engines_Performance_Comparison():
    """Compare performance between cached and non-cached engines"""
    import time
    
    model_dir = './data/tha3/seperable/fp16'
    if not os.path.exists(model_dir):
        print(f"Skipping: Model directory not found: {model_dir}")
        return
    
    # Load test data
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    
    poses = [np.array(pose_data[800 + i]).reshape(1, 45).astype(np.float32) for i in range(200)]
    
    # Test Simple Engine
    print("=" * 60)
    print("THA3EnginesSimple Performance:")
    print("=" * 60)
    engine_simple = THA3EnginesSimple(model_dir)
    engine_simple.setImage(img)
    
    start = time.time()
    for pose in tqdm(poses):
        engine_simple.inference(pose)
    simple_time = time.time() - start
    print(f"Time: {simple_time:.2f}s, FPS: {len(poses)/simple_time:.1f}")
    
    # Test Cached Engine - First Pass
    print("\n" + "=" * 60)
    print("THA3Engines (cached) Performance - First Pass:")
    print("=" * 60)
    engine_cached = THA3Engines(model_dir, vram_cache_size=1.0, use_eyebrow=True)
    engine_cached.syncSetImage(img)
    
    start = time.time()
    for pose in tqdm(poses):
        engine_cached.asyncInfer(pose)
        engine_cached.syncAndGetOutput()
    cached_first_time = time.time() - start
    print(f"Time: {cached_first_time:.2f}s, FPS: {len(poses)/cached_first_time:.1f}")
    print(f"Cache Stats - Hits: {engine_cached.cacher.hits}, Misses: {engine_cached.cacher.miss}")
    
    # Test Cached Engine - Second Pass (cache should be warm)
    print("\n" + "=" * 60)
    print("THA3Engines (cached) Performance - Second Pass:")
    print("=" * 60)
    
    start = time.time()
    for pose in tqdm(poses):
        engine_cached.asyncInfer(pose)
        engine_cached.syncAndGetOutput()
    cached_second_time = time.time() - start
    print(f"Time: {cached_second_time:.2f}s, FPS: {len(poses)/cached_second_time:.1f}")
    print(f"Cache Stats - Hits: {engine_cached.cacher.hits}, Misses: {engine_cached.cacher.miss}, Hit Rate: {engine_cached.cacher.hit_rate:.2%}")
    
    # Test No-Cache Engine
    print("\n" + "=" * 60)
    print("THA3Engines (no cache) Performance:")
    print("=" * 60)
    engine_no_cache = THA3Engines(model_dir, vram_cache_size=0.0, use_eyebrow=True)
    engine_no_cache.syncSetImage(img)
    
    start = time.time()
    for pose in tqdm(poses):
        engine_no_cache.asyncInfer(pose)
        engine_no_cache.syncAndGetOutput()
    no_cache_time = time.time() - start
    print(f"Time: {no_cache_time:.2f}s, FPS: {len(poses)/no_cache_time:.1f}")


if __name__ == '__main__':
    import sys
    
    # Check for command line arguments for video generation
    if len(sys.argv) > 1:
        if sys.argv[1] == '--show-all':
            # Run all video showcase functions
            print("=" * 60)
            print("Running all THA3 video showcase tests")
            print("=" * 60)
            THA3EnginesSimple_ShowVideo()
            THA3Engines_ShowVideo_WithCache_Eyebrow()
            THA3Engines_ShowVideo_WithCache_NoEyebrow()
            THA3Engines_ShowVideo_NoCache()
            THA3Engines_ShowVideo_StandardFP16()
            THA3Engines_ShowVideo_SeperableFP32()
        elif sys.argv[1] == '--show-simple':
            print("=" * 60)
            print("Running THA3EnginesSimple video showcase")
            print("=" * 60)
            THA3EnginesSimple_ShowVideo()
        elif sys.argv[1] == '--show-cached':
            print("=" * 60)
            print("Running THA3Engines cached video showcases")
            print("=" * 60)
            THA3Engines_ShowVideo_WithCache_Eyebrow()
            THA3Engines_ShowVideo_WithCache_NoEyebrow()
        elif sys.argv[1] == '--show-no-cache':
            print("=" * 60)
            print("Running THA3Engines no-cache video showcase")
            print("=" * 60)
            THA3Engines_ShowVideo_NoCache()
        elif sys.argv[1] == '--show-models':
            print("=" * 60)
            print("Running THA3Engines different model video showcases")
            print("=" * 60)
            THA3Engines_ShowVideo_StandardFP16()
            THA3Engines_ShowVideo_SeperableFP32()
        elif sys.argv[1] == '--perf':
            print("=" * 60)
            print("Running THA3 Performance Comparison")
            print("=" * 60)
            THA3Engines_Performance_Comparison()
        else:
            # Run unit tests with remaining arguments
            unittest.main(verbosity=2)
    else:
        # Run unit tests with verbosity
        unittest.main(verbosity=2)
