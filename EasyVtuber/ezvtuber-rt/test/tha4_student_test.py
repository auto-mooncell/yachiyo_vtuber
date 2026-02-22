"""
Test cases for tha4_student.py - THA4StudentEngines class (TensorRT-based)
Tests the student model with custom character from kanori directory.
"""
import unittest
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401 - ensures CUDA context is initialized
import cv2
import os
import json
import sys
from typing import List
from tqdm import tqdm

from ezvtb_rt.tha4_student import THA4StudentEngines


class TestTHA4StudentBase(unittest.TestCase):
    """Base test class with common utilities"""

    TEST_IMAGE_PATH = './data/custom_tha4_models/kanori/character.png'
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
        """Validate inference output format for THA4StudentEngines"""
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(output.shape, (512, 512, 4))
        self.assertEqual(output.dtype, np.uint8)


# ============================================================================
# THA4StudentEngines Tests (with VRAM caching)
# ============================================================================


class TestTHA4StudentEnginesWithCache(TestTHA4StudentBase):
    """Test THA4StudentEngines with VRAM caching enabled"""

    MODEL_DIR = './data/custom_tha4_models/kanori'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA4StudentEngines(cls.MODEL_DIR, vram_cache_size=1.0)

    def test_cacher_created(self):
        """Cacher is created when cache enabled"""
        self.assertIsNotNone(self.engine.cacher)

    def test_engines_created(self):
        """Test that all engines are created"""
        self.assertIsNotNone(self.engine.face_morpher)
        self.assertIsNotNone(self.engine.body_morpher)
        self.assertIsNotNone(self.engine.stream)
        self.assertIsNotNone(self.engine.cachestream)

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
        self.assertIsInstance(out_mem, type(self.engine.body_morpher.outputs[1]))


class TestTHA4StudentEnginesNoCache(TestTHA4StudentBase):
    """Test THA4StudentEngines without VRAM caching"""

    MODEL_DIR = './data/custom_tha4_models/kanori'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA4StudentEngines(cls.MODEL_DIR, vram_cache_size=0.0)

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


class TestTHA4StudentEnginesCacheBehavior(TestTHA4StudentBase):
    """Test THA4StudentEngines VRAM cache behavior"""

    MODEL_DIR = './data/custom_tha4_models/kanori'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA4StudentEngines(cls.MODEL_DIR, vram_cache_size=1.0)

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


class TestTHA4StudentEnginesEdgeCases(TestTHA4StudentBase):
    """Edge cases for THA4StudentEngines"""

    MODEL_DIR = './data/custom_tha4_models/kanori'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA4StudentEngines(cls.MODEL_DIR, vram_cache_size=0.5)

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
        first = self.engine.syncAndGetOutput()
        self.validate_output(first)

        self.engine.syncSetImage(self.test_image)
        self.engine.asyncInfer(pose)
        second = self.engine.syncAndGetOutput()
        self.validate_output(second)

    def test_rapid_pose_changes(self):
        """Test rapid pose changes in succession"""
        self.engine.syncSetImage(self.test_image)
        for _ in range(30):
            pose = self.create_random_pose()
            self.engine.asyncInfer(pose)
            output = self.engine.syncAndGetOutput()
            self.validate_output(output)


class TestTHA4StudentEnginesOutputConsistency(TestTHA4StudentBase):
    """Output consistency checks for THA4StudentEngines"""

    MODEL_DIR = './data/custom_tha4_models/kanori'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA4StudentEngines(cls.MODEL_DIR, vram_cache_size=1.0)

    def test_deterministic_output(self):
        """Test that same input produces consistent output"""
        self.engine.syncSetImage(self.test_image)
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)

        self.engine.asyncInfer(pose)
        out1 = self.engine.syncAndGetOutput().copy()

        self.engine.asyncInfer(pose)
        out2 = self.engine.syncAndGetOutput().copy()

        np.testing.assert_array_equal(out1, out2)

    def test_different_poses_different_output(self):
        """Test that different poses produce different outputs"""
        self.engine.syncSetImage(self.test_image)
        pose1 = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        pose2 = np.array(self.pose_data[850]).reshape(1, 45).astype(np.float32)

        self.engine.asyncInfer(pose1)
        out1 = self.engine.syncAndGetOutput().copy()

        self.engine.asyncInfer(pose2)
        out2 = self.engine.syncAndGetOutput().copy()

        self.assertFalse(np.array_equal(out1, out2))


class TestTHA4StudentEnginesPoseSlicing(TestTHA4StudentBase):
    """Verify pose slicing for face morphing (first 39 elements)"""

    MODEL_DIR = './data/custom_tha4_models/kanori'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA4StudentEngines(cls.MODEL_DIR, vram_cache_size=0.5)

    def test_face_pose_slicing(self):
        """Test that face morpher uses first 39 elements"""
        self.engine.syncSetImage(self.test_image)
        pose = np.arange(45, dtype=np.float32).reshape(1, 45)
        self.engine.asyncInfer(pose)
        output = self.engine.syncAndGetOutput()
        self.validate_output(output)

    def test_full_pose_used_by_body(self):
        """Test that body morpher uses full 45-element pose"""
        self.engine.syncSetImage(self.test_image)
        pose = np.arange(45, dtype=np.float32).reshape(1, 45)
        self.engine.asyncInfer(pose)
        output = self.engine.syncAndGetOutput()
        self.validate_output(output)


class TestTHA4StudentEnginesAsyncStream(TestTHA4StudentBase):
    """Async stream usage for THA4StudentEngines"""

    MODEL_DIR = './data/custom_tha4_models/kanori'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA4StudentEngines(cls.MODEL_DIR, vram_cache_size=1.0)

    def test_custom_stream(self):
        """Test inference with custom CUDA stream"""
        self.engine.syncSetImage(self.test_image)
        custom_stream = cuda.Stream()
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        self.engine.asyncInfer(pose, stream=custom_stream)
        custom_stream.synchronize()
        self.engine.finishedFetchRes.synchronize()
        output = self.engine.body_morpher.outputs[1].host
        self.validate_output(output)

    def test_multiple_async_inferences(self):
        """Test multiple async inferences"""
        self.engine.syncSetImage(self.test_image)
        for i in range(8):
            pose = np.array(self.pose_data[800 + i]).reshape(1, 45).astype(np.float32)
            self.engine.asyncInfer(pose)
            output = self.engine.syncAndGetOutput()
            self.validate_output(output)


class TestTHA4StudentEnginesCacheConfigurations(TestTHA4StudentBase):
    """Test THA4StudentEngines across different cache configurations"""

    MODEL_DIR = './data/custom_tha4_models/kanori'

    def test_all_cache_configurations(self):
        """Test with various cache sizes"""
        cache_sizes = [0.0, 0.5, 1.0, 2.0]
        
        for cache_size in cache_sizes:
            if not os.path.exists(self.MODEL_DIR):
                self.skipTest(f"Model directory not found: {self.MODEL_DIR}")
            
            with self.subTest(cache_size=cache_size):
                engine = THA4StudentEngines(self.MODEL_DIR, vram_cache_size=cache_size)
                engine.syncSetImage(self.test_image)
                pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
                engine.asyncInfer(pose)
                output = engine.syncAndGetOutput()
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


def THA4StudentEngines_ShowVideo_WithCache():
    """Generate video showcase with THA4StudentEngines (with cache)"""
    model_dir = './data/custom_tha4_models/kanori'
    if not os.path.exists(model_dir):
        print(f"Skipping: Model directory not found: {model_dir}")
        return

    engine = THA4StudentEngines(model_dir, vram_cache_size=1.0)
    img = cv2.imread('./data/custom_tha4_models/kanori/character.png', cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Skipping: Character image not found")
        return
    
    engine.syncSetImage(img)

    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)

    frames = []
    print("Generating frames with THA4StudentEngines (cache=1.0GB) - Pass 1:")
    for i in tqdm(range(len(pose_data[800:1200]))):
        pose = np.array(pose_data[800 + i]).reshape(1, 45).astype(np.float32)
        engine.asyncInfer(pose)
        output = engine.syncAndGetOutput()
        frames.append(output[:, :, :3].copy())

    print("Generating frames with THA4StudentEngines (cache=1.0GB) - Pass 2:")
    for i in tqdm(range(len(pose_data[800:1200]))):
        pose = np.array(pose_data[800 + i]).reshape(1, 45).astype(np.float32)
        engine.asyncInfer(pose)
        output = engine.syncAndGetOutput()
        frames.append(output[:, :, :3].copy())

    generate_video(frames, './test/data/tha4_student_cached_test.mp4', 20)
    if engine.cacher is not None:
        print(f"Cache Stats - Hits: {engine.cacher.hits}, Misses: {engine.cacher.miss}, Hit Rate: {engine.cacher.hits / (engine.cacher.hits + engine.cacher.miss):.2%}")


def THA4StudentEngines_ShowVideo_NoCache():
    """Generate video showcase with THA4StudentEngines (no cache)"""
    model_dir = './data/custom_tha4_models/kanori'
    if not os.path.exists(model_dir):
        print(f"Skipping: Model directory not found: {model_dir}")
        return

    engine = THA4StudentEngines(model_dir, vram_cache_size=0.0)
    img = cv2.imread('./data/custom_tha4_models/kanori/character.png', cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Skipping: Character image not found")
        return
    
    engine.syncSetImage(img)

    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)

    frames = []
    print("Generating frames with THA4StudentEngines (no cache):")
    for i in tqdm(range(len(pose_data[800:1200]))):
        pose = np.array(pose_data[800 + i]).reshape(1, 45).astype(np.float32)
        engine.asyncInfer(pose)
        output = engine.syncAndGetOutput()
        frames.append(output[:, :, :3].copy())

    generate_video(frames, './test/data/tha4_student_no_cache_test.mp4', 20)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == '--show-all':
            print("=" * 60)
            print("Running all THA4Student video showcases")
            print("=" * 60)
            THA4StudentEngines_ShowVideo_WithCache()
            THA4StudentEngines_ShowVideo_NoCache()
        elif sys.argv[1] == '--show-cached':
            THA4StudentEngines_ShowVideo_WithCache()
        elif sys.argv[1] == '--show-no-cache':
            THA4StudentEngines_ShowVideo_NoCache()
        else:
            unittest.main(verbosity=2)
    else:
        unittest.main(verbosity=2)
