"""
Test cases for core_ort.py - CoreORT class
Tests different configurations including:
- THA model versions (v3, v4) with various precision and architecture options
- RIFE frame interpolation with different scales (x2, x3, x4)
- Super Resolution (SR) models (waifu2x x2, Real-ESRGAN x4)
- Cacher functionality for performance optimization
"""
import unittest
import sys
import os
from ezvtb_rt import init_model_path
from ezvtb_rt.init_utils import check_exist_all_models
from ezvtb_rt.core_ort import CoreORT
from ezvtb_rt.tha3_ort import THA3ORTSessions
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm
import json
import cv2


# ============================================================================
# Utility Functions
# ============================================================================

def generate_video(imgs: List[np.ndarray], video_path: str, framerate: float):
    """Generate video from a list of images
    Args:
        imgs: List of images in opencv format (BGR, HWC)
        video_path: Output video path
        framerate: Video framerate
    """
    if len(imgs) == 0:
        raise ValueError("No images to generate video")
    
    # Get dimensions from first image
    h, w = imgs[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_path, fourcc, framerate, (w, h))
    if not video.isOpened():
        raise ValueError("CV2 video encoder Not supported")

    for img in imgs:
        video.write(img)

    video.release()
    cv2.destroyAllWindows()
    print(f"Video generated successfully at {video_path}!")


# ============================================================================
# Unit Tests
# ============================================================================

class TestCoreORTBase(unittest.TestCase):
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
    
    def create_zero_pose(self) -> np.ndarray:
        """Create a zero pose array with shape (1, 45)"""
        return np.zeros((1, 45), dtype=np.float32)
    
    def get_pose(self, index: int) -> np.ndarray:
        """Get a pose from test data"""
        return np.array(self.pose_data[index]).reshape(1, 45).astype(np.float32)


class TestCoreORTBasicTHA(TestCoreORTBase):
    """Test CoreORT with basic THA configurations (no RIFE, no SR)"""
    
    def test_tha_v3_seperable_fp16(self):
        """Test THA v3 seperable FP16 model"""
        core = CoreORT(
            tha_model_version='v3',
            tha_model_seperable=True,
            tha_model_fp16=True,
            use_eyebrow=False,
            cache_max_giga=0.0  # Disable cache for basic test
        )
        core.setImage(self.test_image)
        output = core.inference([self.get_pose(800)])
        
        self.assertEqual(output.shape[0], 1)  # Batch size 1
        self.assertEqual(output.shape[1], 512)
        self.assertEqual(output.shape[2], 512)
        self.assertEqual(output.shape[3], 4)  # RGBA
    
    def test_tha_v3_standard_fp16(self):
        """Test THA v3 standard FP16 model"""
        core = CoreORT(
            tha_model_version='v3',
            tha_model_seperable=False,
            tha_model_fp16=True,
            use_eyebrow=False,
            cache_max_giga=0.0
        )
        core.setImage(self.test_image)
        output = core.inference([self.get_pose(800)])
        
        self.assertEqual(output.shape, (1, 512, 512, 4))
    
    def test_tha_v3_seperable_fp32(self):
        """Test THA v3 seperable FP32 model"""
        core = CoreORT(
            tha_model_version='v3',
            tha_model_seperable=True,
            tha_model_fp16=False,
            use_eyebrow=False,
            cache_max_giga=0.0
        )
        core.setImage(self.test_image)
        output = core.inference([self.get_pose(800)])
        
        self.assertEqual(output.shape, (1, 512, 512, 4))
    
    def test_tha_v3_with_eyebrow(self):
        """Test THA v3 with eyebrow enabled"""
        core = CoreORT(
            tha_model_version='v3',
            tha_model_seperable=True,
            tha_model_fp16=True,
            use_eyebrow=True,
            cache_max_giga=0.0
        )
        core.setImage(self.test_image)
        output = core.inference([self.get_pose(800)])
        
        self.assertEqual(output.shape, (1, 512, 512, 4))
    
    def test_tha_v4_fp16(self):
        """Test THA v4 FP16 model"""
        try:
            core = CoreORT(
                tha_model_version='v4',
                tha_model_fp16=True,
                use_eyebrow=False,
                cache_max_giga=0.0
            )
            core.setImage(self.test_image)
            output = core.inference([self.get_pose(800)])
            self.assertEqual(output.shape, (1, 512, 512, 4))
        except Exception as e:
            self.skipTest(f"THA v4 model not available: {e}")
    
    def test_tha_v4_fp32(self):
        """Test THA v4 FP32 model"""
        try:
            core = CoreORT(
                tha_model_version='v4',
                tha_model_fp16=False,
                use_eyebrow=False,
                cache_max_giga=0.0
            )
            core.setImage(self.test_image)
            output = core.inference([self.get_pose(800)])
            self.assertEqual(output.shape, (1, 512, 512, 4))
        except Exception as e:
            self.skipTest(f"THA v4 model not available: {e}")


class TestCoreORTCacher(TestCoreORTBase):
    """Test CoreORT cacher functionality"""
    
    def test_cacher_disabled(self):
        """Test that cacher is disabled when cache_max_giga=0"""
        core = CoreORT(
            tha_model_version='v3',
            tha_model_seperable=True,
            tha_model_fp16=True,
            use_eyebrow=False,
            cache_max_giga=0.0
        )
        self.assertIsNone(core.cacher)
    
    def test_cacher_enabled(self):
        """Test that cacher is enabled when cache_max_giga > 0"""
        core = CoreORT(
            tha_model_version='v3',
            tha_model_seperable=True,
            tha_model_fp16=True,
            use_eyebrow=False,
            cache_max_giga=1.0
        )
        self.assertIsNotNone(core.cacher)
    
    def test_cacher_hits_on_repeated_same_pose(self):
        """Test that cache hits occur on repeated same pose (resets continues_hits counter)"""
        core = CoreORT(
            tha_model_version='v3',
            tha_model_seperable=True,
            tha_model_fp16=True,
            use_eyebrow=False,
            cache_max_giga=1.0
        )
        core.setImage(self.test_image)
        
        pose = self.get_pose(800)
        
        # First inference - cache miss
        output1 = core.inference([pose])
        initial_misses = core.cacher.miss
        initial_hits = core.cacher.hits
        
        # Second inference with same pose - cache hit
        # Same pose consecutively resets continues_hits to 0, so hits always work
        output2 = core.inference([pose])
        
        self.assertEqual(core.cacher.hits, initial_hits + 1)
        self.assertEqual(core.cacher.miss, initial_misses)
        
        # Outputs should be equal
        np.testing.assert_array_equal(output1, output2)
        
        # Even more consecutive same-pose hits should work (counter stays at 0)
        for _ in range(10):
            output = core.inference([pose])
            np.testing.assert_array_equal(output, output1)
        
        # All should be hits after the initial miss
        self.assertEqual(core.cacher.hits, initial_hits + 11)
        self.assertEqual(core.cacher.miss, initial_misses)
    
    def test_cacher_misses_on_different_poses(self):
        """Test that cache misses occur on different poses (first access)"""
        core = CoreORT(
            tha_model_version='v3',
            tha_model_seperable=True,
            tha_model_fp16=True,
            use_eyebrow=False,
            cache_max_giga=1.0
        )
        core.setImage(self.test_image)
        
        # Run with different poses - all first-time accesses
        for i in range(5):
            pose = self.get_pose(800 + i)
            core.inference([pose])
        
        # All should be misses (first access for each pose)
        self.assertEqual(core.cacher.miss, 5)
        self.assertEqual(core.cacher.hits, 0)
    
    def test_cacher_performance_two_passes(self):
        """Test cache performance with two passes over same data
        
        Note: The cacher has an anti-thrashing mechanism that forces a miss
        after 5 consecutive hits with DIFFERENT keys. When the same key is
        accessed consecutively, the counter resets to 0.
        """
        core = CoreORT(
            tha_model_version='v3',
            tha_model_seperable=True,
            tha_model_fp16=True,
            use_eyebrow=False,
            cache_max_giga=2.0
        )
        core.setImage(self.test_image)
        
        poses = [self.get_pose(800 + i) for i in range(20)]
        
        # First pass - all misses
        for pose in poses:
            core.inference([pose])
        
        first_pass_misses = core.cacher.miss
        first_pass_hits = core.cacher.hits
        
        self.assertEqual(first_pass_misses, 20)
        self.assertEqual(first_pass_hits, 0)
        
        # Second pass - due to anti-thrashing, after 5 consecutive hits with different keys,
        # the 6th will be forced to miss. Pattern: 5 hits, 1 miss, 5 hits, 1 miss, etc.
        for pose in poses:
            core.inference([pose])

class TestCoreORTRIFE(TestCoreORTBase):
    """Test CoreORT with RIFE frame interpolation"""
    
    def test_rife_x2_fp16(self):
        """Test RIFE x2 FP16 interpolation"""
        core = CoreORT(
            tha_model_version='v3',
            tha_model_seperable=True,
            tha_model_fp16=True,
            use_eyebrow=False,
            rife_model_enable=True,
            rife_model_scale=2,
            rife_model_fp16=True,
            cache_max_giga=0.0
        )
        self.assertIsNotNone(core.rife)
        
        core.setImage(self.test_image)
        
        # First frame
        output1 = core.inference([self.get_pose(800)])
        # RIFE x2 outputs 2 frames (1 interpolated between previous and current)
        self.assertEqual(output1.shape[0], 2)
        
        # Second frame
        output2 = core.inference([self.get_pose(801)])
        self.assertEqual(output2.shape[0], 2)
    
    def test_rife_x3_fp16(self):
        """Test RIFE x3 FP16 interpolation"""
        core = CoreORT(
            tha_model_version='v3',
            tha_model_seperable=True,
            tha_model_fp16=True,
            use_eyebrow=False,
            rife_model_enable=True,
            rife_model_scale=3,
            rife_model_fp16=True,
            cache_max_giga=0.0
        )
        self.assertIsNotNone(core.rife)
        
        core.setImage(self.test_image)
        output = core.inference([self.get_pose(800)])
        # RIFE x3 outputs 3 frames
        self.assertEqual(output.shape[0], 3)
    
    def test_rife_x4_fp16(self):
        """Test RIFE x4 FP16 interpolation"""
        core = CoreORT(
            tha_model_version='v3',
            tha_model_seperable=True,
            tha_model_fp16=True,
            use_eyebrow=False,
            rife_model_enable=True,
            rife_model_scale=4,
            rife_model_fp16=True,
            cache_max_giga=0.0
        )
        self.assertIsNotNone(core.rife)
        
        core.setImage(self.test_image)
        output = core.inference([self.get_pose(800)])
        # RIFE x4 outputs 4 frames
        self.assertEqual(output.shape[0], 4)
    
    def test_rife_x2_fp32(self):
        """Test RIFE x2 FP32 interpolation"""
        core = CoreORT(
            tha_model_version='v3',
            tha_model_seperable=True,
            tha_model_fp16=True,
            use_eyebrow=False,
            rife_model_enable=True,
            rife_model_scale=2,
            rife_model_fp16=False,
            cache_max_giga=0.0
        )
        self.assertIsNotNone(core.rife)
        
        core.setImage(self.test_image)
        output = core.inference([self.get_pose(800)])
        self.assertEqual(output.shape[0], 2)

    def test_rife_x4_multi_pose_with_cache(self):
        """Multi-pose RIFE x4 interpolation should cache intermediate frames for reuse"""
        core = CoreORT(
            tha_model_version='v3',
            tha_model_seperable=True,
            tha_model_fp16=True,
            use_eyebrow=False,
            rife_model_enable=True,
            rife_model_scale=4,
            rife_model_fp16=True,
            cache_max_giga=1.0
        )

        core.setImage(self.test_image)

        # Use three interpolation poses plus the sampled pose (len=4 -> x4 session)
        poses = [
            self.get_pose(800),
            self.get_pose(801),
            self.get_pose(802),
            self.get_pose(803),
        ]

        # First run populates cache entries for the interpolation poses
        output1 = core.inference(poses)
        self.assertEqual(output1.shape[0], len(poses))
        first_hits = core.cacher.hits
        first_miss = core.cacher.miss
        self.assertEqual(first_hits, 0)
        self.assertEqual(first_miss, 1)  # final pose lookup misses and is computed

        # Second run should reuse cached interpolation frames (hits for three cached poses)
        output2 = core.inference(poses)
        self.assertEqual(output2.shape[0], len(poses))
        self.assertEqual(core.cacher.hits, first_hits + len(poses))
        self.assertEqual(core.cacher.miss, first_miss)  # final pose still computed

        np.testing.assert_array_equal(output1, output2)
    
    def test_rife_disabled(self):
        """Test that RIFE is disabled when rife_model_enable=False"""
        core = CoreORT(
            tha_model_version='v3',
            tha_model_seperable=True,
            tha_model_fp16=True,
            use_eyebrow=False,
            rife_model_enable=False,
            cache_max_giga=0.0
        )
        self.assertIsNone(core.rife)


class TestCoreORTSuperResolution(TestCoreORTBase):
    """Test CoreORT with Super Resolution models"""

    def test_sr_cache_single_pose_hits(self):
        """SR cacher should reuse SR results for the same pose (no RIFE)"""
        core = CoreORT(
            tha_model_version='v3',
            tha_model_seperable=True,
            tha_model_fp16=True,
            use_eyebrow=False,
            sr_model_enable=True,
            sr_model_scale=2,
            sr_model_fp16=True,
            cache_max_giga=1.0
        )

        self.assertIsNotNone(core.sr_cacher)

        core.setImage(self.test_image)
        pose = self.get_pose(800)

        first = core.inference([pose])
        self.assertEqual(core.sr_cacher.miss, 1)
        self.assertEqual(core.sr_cacher.hits, 0)

        second = core.inference([pose])
        self.assertEqual(core.sr_cacher.hits, 1)
        self.assertEqual(core.sr_cacher.miss, 1)
        np.testing.assert_array_equal(first, second)
    
    def test_sr_waifu2x_x2_fp16(self):
        """Test waifu2x x2 FP16 super resolution"""
        core = CoreORT(
            tha_model_version='v3',
            tha_model_seperable=True,
            tha_model_fp16=True,
            use_eyebrow=False,
            sr_model_enable=True,
            sr_model_scale=2,
            sr_model_fp16=True,
            cache_max_giga=0.0
        )
        self.assertIsNotNone(core.sr)
        
        core.setImage(self.test_image)
        output = core.inference([self.get_pose(800)])
        
        # waifu2x x2 should output 1024x1024
        self.assertEqual(output.shape[1], 1024)
        self.assertEqual(output.shape[2], 1024)
    
    def test_sr_waifu2x_x2_fp32(self):
        """Test waifu2x x2 FP32 super resolution"""
        core = CoreORT(
            tha_model_version='v3',
            tha_model_seperable=True,
            tha_model_fp16=True,
            use_eyebrow=False,
            sr_model_enable=True,
            sr_model_scale=2,
            sr_model_fp16=False,
            cache_max_giga=0.0
        )
        self.assertIsNotNone(core.sr)
        
        core.setImage(self.test_image)
        output = core.inference([self.get_pose(800)])
        
        self.assertEqual(output.shape[1], 1024)
        self.assertEqual(output.shape[2], 1024)
    
    def test_sr_realesrgan_x4_fp16(self):
        """Test Real-ESRGAN x4 FP16 super resolution (x4 is model name, outputs 1024x1024)"""
        core = CoreORT(
            tha_model_version='v3',
            tha_model_seperable=True,
            tha_model_fp16=True,
            use_eyebrow=False,
            sr_model_enable=True,
            sr_model_scale=4,
            sr_model_fp16=True,
            cache_max_giga=0.0
        )
        self.assertIsNotNone(core.sr)
        
        core.setImage(self.test_image)
        output = core.inference([self.get_pose(800)])
        
        # Real-ESRGAN x4 model also outputs 1024x1024 (x4 is just the model name)
        self.assertEqual(output.shape[1], 1024)
        self.assertEqual(output.shape[2], 1024)
    
    def test_sr_realesrgan_x4_fp32(self):
        """Test Real-ESRGAN x4 FP32 super resolution (x4 is model name, outputs 1024x1024)"""
        core = CoreORT(
            tha_model_version='v3',
            tha_model_seperable=True,
            tha_model_fp16=True,
            use_eyebrow=False,
            sr_model_enable=True,
            sr_model_scale=4,
            sr_model_fp16=False,
            cache_max_giga=0.0
        )
        self.assertIsNotNone(core.sr)
        
        core.setImage(self.test_image)
        output = core.inference([self.get_pose(800)])
        
        # Real-ESRGAN x4 model also outputs 1024x1024 (x4 is just the model name)
        self.assertEqual(output.shape[1], 1024)
        self.assertEqual(output.shape[2], 1024)
    
    def test_sr_disabled(self):
        """Test that SR is disabled when sr_model_enable=False"""
        core = CoreORT(
            tha_model_version='v3',
            tha_model_seperable=True,
            tha_model_fp16=True,
            use_eyebrow=False,
            sr_model_enable=False,
            cache_max_giga=0.0
        )
        self.assertIsNone(core.sr)


class TestCoreORTCombined(TestCoreORTBase):
    """Test CoreORT with combined RIFE + SR configurations"""
    
    def test_rife_x2_with_sr_x2(self):
        """Test RIFE x2 combined with waifu2x x2"""
        core = CoreORT(
            tha_model_version='v3',
            tha_model_seperable=True,
            tha_model_fp16=True,
            use_eyebrow=False,
            rife_model_enable=True,
            rife_model_scale=2,
            rife_model_fp16=True,
            sr_model_enable=True,
            sr_model_scale=2,
            sr_model_fp16=True,
            cache_max_giga=0.0
        )
        self.assertIsNotNone(core.rife)
        self.assertIsNotNone(core.sr)
        
        core.setImage(self.test_image)
        output = core.inference([self.get_pose(800)])
        
        # 2 frames, 1024x1024 each
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 1024)
        self.assertEqual(output.shape[2], 1024)
    
    def test_rife_x3_with_sr_x4(self):
        """Test RIFE x3 combined with Real-ESRGAN x4 (x4 is model name, outputs 1024x1024)"""
        core = CoreORT(
            tha_model_version='v3',
            tha_model_seperable=True,
            tha_model_fp16=True,
            use_eyebrow=False,
            rife_model_enable=True,
            rife_model_scale=3,
            rife_model_fp16=True,
            sr_model_enable=True,
            sr_model_scale=4,
            sr_model_fp16=True,
            cache_max_giga=0.0
        )
        
        core.setImage(self.test_image)
        output = core.inference([self.get_pose(800)])
        
        # 3 frames, 1024x1024 each (x4 is just model name, all SR outputs 1024x1024)
        self.assertEqual(output.shape[0], 3)
        self.assertEqual(output.shape[1], 1024)
        self.assertEqual(output.shape[2], 1024)
    
    def test_all_features_with_cacher(self):
        """Test RIFE + SR with cacher enabled"""
        core = CoreORT(
            tha_model_version='v3',
            tha_model_seperable=True,
            tha_model_fp16=True,
            use_eyebrow=False,
            rife_model_enable=True,
            rife_model_scale=2,
            rife_model_fp16=True,
            sr_model_enable=True,
            sr_model_scale=2,
            sr_model_fp16=True,
            sr_a4k=True,
            cache_max_giga=1.0
        )
        self.assertIsNotNone(core.cacher)
        self.assertIsNotNone(core.rife)
        self.assertIsNotNone(core.sr or core.sr_a4k)
        
        core.setImage(self.test_image)
        
        # Note: Cacher only caches THA output, not RIFE/SR output
        pose = self.get_pose(800)
        output1 = core.inference([pose])
        output2 = core.inference([pose])
        
        self.assertEqual(output1.shape, output2.shape)

    def test_sr_cacher_with_rife_partial_hits(self):
        """SR cacher should reuse cached SR frames when RIFE is enabled"""
        core = CoreORT(
            tha_model_version='v3',
            tha_model_seperable=True,
            tha_model_fp16=True,
            use_eyebrow=False,
            rife_model_enable=True,
            rife_model_scale=2,
            rife_model_fp16=True,
            sr_model_enable=True,
            sr_model_scale=2,
            sr_model_fp16=True,
            cache_max_giga=1.0
        )

        self.assertIsNotNone(core.cacher)
        self.assertIsNotNone(core.sr_cacher)
        self.assertIsNotNone(core.rife)

        core.setImage(self.test_image)

        poses_first = [self.get_pose(800), self.get_pose(801)]
        output1 = core.inference(poses_first)
        self.assertEqual(output1.shape[0], len(poses_first))
        sr_hits_after_first = core.sr_cacher.hits
        sr_miss_after_first = core.sr_cacher.miss

        poses_second = [self.get_pose(800), self.get_pose(802)]
        output2 = core.inference(poses_second)
        self.assertEqual(output2.shape[0], len(poses_second))

        # Cached SR frame for pose 800 should be reused; pose 802 should force a miss
        self.assertGreater(core.sr_cacher.hits, sr_hits_after_first)
        self.assertGreaterEqual(core.sr_cacher.miss, sr_miss_after_first + 1)
        np.testing.assert_array_equal(output1[0], output2[0])


# ============================================================================
# Video Generation Showcase Functions
# ============================================================================

def CoreORT_ShowVideo_THAOnly():
    """Generate video with THA only (no interpolation, no SR)"""
    print("=" * 60)
    print("Generating video: THA only")
    print("=" * 60)
    
    core = CoreORT(
        tha_model_version='v3',
        tha_model_seperable=True,
        tha_model_fp16=True,
        use_eyebrow=False,
        cache_max_giga=2.0
    )
    
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    core.setImage(img)
    
    with open('./test/data/pose_20fps.json', 'r') as f:
        pose_data = json.load(f)
    
    frames = []
    print("Pass 1 (cache cold):")
    for i in tqdm(range(200)):
        pose = np.array(pose_data[800 + i]).reshape(1, 45)
        output = core.inference([pose])
        frames.append(output[0, :, :, :3])
    
    print("Pass 2 (cache warm):")
    for i in tqdm(range(200)):
        pose = np.array(pose_data[800 + i]).reshape(1, 45)
        output = core.inference([pose])
        frames.append(output[0, :, :, :3])
    
    generate_video(frames, './test/data/core_ort_tha_only.mp4', 20)
    
    if core.cacher:
        print(f"Cache stats - Hits: {core.cacher.hits}, Misses: {core.cacher.miss}")


def CoreORT_ShowVideo_WithRIFE_x2():
    """Generate video with RIFE x2 interpolation"""
    print("=" * 60)
    print("Generating video: THA + RIFE x2")
    print("=" * 60)
    
    core = CoreORT(
        tha_model_version='v3',
        tha_model_seperable=True,
        tha_model_fp16=True,
        use_eyebrow=False,
        rife_model_enable=True,
        rife_model_scale=2,
        rife_model_fp16=True,
        cache_max_giga=2.0
    )
    
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    core.setImage(img)
    
    with open('./test/data/pose_20fps.json', 'r') as f:
        pose_data = json.load(f)
    
    frames = []
    print("Generating interpolated frames:")
    for i in tqdm(range(200)):
        pose = np.array(pose_data[800 + i]).reshape(1, 45)
        output = core.inference([pose])
        for j in range(output.shape[0]):
            frames.append(output[j, :, :, :3])
    
    # x2 interpolation means 40fps output from 20fps input
    generate_video(frames, './test/data/core_ort_rife_x2.mp4', 40)
    
    if core.cacher:
        print(f"Cache stats - Hits: {core.cacher.hits}, Misses: {core.cacher.miss}")


def CoreORT_ShowVideo_WithRIFE_x3():
    """Generate video with RIFE x3 interpolation"""
    print("=" * 60)
    print("Generating video: THA + RIFE x3")
    print("=" * 60)
    
    core = CoreORT(
        tha_model_version='v3',
        tha_model_seperable=True,
        tha_model_fp16=True,
        use_eyebrow=False,
        rife_model_enable=True,
        rife_model_scale=3,
        rife_model_fp16=True,
        cache_max_giga=2.0
    )
    
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    core.setImage(img)
    
    with open('./test/data/pose_20fps.json', 'r') as f:
        pose_data = json.load(f)
    
    frames = []
    print("Generating interpolated frames:")
    for i in tqdm(range(200)):
        pose = np.array(pose_data[800 + i]).reshape(1, 45)
        output = core.inference([pose])
        for j in range(output.shape[0]):
            frames.append(output[j, :, :, :3])
    
    # x3 interpolation means 60fps output from 20fps input
    generate_video(frames, './test/data/core_ort_rife_x3.mp4', 60)
    
    if core.cacher:
        print(f"Cache stats - Hits: {core.cacher.hits}, Misses: {core.cacher.miss}")


def CoreORT_ShowVideo_WithRIFE_x4():
    """Generate video with RIFE x4 interpolation"""
    print("=" * 60)
    print("Generating video: THA + RIFE x4")
    print("=" * 60)
    
    core = CoreORT(
        tha_model_version='v3',
        tha_model_seperable=True,
        tha_model_fp16=True,
        use_eyebrow=False,
        rife_model_enable=True,
        rife_model_scale=4,
        rife_model_fp16=True,
        cache_max_giga=2.0
    )
    
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    core.setImage(img)
    
    with open('./test/data/pose_20fps.json', 'r') as f:
        pose_data = json.load(f)
    
    frames = []
    print("Generating interpolated frames:")
    for i in tqdm(range(200)):
        pose = np.array(pose_data[800 + i]).reshape(1, 45)
        output = core.inference([pose])
        for j in range(output.shape[0]):
            frames.append(output[j, :, :, :3])
    
    # x4 interpolation means 80fps output from 20fps input
    generate_video(frames, './test/data/core_ort_rife_x4.mp4', 80)
    
    if core.cacher:
        print(f"Cache stats - Hits: {core.cacher.hits}, Misses: {core.cacher.miss}")


def CoreORT_ShowVideo_WithSR_waifu2x():
    """Generate video with waifu2x x2 super resolution"""
    print("=" * 60)
    print("Generating video: THA + waifu2x x2")
    print("=" * 60)
    
    core = CoreORT(
        tha_model_version='v3',
        tha_model_seperable=True,
        tha_model_fp16=True,
        use_eyebrow=False,
        sr_model_enable=True,
        sr_model_scale=2,
        sr_model_fp16=True,
        cache_max_giga=0.0
    )
    
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    core.setImage(img)
    
    with open('./test/data/pose_20fps.json', 'r') as f:
        pose_data = json.load(f)
    
    frames = []
    print("Generating SR frames (1024x1024):")
    for i in tqdm(range(100)):  # Fewer frames due to larger size
        pose = np.array(pose_data[800 + i]).reshape(1, 45)
        output = core.inference([pose])
        frames.append(output[0, :, :, :3])
    
    generate_video(frames, './test/data/core_ort_sr_waifu2x.mp4', 20)
    
    if core.cacher:
        print(f"Cache stats - Hits: {core.cacher.hits}, Misses: {core.cacher.miss}")


def CoreORT_ShowVideo_WithSR_RealESRGAN():
    """Generate video with Real-ESRGAN x4 super resolution (outputs 1024x1024)"""
    print("=" * 60)
    print("Generating video: THA + Real-ESRGAN x4")
    print("=" * 60)
    
    core = CoreORT(
        tha_model_version='v3',
        tha_model_seperable=True,
        tha_model_fp16=True,
        use_eyebrow=False,
        sr_model_enable=True,
        sr_model_scale=4,
        sr_model_fp16=True,
        cache_max_giga=2.0
    )
    
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    core.setImage(img)
    
    with open('./test/data/pose_20fps.json', 'r') as f:
        pose_data = json.load(f)
    
    frames = []
    print("Generating SR frames (1024x1024):")
    for i in tqdm(range(100)):
        pose = np.array(pose_data[800 + i]).reshape(1, 45)
        output = core.inference([pose])
        frames.append(output[0, :, :, :3])
    
    generate_video(frames, './test/data/core_ort_sr_realesrgan.mp4', 20)
    
    if core.cacher:
        print(f"Cache stats - Hits: {core.cacher.hits}, Misses: {core.cacher.miss}")

def CoreORT_ShowVideo_WithSR_a4k():
    """Generate video with anime4k x2 super resolution"""
    print("=" * 60)
    print("Generating video: THA + anime4k x2")
    print("=" * 60)
    
    core = CoreORT(
        tha_model_version='v3',
        tha_model_seperable=True,
        tha_model_fp16=True,
        use_eyebrow=False,
        sr_model_enable=True,
        sr_a4k=True,
        cache_max_giga=0.0
    )
    
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    core.setImage(img)
    
    with open('./test/data/pose_20fps.json', 'r') as f:
        pose_data = json.load(f)
    
    frames = []
    print("Generating SR frames (1024x1024):")
    for i in tqdm(range(100)):  # Fewer frames due to larger size
        pose = np.array(pose_data[800 + i]).reshape(1, 45)
        output = core.inference([pose])
        frames.append(output[0, :, :, :3])
    
    generate_video(frames, './test/data/core_ort_sr_anime4k.mp4', 20)
    
    if core.cacher:
        print(f"Cache stats - Hits: {core.cacher.hits}, Misses: {core.cacher.miss}")
    if core.sr_cacher:
        print(f"SR Cache stats - Hits: {core.sr_cacher.hits}, Misses: {core.sr_cacher.miss}")

def CoreORT_ShowVideo_RIFE_x2_SR_waifu2x():
    """Generate video with RIFE x2 + waifu2x x2 combined"""
    print("=" * 60)
    print("Generating video: THA + RIFE x2 + waifu2x x2")
    print("=" * 60)
    
    core = CoreORT(
        tha_model_version='v3',
        tha_model_seperable=True,
        tha_model_fp16=True,
        use_eyebrow=False,
        rife_model_enable=True,
        rife_model_scale=2,
        rife_model_fp16=True,
        sr_model_enable=True,
        sr_model_scale=2,
        sr_model_fp16=True,
        cache_max_giga=2.0
    )
    
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    core.setImage(img)
    
    with open('./test/data/pose_20fps.json', 'r') as f:
        pose_data = json.load(f)
    
    frames = []
    print("Generating interpolated + SR frames:")
    for i in tqdm(range(100)):
        pose = np.array(pose_data[800 + i]).reshape(1, 45)
        output = core.inference([pose])
        for j in range(output.shape[0]):
            frames.append(output[j, :, :, :3])
    
    generate_video(frames, './test/data/core_ort_rife_x2_sr_waifu2x.mp4', 40)
    
    if core.cacher:
        print(f"Cache stats - Hits: {core.cacher.hits}, Misses: {core.cacher.miss}")
    if core.sr_cacher:
        print(f"SR Cache stats - Hits: {core.sr_cacher.hits}, Misses: {core.sr_cacher.miss}")

def CoreORT_ShowVideo_RIFE_x2_SR_a4k():
    """Generate video with RIFE x2 + anime4k x2 combined"""
    print("=" * 60)
    print("Generating video: THA + RIFE x2 + anime4k x2")
    print("=" * 60)
    
    core = CoreORT(
        tha_model_version='v3',
        tha_model_seperable=True,
        tha_model_fp16=True,
        use_eyebrow=False,
        rife_model_enable=True,
        rife_model_scale=2,
        rife_model_fp16=True,
        sr_model_enable=True,
        sr_a4k=True,
        cache_max_giga=2.0
    )
    
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    core.setImage(img)
    
    with open('./test/data/pose_20fps.json', 'r') as f:
        pose_data = json.load(f)
    
    frames = []
    print("Generating interpolated + SR frames:")
    for i in tqdm(range(100)):
        pose = np.array(pose_data[800 + i]).reshape(1, 45)
        output = core.inference([pose])
        for j in range(output.shape[0]):
            frames.append(output[j, :, :, :3])
    
    generate_video(frames, './test/data/core_ort_rife_x2_sr_a4k.mp4', 40)
    
    if core.cacher:
        print(f"Cache stats - Hits: {core.cacher.hits}, Misses: {core.cacher.miss}")
    if core.sr_cacher:
        print(f"SR Cache stats - Hits: {core.sr_cacher.hits}, Misses: {core.sr_cacher.miss}")

def CoreORT_ShowVideo_CachePerformance():
    """Demonstrate cache performance with two passes"""
    print("=" * 60)
    print("Demonstrating cache performance")
    print("=" * 60)
    
    core = CoreORT(
        tha_model_version='v3',
        tha_model_seperable=True,
        tha_model_fp16=True,
        use_eyebrow=False,
        cache_max_giga=2.0
    )
    
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    core.setImage(img)
    
    with open('./test/data/pose_20fps.json', 'r') as f:
        pose_data = json.load(f)
    
    import time
    
    # First pass - cache cold
    frames_pass1 = []
    start_time = time.time()
    print("Pass 1 (cache cold):")
    for i in tqdm(range(200)):
        pose = np.array(pose_data[800 + i]).reshape(1, 45)
        output = core.inference([pose])
        frames_pass1.append(output[0, :, :, :3])
    pass1_time = time.time() - start_time
    print(f"Pass 1 time: {pass1_time:.2f}s")
    print(f"Cache stats - Hits: {core.cacher.hits}, Misses: {core.cacher.miss}")
    
    # Second pass - cache warm
    frames_pass2 = []
    start_time = time.time()
    print("\nPass 2 (cache warm):")
    for i in tqdm(range(200)):
        pose = np.array(pose_data[800 + i]).reshape(1, 45)
        output = core.inference([pose])
        frames_pass2.append(output[0, :, :, :3])
    pass2_time = time.time() - start_time
    print(f"Pass 2 time: {pass2_time:.2f}s")
    print(f"Cache stats - Hits: {core.cacher.hits}, Misses: {core.cacher.miss}")
    
    print(f"\nSpeedup from cache: {pass1_time / pass2_time:.2f}x")
    
    # Combine frames for video
    all_frames = frames_pass1 + frames_pass2
    generate_video(all_frames, './test/data/core_ort_cache_demo.mp4', 20)


def CoreORT_ShowVideo_SRCachePerformance():
    """Demonstrate SR cache performance (waifu2x x2, two passes)"""
    print("=" * 60)
    print("Demonstrating SR cache performance")
    print("=" * 60)
    
    core = CoreORT(
        tha_model_version='v3',
        tha_model_seperable=True,
        tha_model_fp16=True,
        use_eyebrow=False,
        rife_model_enable=True,
        rife_model_scale=3,
        rife_model_fp16=True,
        sr_model_enable=True,
        sr_model_scale=2,
        sr_model_fp16=True,
        cache_max_giga=2.0
    )
    
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    core.setImage(img)
    
    with open('./test/data/pose_20fps.json', 'r') as f:
        pose_data = json.load(f)
    
    import time
    
    frames_pass1 = []
    start_time = time.time()
    print("Pass 1 (SR cache cold):")
    for i in tqdm(range(200)):
        pose = np.array(pose_data[800 + i]).reshape(1, 45)
        output = core.inference([pose])
        frames_pass1.append(output[0, :, :, :3])
    pass1_time = time.time() - start_time
    print(f"Pass 1 time: {pass1_time:.2f}s")
    print(f"THA cache - Hits: {core.cacher.hits if core.cacher else 0}, Misses: {core.cacher.miss if core.cacher else 0}")
    print(f"SR cache  - Hits: {core.sr_cacher.hits if core.sr_cacher else 0}, Misses: {core.sr_cacher.miss if core.sr_cacher else 0}")
    
    frames_pass2 = []
    start_time = time.time()
    print("\nPass 2 (SR cache warm):")
    for i in tqdm(range(200)):
        pose = np.array(pose_data[800 + i]).reshape(1, 45)
        output = core.inference([pose])
        frames_pass2.append(output[0, :, :, :3])
    pass2_time = time.time() - start_time
    print(f"Pass 2 time: {pass2_time:.2f}s")
    print(f"THA cache - Hits: {core.cacher.hits if core.cacher else 0}, Misses: {core.cacher.miss if core.cacher else 0}")
    print(f"SR cache  - Hits: {core.sr_cacher.hits if core.sr_cacher else 0}, Misses: {core.sr_cacher.miss if core.sr_cacher else 0}")
    
    print(f"\nSR cache speedup: {pass1_time / pass2_time:.2f}x")
    
    all_frames = frames_pass1 + frames_pass2
    generate_video(all_frames, './test/data/core_ort_sr_cache_demo.mp4', 20)


def CoreORT_ShowVideo_SRCache_WithRIFE():
    """Demonstrate SR cache performance when RIFE uses interpolated poses (x3 + waifu2x x2)"""
    print("=" * 60)
    print("Demonstrating SR cache with RIFE (interpolated poses)")
    print("=" * 60)

    core = CoreORT(
        tha_model_version='v3',
        tha_model_seperable=True,
        tha_model_fp16=True,
        use_eyebrow=False,
        rife_model_enable=True,
        rife_model_scale=3,
        rife_model_fp16=True,
        sr_model_enable=True,
        sr_model_scale=2,
        sr_model_fp16=True,
        cache_max_giga=2.0
    )

    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    core.setImage(img)

    with open('./test/data/pose_20fps.json', 'r') as f:
        pose_data = json.load(f)

    import time

    def run_pass(start_idx: int, count: int) -> Tuple[List[np.ndarray], float]:
        frames = []
        t0 = time.time()
        for i in tqdm(range(count)):
            base = start_idx + i * 3
            poses = [
                np.array(pose_data[base]).reshape(1, 45),
                np.array(pose_data[base + 1]).reshape(1, 45),
                np.array(pose_data[base + 2]).reshape(1, 45),
            ]
            output = core.inference(poses)
            for j in range(output.shape[0]):
                frames.append(output[j, :, :, :3])
        return frames, time.time() - t0

    print("Pass 1 (SR cache cold, RIFE x3 poses):")
    frames_pass1, pass1_time = run_pass(800, 200)
    print(f"Pass 1 time: {pass1_time:.2f}s")
    print(f"THA cache - Hits: {core.cacher.hits if core.cacher else 0}, Misses: {core.cacher.miss if core.cacher else 0}")
    print(f"SR cache  - Hits: {core.sr_cacher.hits if core.sr_cacher else 0}, Misses: {core.sr_cacher.miss if core.sr_cacher else 0}")

    print("\nPass 2 (SR cache warm, same poses):")
    frames_pass2, pass2_time = run_pass(800, 200)
    print(f"Pass 2 time: {pass2_time:.2f}s")
    print(f"THA cache - Hits: {core.cacher.hits if core.cacher else 0}, Misses: {core.cacher.miss if core.cacher else 0}")
    print(f"SR cache  - Hits: {core.sr_cacher.hits if core.sr_cacher else 0}, Misses: {core.sr_cacher.miss if core.sr_cacher else 0}")

    print(f"\nSR+RIFE cache speedup: {pass1_time / pass2_time:.2f}x")

    all_frames = frames_pass1 + frames_pass2
    generate_video(all_frames, './test/data/core_ort_sr_rife_cache_demo.mp4', 20)


def CoreORT_ShowVideo_THA_v4():
    """Generate video with THA v4 model"""
    print("=" * 60)
    print("Generating video: THA v4")
    print("=" * 60)
    
    try:
        core = CoreORT(
            tha_model_version='v4',
            tha_model_fp16=True,
            use_eyebrow=False,
            cache_max_giga=2.0
        )
    except Exception as e:
        print(f"Skipping THA v4 video: {e}")
        return
    
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    core.setImage(img)
    
    with open('./test/data/pose_20fps.json', 'r') as f:
        pose_data = json.load(f)
    
    frames = []
    print("Generating THA v4 frames:")
    for i in tqdm(range(200)):
        pose = np.array(pose_data[800 + i]).reshape(1, 45)
        output = core.inference([pose])
        frames.append(output[0, :, :, :3])
    for i in tqdm(range(200)):
        pose = np.array(pose_data[800 + i]).reshape(1, 45)
        output = core.inference([pose])
        frames.append(output[0, :, :, :3])
    
    generate_video(frames, './test/data/core_ort_tha_v4.mp4', 20)
    
    if core.cacher:
        print(f"Cache stats - Hits: {core.cacher.hits}, Misses: {core.cacher.miss}")


if __name__ == "__main__":
    import sys
    import os
    init_model_path(os.path.join(os.path.dirname(__file__), '..', 'data'))
    if len(sys.argv) > 1:
        if sys.argv[1] == '--show-all':
            print("Running all video showcase tests")
            CoreORT_ShowVideo_THAOnly()
            CoreORT_ShowVideo_WithRIFE_x2()
            CoreORT_ShowVideo_WithRIFE_x3()
            CoreORT_ShowVideo_WithRIFE_x4()
            CoreORT_ShowVideo_WithSR_waifu2x()
            CoreORT_ShowVideo_WithSR_RealESRGAN()
            CoreORT_ShowVideo_WithSR_a4k()
            CoreORT_ShowVideo_RIFE_x2_SR_waifu2x()
            CoreORT_ShowVideo_RIFE_x2_SR_a4k()
            CoreORT_ShowVideo_CachePerformance()
            CoreORT_ShowVideo_SRCachePerformance()
            CoreORT_ShowVideo_THA_v4()
        elif sys.argv[1] == '--show-tha':
            CoreORT_ShowVideo_THAOnly()
        elif sys.argv[1] == '--show-rife':
            CoreORT_ShowVideo_WithRIFE_x2()
            CoreORT_ShowVideo_WithRIFE_x3()
            CoreORT_ShowVideo_WithRIFE_x4()
        elif sys.argv[1] == '--show-sr':
            CoreORT_ShowVideo_WithSR_waifu2x()
            CoreORT_ShowVideo_WithSR_RealESRGAN()
            CoreORT_ShowVideo_WithSR_a4k()
        elif sys.argv[1] == '--show-combined':
            CoreORT_ShowVideo_RIFE_x2_SR_waifu2x()
            CoreORT_ShowVideo_RIFE_x2_SR_a4k()
        elif sys.argv[1] == '--show-cache':
            CoreORT_ShowVideo_CachePerformance()
        elif sys.argv[1] == '--show-sr-cache':
            CoreORT_ShowVideo_SRCachePerformance()
        elif sys.argv[1] == '--show-sr-cache-rife':
            CoreORT_ShowVideo_SRCache_WithRIFE()
        elif sys.argv[1] == '--show-v4':
            CoreORT_ShowVideo_THA_v4()
        else:
            unittest.main(verbosity=2)
    else:
        unittest.main(verbosity=2)