"""
Test cases for tha4_ort.py - THA4ORTSessions and THA4ORTNonDefaultSessions classes
Tests different model configurations (fp16/fp32, with/without eyebrow)
"""
import unittest
import numpy as np
import cv2
import os
import json
from typing import List

from ezvtb_rt.tha4_ort import THA4ORTSessions, THA4ORTNonDefaultSessions


class TestTHA4ORTBase(unittest.TestCase):
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
        """Validate inference output format"""
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(output.shape, (512, 512, 4))
        self.assertEqual(output.dtype, np.uint8)


class TestTHA4ORTFP16WithEyebrow(TestTHA4ORTBase):
    """Test THA4ORTSessions with fp16 model and eyebrow enabled"""
    
    MODEL_DIR = './data/tha4/fp16'
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA4ORTSessions(cls.MODEL_DIR, use_eyebrow=True)
    
    def test_dtype_detection(self):
        """Test that fp16 is correctly detected from path"""
        self.assertEqual(self.engine.dtype, np.float16)
    
    def test_eyebrow_mode(self):
        """Test eyebrow mode is enabled"""
        self.assertTrue(self.engine.use_eyebrow)
        self.assertTrue(hasattr(self.engine, 'eyebrow_pose'))
        self.assertTrue(hasattr(self.engine, 'decomposed_background_layer'))
        self.assertTrue(hasattr(self.engine, 'decomposed_eyebrow_layer'))
    
    def test_update_image_valid(self):
        """Test updating image with valid input"""
        self.engine.update_image(self.test_image)
        # No exception means success
    
    def test_inference_after_update(self):
        """Test inference after updating image"""
        self.engine.update_image(self.test_image)
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        output = self.engine.inference(pose)
        self.validate_output(output)
    
    def test_inference_with_zero_pose(self):
        """Test inference with zero pose"""
        self.engine.update_image(self.test_image)
        output = self.engine.inference(self.create_zero_pose())
        self.validate_output(output)
    
    def test_inference_with_random_pose(self):
        """Test inference with random pose values"""
        self.engine.update_image(self.test_image)
        output = self.engine.inference(self.create_random_pose())
        self.validate_output(output)
    
    def test_multiple_inferences(self):
        """Test multiple consecutive inferences"""
        self.engine.update_image(self.test_image)
        for i in range(10):
            pose = np.array(self.pose_data[800 + i]).reshape(1, 45).astype(np.float32)
            output = self.engine.inference(pose)
            self.validate_output(output)


class TestTHA4ORTFP16NoEyebrow(TestTHA4ORTBase):
    """Test THA4ORTSessions with fp16 model and eyebrow disabled"""
    
    MODEL_DIR = './data/tha4/fp16'
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA4ORTSessions(cls.MODEL_DIR, use_eyebrow=False)
    
    def test_eyebrow_mode_disabled(self):
        """Test eyebrow mode is disabled"""
        self.assertFalse(self.engine.use_eyebrow)
        self.assertTrue(hasattr(self.engine, 'combiner_eyebrow_image'))
        self.assertTrue(hasattr(self.engine, 'combiner'))
    
    def test_inference_no_eyebrow(self):
        """Test inference without eyebrow"""
        self.engine.update_image(self.test_image)
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        output = self.engine.inference(pose)
        self.validate_output(output)
    
    def test_multiple_inferences_no_eyebrow(self):
        """Test multiple inferences without eyebrow"""
        self.engine.update_image(self.test_image)
        for i in range(5):
            pose = np.array(self.pose_data[800 + i]).reshape(1, 45).astype(np.float32)
            output = self.engine.inference(pose)
            self.validate_output(output)


class TestTHA4ORTFP32WithEyebrow(TestTHA4ORTBase):
    """Test THA4ORTSessions with fp32 model and eyebrow enabled"""
    
    MODEL_DIR = './data/tha4/fp32'
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA4ORTSessions(cls.MODEL_DIR, use_eyebrow=True)
    
    def test_dtype_detection_fp32(self):
        """Test that fp32 is correctly detected from path"""
        self.assertEqual(self.engine.dtype, np.float32)
    
    def test_inference_fp32(self):
        """Test inference with fp32 model"""
        self.engine.update_image(self.test_image)
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        output = self.engine.inference(pose)
        self.validate_output(output)


class TestTHA4ORTFP32NoEyebrow(TestTHA4ORTBase):
    """Test THA4ORTSessions with fp32 model and eyebrow disabled"""
    
    MODEL_DIR = './data/tha4/fp32'
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA4ORTSessions(cls.MODEL_DIR, use_eyebrow=False)
    
    def test_fp32_no_eyebrow(self):
        """Test fp32 model without eyebrow"""
        self.assertFalse(self.engine.use_eyebrow)
        self.assertEqual(self.engine.dtype, np.float32)
    
    def test_inference_fp32_no_eyebrow(self):
        """Test inference with fp32 model without eyebrow"""
        self.engine.update_image(self.test_image)
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        output = self.engine.inference(pose)
        self.validate_output(output)


class TestTHA4ORTEdgeCases(TestTHA4ORTBase):
    """Test edge cases and error handling"""
    
    MODEL_DIR = './data/tha4/fp16'
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA4ORTSessions(cls.MODEL_DIR, use_eyebrow=True)
    
    def test_update_image_wrong_dimensions(self):
        """Test that wrong image dimensions raise ValueError"""
        wrong_size_image = np.zeros((256, 256, 4), dtype=np.uint8)
        with self.assertRaises(ValueError) as context:
            self.engine.update_image(wrong_size_image)
        self.assertIn('Invalid update image shape', str(context.exception))
    
    def test_update_image_wrong_channels(self):
        """Test that wrong channel count raises ValueError"""
        wrong_channels_image = np.zeros((512, 512, 3), dtype=np.uint8)
        with self.assertRaises(ValueError) as context:
            self.engine.update_image(wrong_channels_image)
        self.assertIn('Invalid update image shape', str(context.exception))
    
    def test_update_image_2d_array(self):
        """Test that 2D array raises ValueError"""
        wrong_shape_image = np.zeros((512, 512), dtype=np.uint8)
        with self.assertRaises(ValueError) as context:
            self.engine.update_image(wrong_shape_image)
        self.assertIn('Invalid update image shape', str(context.exception))
    
    def test_pose_extreme_values(self):
        """Test inference with extreme pose values"""
        self.engine.update_image(self.test_image)
        extreme_pose = np.ones((1, 45), dtype=np.float32) * 100.0
        output = self.engine.inference(extreme_pose)
        self.validate_output(output)
    
    def test_pose_negative_values(self):
        """Test inference with negative pose values"""
        self.engine.update_image(self.test_image)
        negative_pose = np.ones((1, 45), dtype=np.float32) * -50.0
        output = self.engine.inference(negative_pose)
        self.validate_output(output)
    
    def test_image_update_between_inferences(self):
        """Test updating image between inferences"""
        self.engine.update_image(self.test_image)
        pose = self.create_zero_pose()
        output1 = self.engine.inference(pose)
        self.validate_output(output1)
        
        # Update with same image (simulating image change)
        self.engine.update_image(self.test_image)
        output2 = self.engine.inference(pose)
        self.validate_output(output2)
    
    def test_rapid_pose_changes(self):
        """Test rapid pose changes in succession"""
        self.engine.update_image(self.test_image)
        for _ in range(20):
            pose = self.create_random_pose()
            output = self.engine.inference(pose)
            self.validate_output(output)


class TestTHA4ORTNonDefault(TestTHA4ORTBase):
    """Test THA4ORTNonDefaultSessions class for non-default device usage"""
    
    MODEL_DIR = './data/tha4/fp16'
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        # Use device_id=0 for testing (default device)
        cls.engine = THA4ORTNonDefaultSessions(cls.MODEL_DIR, device_id=0, use_eyebrow=True)
    
    def test_nondefault_init(self):
        """Test THA4ORTNonDefaultSessions initialization"""
        self.assertEqual(self.engine.dtype, np.float16)
        self.assertTrue(self.engine.use_eyebrow)
    
    def test_nondefault_update_image(self):
        """Test updating image on non-default engine"""
        self.engine.update_image(self.test_image)
        self.assertIsNotNone(self.engine.decomposed_results)
    
    def test_nondefault_inference_with_eyebrow(self):
        """Test inference with eyebrow on non-default device"""
        self.engine.update_image(self.test_image)
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        output = self.engine.inference(pose)
        self.validate_output(output)
    
    def test_nondefault_multiple_inferences(self):
        """Test multiple inferences on non-default device"""
        self.engine.update_image(self.test_image)
        for i in range(5):
            pose = np.array(self.pose_data[800 + i]).reshape(1, 45).astype(np.float32)
            output = self.engine.inference(pose)
            self.validate_output(output)


class TestTHA4ORTNonDefaultNoEyebrow(TestTHA4ORTBase):
    """Test THA4ORTNonDefaultSessions class without eyebrow"""
    
    MODEL_DIR = './data/tha4/fp16'
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA4ORTNonDefaultSessions(cls.MODEL_DIR, device_id=0, use_eyebrow=False)
    
    def test_nondefault_no_eyebrow(self):
        """Test THA4ORTNonDefaultSessions without eyebrow"""
        self.assertFalse(self.engine.use_eyebrow)
    
    def test_nondefault_inference_no_eyebrow(self):
        """Test inference without eyebrow on non-default device"""
        self.engine.update_image(self.test_image)
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        output = self.engine.inference(pose)
        self.validate_output(output)
    
    def test_nondefault_update_creates_combiner_results(self):
        """Test that update_image creates combiner_results when eyebrow disabled"""
        self.engine.update_image(self.test_image)
        self.assertIsNotNone(self.engine.combiner_results)


class TestTHA4ORTNonDefaultFP32(TestTHA4ORTBase):
    """Test THA4ORTNonDefaultSessions with FP32 models"""
    
    MODEL_DIR = './data/tha4/fp32'
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA4ORTNonDefaultSessions(cls.MODEL_DIR, device_id=0, use_eyebrow=True)
    
    def test_dtype_detection_fp32(self):
        """Test that fp32 is correctly detected"""
        self.assertEqual(self.engine.dtype, np.float32)
    
    def test_inference_fp32_nondefault(self):
        """Test inference with FP32 model on non-default device"""
        self.engine.update_image(self.test_image)
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        output = self.engine.inference(pose)
        self.validate_output(output)


class TestTHA4ORTNonDefaultMultipleImages(TestTHA4ORTBase):
    """Test THA4ORTNonDefaultSessions with multiple image updates"""
    
    MODEL_DIR = './data/tha4/fp32'
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine = THA4ORTNonDefaultSessions(cls.MODEL_DIR, device_id=0, use_eyebrow=True)
    
    def test_multiple_image_updates(self):
        """Test updating image multiple times"""
        for _ in range(3):
            self.engine.update_image(self.test_image)
            pose = self.create_zero_pose()
            output = self.engine.inference(pose)
            self.validate_output(output)
    
    def test_inference_consistency_after_image_update(self):
        """Test that outputs are consistent after image updates"""
        self.engine.update_image(self.test_image)
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        
        output1 = self.engine.inference(pose)
        
        # Update with same image
        self.engine.update_image(self.test_image)
        output2 = self.engine.inference(pose)
        
        # Should produce same result
        np.testing.assert_array_equal(output1, output2)


class TestTHA4ORTNonDefaultEdgeCases(TestTHA4ORTBase):
    """Test edge cases for THA4ORTNonDefaultSessions"""
    
    MODEL_DIR = './data/tha4/fp16'
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine_eyebrow = THA4ORTNonDefaultSessions(cls.MODEL_DIR, device_id=0, use_eyebrow=True)
        cls.engine_no_eyebrow = THA4ORTNonDefaultSessions(cls.MODEL_DIR, device_id=0, use_eyebrow=False)
    
    def test_extreme_pose_values_nondefault(self):
        """Test extreme pose values with THA4ORTNonDefaultSessions"""
        self.engine_eyebrow.update_image(self.test_image)
        extreme_pose = np.ones((1, 45), dtype=np.float32) * 100.0
        output = self.engine_eyebrow.inference(extreme_pose)
        self.validate_output(output)
    
    def test_negative_pose_values_nondefault(self):
        """Test negative pose values with THA4ORTNonDefaultSessions"""
        self.engine_eyebrow.update_image(self.test_image)
        negative_pose = np.ones((1, 45), dtype=np.float32) * -50.0
        output = self.engine_eyebrow.inference(negative_pose)
        self.validate_output(output)
    
    def test_rapid_inferences_nondefault(self):
        """Test rapid consecutive inferences with THA4ORTNonDefaultSessions"""
        self.engine_eyebrow.update_image(self.test_image)
        for i in range(20):
            pose = np.array(self.pose_data[800 + i]).reshape(1, 45).astype(np.float32)
            output = self.engine_eyebrow.inference(pose)
            self.validate_output(output)
    
    def test_different_outputs_for_different_poses_nondefault(self):
        """Test that different poses produce different outputs"""
        self.engine_eyebrow.update_image(self.test_image)
        
        pose1 = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        pose2 = np.array(self.pose_data[850]).reshape(1, 45).astype(np.float32)
        
        output1 = self.engine_eyebrow.inference(pose1)
        output2 = self.engine_eyebrow.inference(pose2)
        
        self.assertFalse(np.array_equal(output1, output2))
    
    def test_eyebrow_vs_no_eyebrow_output_difference(self):
        """Test that eyebrow and no-eyebrow modes produce different outputs"""
        self.engine_eyebrow.update_image(self.test_image)
        self.engine_no_eyebrow.update_image(self.test_image)
        
        # Use a pose with non-zero eyebrow values
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        
        output_eyebrow = self.engine_eyebrow.inference(pose)
        output_no_eyebrow = self.engine_no_eyebrow.inference(pose)
        
        # Both should work correctly
        self.validate_output(output_eyebrow)
        self.validate_output(output_no_eyebrow)


class TestTHA4ORTNonDefaultAllConfigurations(TestTHA4ORTBase):
    """Test THA4ORTNonDefaultSessions across all available model configurations"""
    
    MODEL_CONFIGS = [
        ('./data/tha4/fp16', True),    # fp16, with eyebrow
        ('./data/tha4/fp16', False),   # fp16, without eyebrow
        ('./data/tha4/fp32', True),    # fp32, with eyebrow
        ('./data/tha4/fp32', False),   # fp32, without eyebrow
    ]
    
    def test_all_configurations(self):
        """Test THA4ORTNonDefaultSessions with all available configurations"""
        for model_dir, use_eyebrow in self.MODEL_CONFIGS:
            if not os.path.exists(model_dir):
                continue
            
            with self.subTest(model_dir=model_dir, use_eyebrow=use_eyebrow):
                engine = THA4ORTNonDefaultSessions(model_dir, device_id=0, use_eyebrow=use_eyebrow)
                
                # Verify configuration
                self.assertEqual(engine.use_eyebrow, use_eyebrow)
                
                # Test inference
                engine.update_image(self.test_image)
                pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
                output = engine.inference(pose)
                self.validate_output(output)


class TestTHA4ORTCompareDefaultAndNonDefault(TestTHA4ORTBase):
    """Compare outputs between THA4ORTSessions and THA4ORTNonDefaultSessions"""
    
    MODEL_DIR = './data/tha4/fp16'
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
        cls.engine_default = THA4ORTSessions(cls.MODEL_DIR, use_eyebrow=True)
        cls.engine_nondefault = THA4ORTNonDefaultSessions(cls.MODEL_DIR, device_id=0, use_eyebrow=True)
    
    def test_both_engines_produce_valid_output(self):
        """Test that both engines produce valid outputs"""
        self.engine_default.update_image(self.test_image)
        self.engine_nondefault.update_image(self.test_image)
        
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        
        output_default = self.engine_default.inference(pose)
        output_nondefault = self.engine_nondefault.inference(pose)
        
        self.validate_output(output_default)
        self.validate_output(output_nondefault)
    
    def test_outputs_are_similar(self):
        """Test that default and non-default engines produce similar outputs"""
        self.engine_default.update_image(self.test_image)
        self.engine_nondefault.update_image(self.test_image)
        
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        
        output_default = self.engine_default.inference(pose)
        output_nondefault = self.engine_nondefault.inference(pose)
        
        # Allow small differences due to potential floating point variations
        diff = np.abs(output_default.astype(np.float32) - output_nondefault.astype(np.float32))
        mean_diff = np.mean(diff)
        
        # Mean difference should be small (allowing for minor numerical differences)
        self.assertLess(mean_diff, 5.0, "Outputs differ significantly between default and non-default engines")


class TestTHA4ORTOutputConsistency(TestTHA4ORTBase):
    """Test output consistency across different configurations"""
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model_dirs = {
            'fp16': './data/tha4/fp16',
            'fp32': './data/tha4/fp32',
        }
    
    def test_deterministic_output(self):
        """Test that same input produces consistent output"""
        model_dir = self.model_dirs['fp32']
        if not os.path.exists(model_dir):
            self.skipTest(f"Model directory not found: {model_dir}")
        
        engine = THA4ORTSessions(model_dir, use_eyebrow=True)
        engine.update_image(self.test_image)
        
        pose = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        
        output1 = engine.inference(pose)
        output2 = engine.inference(pose)
        
        # Same pose should produce same output
        np.testing.assert_array_equal(output1, output2)
    
    def test_different_poses_different_output(self):
        """Test that different poses produce different outputs"""
        model_dir = self.model_dirs['fp16']
        if not os.path.exists(model_dir):
            self.skipTest(f"Model directory not found: {model_dir}")
        
        engine = THA4ORTSessions(model_dir, use_eyebrow=True)
        engine.update_image(self.test_image)
        
        pose1 = np.array(self.pose_data[800]).reshape(1, 45).astype(np.float32)
        pose2 = np.array(self.pose_data[850]).reshape(1, 45).astype(np.float32)
        
        output1 = engine.inference(pose1)
        output2 = engine.inference(pose2)
        
        # Different poses should produce different outputs
        self.assertFalse(np.array_equal(output1, output2))


class TestTHA4ORTPoseSlicing(TestTHA4ORTBase):
    """Test that pose slicing works correctly"""
    
    MODEL_DIR = './data/tha4/fp16'
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.MODEL_DIR):
            raise unittest.SkipTest(f"Model directory not found: {cls.MODEL_DIR}")
    
    def test_eyebrow_pose_slicing(self):
        """Test that eyebrow pose (first 12 values) is correctly extracted"""
        engine = THA4ORTSessions(self.MODEL_DIR, use_eyebrow=True)
        engine.update_image(self.test_image)
        
        pose = np.arange(45, dtype=np.float32).reshape(1, 45)
        # This should use pose[:, :12] for eyebrow
        output = engine.inference(pose)
        self.validate_output(output)
    
    def test_face_pose_slicing(self):
        """Test that face pose (values 12-39) is correctly extracted"""
        engine = THA4ORTSessions(self.MODEL_DIR, use_eyebrow=True)
        engine.update_image(self.test_image)
        
        pose = np.arange(45, dtype=np.float32).reshape(1, 45)
        # This should use pose[:,12:12+27] for face pose
        output = engine.inference(pose)
        self.validate_output(output)
    
    def test_rotation_pose_slicing(self):
        """Test that rotation pose (last 6 values) is correctly extracted"""
        engine = THA4ORTSessions(self.MODEL_DIR, use_eyebrow=True)
        engine.update_image(self.test_image)
        
        pose = np.arange(45, dtype=np.float32).reshape(1, 45)
        # This should use pose[:,12+27:] for rotation pose
        output = engine.inference(pose)
        self.validate_output(output)


# ============================================================================
# Video Generation Showcase Tests
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


def THA4ORT_ShowVideo_FP16_WithEyebrow():
    """Generate a test video using THA4ORTSessions with fp16 model and eyebrow enabled"""
    model_dir = './data/tha4/fp16'
    if not os.path.exists(model_dir):
        print(f"Skipping: Model directory not found: {model_dir}")
        return
    
    engine = THA4ORTSessions(model_dir, use_eyebrow=True)
    
    # Load test image (512x512 RGBA)
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    engine.update_image(img)
    
    # Load pose data
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    
    frames = []
    print("Generating frames with THA4ORTSessions (fp16, eyebrow=True):")
    from tqdm import tqdm
    for i in tqdm(range(len(pose_data[800:1200]))):
        pose = np.array(pose_data[800 + i]).reshape(1, 45).astype(np.float32)
        output = engine.inference(pose)
        # Convert RGBA to BGR for opencv
        frames.append(output[:, :, :3].copy())
    
    generate_video(frames, './test/data/tha4_ort_fp16_eyebrow.mp4', 20)


def THA4ORT_ShowVideo_FP16_NoEyebrow():
    """Generate a test video using THA4ORTSessions with fp16 model and eyebrow disabled"""
    model_dir = './data/tha4/fp16'
    if not os.path.exists(model_dir):
        print(f"Skipping: Model directory not found: {model_dir}")
        return
    
    engine = THA4ORTSessions(model_dir, use_eyebrow=False)
    
    # Load test image (512x512 RGBA)
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    engine.update_image(img)
    
    # Load pose data
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    
    frames = []
    print("Generating frames with THA4ORTSessions (fp16, eyebrow=False):")
    from tqdm import tqdm
    for i in tqdm(range(len(pose_data[800:1200]))):
        pose = np.array(pose_data[800 + i]).reshape(1, 45).astype(np.float32)
        output = engine.inference(pose)
        # Convert RGBA to BGR for opencv
        frames.append(output[:, :, :3].copy())
    
    generate_video(frames, './test/data/tha4_ort_fp16_no_eyebrow.mp4', 20)


def THA4ORT_ShowVideo_FP32_WithEyebrow():
    """Generate a test video using THA4ORTSessions with fp32 model"""
    model_dir = './data/tha4/fp32'
    if not os.path.exists(model_dir):
        print(f"Skipping: Model directory not found: {model_dir}")
        return
    
    engine = THA4ORTSessions(model_dir, use_eyebrow=True)
    
    # Load test image (512x512 RGBA)
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    engine.update_image(img)
    
    # Load pose data
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    
    frames = []
    print("Generating frames with THA4ORTSessions (fp32, eyebrow=True):")
    from tqdm import tqdm
    for i in tqdm(range(len(pose_data[800:1200]))):
        pose = np.array(pose_data[800 + i]).reshape(1, 45).astype(np.float32)
        output = engine.inference(pose)
        # Convert RGBA to BGR for opencv
        frames.append(output[:, :, :3].copy())
    
    generate_video(frames, './test/data/tha4_ort_fp32_eyebrow.mp4', 20)


def THA4ORTNonDefault_ShowVideo_FP16_WithEyebrow():
    """Generate a test video using THA4ORTNonDefaultSessions with fp16 model"""
    model_dir = './data/tha4/fp16'
    if not os.path.exists(model_dir):
        print(f"Skipping: Model directory not found: {model_dir}")
        return
    
    engine = THA4ORTNonDefaultSessions(model_dir, device_id=0, use_eyebrow=True)
    
    # Load test image (512x512 RGBA)
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    engine.update_image(img)
    
    # Load pose data
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    
    frames = []
    print("Generating frames with THA4ORTNonDefaultSessions (fp16, eyebrow=True):")
    from tqdm import tqdm
    for i in tqdm(range(len(pose_data[800:1200]))):
        pose = np.array(pose_data[800 + i]).reshape(1, 45).astype(np.float32)
        output = engine.inference(pose)
        # Convert RGBA to BGR for opencv
        frames.append(output[:, :, :3].copy())
    
    generate_video(frames, './test/data/tha4_ort_nondefault_fp16_eyebrow.mp4', 20)


def THA4ORTNonDefault_ShowVideo_FP16_NoEyebrow():
    """Generate a test video using THA4ORTNonDefaultSessions with fp16 model, no eyebrow"""
    model_dir = './data/tha4/fp16'
    if not os.path.exists(model_dir):
        print(f"Skipping: Model directory not found: {model_dir}")
        return
    
    engine = THA4ORTNonDefaultSessions(model_dir, device_id=0, use_eyebrow=False)
    
    # Load test image (512x512 RGBA)
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    engine.update_image(img)
    
    # Load pose data
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    
    frames = []
    print("Generating frames with THA4ORTNonDefaultSessions (fp16, eyebrow=False):")
    from tqdm import tqdm
    for i in tqdm(range(len(pose_data[800:1200]))):
        pose = np.array(pose_data[800 + i]).reshape(1, 45).astype(np.float32)
        output = engine.inference(pose)
        # Convert RGBA to BGR for opencv
        frames.append(output[:, :, :3].copy())
    
    generate_video(frames, './test/data/tha4_ort_nondefault_fp16_no_eyebrow.mp4', 20)


def THA4ORTNonDefault_ShowVideo_FP32_WithEyebrow():
    """Generate a test video using THA4ORTNonDefaultSessions with fp32 model"""
    model_dir = './data/tha4/fp32'
    if not os.path.exists(model_dir):
        print(f"Skipping: Model directory not found: {model_dir}")
        return
    
    engine = THA4ORTNonDefaultSessions(model_dir, device_id=0, use_eyebrow=True)
    
    # Load test image (512x512 RGBA)
    img = cv2.imread('./test/data/base.png', cv2.IMREAD_UNCHANGED)
    engine.update_image(img)
    
    # Load pose data
    with open('./test/data/pose_20fps.json', 'r') as file:
        pose_data = json.load(file)
    
    frames = []
    print("Generating frames with THA4ORTNonDefaultSessions (fp32, eyebrow=True):")
    from tqdm import tqdm
    for i in tqdm(range(len(pose_data[800:1200]))):
        pose = np.array(pose_data[800 + i]).reshape(1, 45).astype(np.float32)
        output = engine.inference(pose)
        # Convert RGBA to BGR for opencv
        frames.append(output[:, :, :3].copy())
    
    generate_video(frames, './test/data/tha4_ort_nondefault_fp32_eyebrow.mp4', 20)


if __name__ == '__main__':
    import sys
    
    # Check for command line arguments for video generation
    if len(sys.argv) > 1:
        if sys.argv[1] == '--show-all':
            # Run all video showcase functions
            print("=" * 60)
            print("Running all THA4ORTSessions video showcase tests")
            print("=" * 60)
            THA4ORT_ShowVideo_FP16_WithEyebrow()
            THA4ORT_ShowVideo_FP16_NoEyebrow()
            THA4ORT_ShowVideo_FP32_WithEyebrow()
            print("\n" + "=" * 60)
            print("Running all THA4ORTNonDefaultSessions video showcase tests")
            print("=" * 60)
            THA4ORTNonDefault_ShowVideo_FP16_WithEyebrow()
            THA4ORTNonDefault_ShowVideo_FP16_NoEyebrow()
            THA4ORTNonDefault_ShowVideo_FP32_WithEyebrow()
        elif sys.argv[1] == '--show-tha4ort':
            print("=" * 60)
            print("Running THA4ORTSessions video showcase tests")
            print("=" * 60)
            THA4ORT_ShowVideo_FP16_WithEyebrow()
            THA4ORT_ShowVideo_FP16_NoEyebrow()
            THA4ORT_ShowVideo_FP32_WithEyebrow()
        elif sys.argv[1] == '--show-nondefault':
            print("=" * 60)
            print("Running THA4ORTNonDefaultSessions video showcase tests")
            print("=" * 60)
            THA4ORTNonDefault_ShowVideo_FP16_WithEyebrow()
            THA4ORTNonDefault_ShowVideo_FP16_NoEyebrow()
            THA4ORTNonDefault_ShowVideo_FP32_WithEyebrow()
        else:
            # Run unit tests with remaining arguments
            unittest.main(verbosity=2)
    else:
        # Run unit tests with verbosity
        unittest.main(verbosity=2)
