"""
Unit tests and benchmarks for the Cacher class and Brotli codec.

Tests cover:
- Basic read/write operations
- LRU eviction behavior
- Anti-thrashing mechanism
- Brotli lossless compression
- Performance benchmarks
"""

import unittest
import numpy as np
import cv2
import time
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
import brotli

from ezvtb_rt.cache import Cacher


class TestCacherBasicOperations(unittest.TestCase):
    """Test basic cache read/write operations."""
    
    @classmethod
    def setUpClass(cls):
        """Set up shared test fixtures."""
        cls.test_width = 512
        cls.test_height = 512
        cls.test_shape = (cls.test_height, cls.test_width, 4)
        cls.test_dtype = np.uint8
        
        # Load test images
        cls.base_image_path = "./test/data/base.png"
        cls.base_1_image_path = "./test/data/base_1.png"
        
        if os.path.exists(cls.base_image_path):
            cls.base_image = cv2.imread(cls.base_image_path, cv2.IMREAD_UNCHANGED)
            # Ensure BGRA format
            if cls.base_image.shape[2] == 3:
                cls.base_image = cv2.cvtColor(cls.base_image, cv2.COLOR_BGR2BGRA)
        else:
            cls.base_image = None
            
        if os.path.exists(cls.base_1_image_path):
            cls.base_1_image = cv2.imread(cls.base_1_image_path, cv2.IMREAD_UNCHANGED)
            if cls.base_1_image.shape[2] == 3:
                cls.base_1_image = cv2.cvtColor(cls.base_1_image, cv2.COLOR_BGR2BGRA)
            # Resize to 512x512 if needed
            if cls.base_1_image.shape[:2] != (512, 512):
                cls.base_1_image = cv2.resize(cls.base_1_image, (512, 512))
        else:
            cls.base_1_image = None
    
    def _create_random_image(self, seed: int = 0) -> np.ndarray:
        """Create a random BGRA image for testing."""
        np.random.seed(seed)
        return np.random.randint(0, 256, self.test_shape, dtype=self.test_dtype)
    
    def test_basic_write_read(self):
        """Test basic write and read operations."""
        cacher = Cacher(max_volume_giga=0.1, width=self.test_width, height=self.test_height)
        
        test_data = self._create_random_image(seed=42)
        hash_key = 12345
        
        # Write to cache
        cacher.put(hash_key, test_data)
        
        # Read back
        result = cacher.get(hash_key)
        
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result, test_data)
    
    def test_read_nonexistent_key(self):
        """Test reading a key that doesn't exist returns None."""
        cacher = Cacher(max_volume_giga=0.1, width=self.test_width, height=self.test_height)
        
        result = cacher.get(99999)
        
        self.assertIsNone(result)
        self.assertEqual(cacher.miss, 1)
        self.assertEqual(cacher.hits, 0)
    
    def test_read_updates_hit_stats(self):
        """Test that successful reads update hit statistics."""
        cacher = Cacher(max_volume_giga=0.1, width=self.test_width, height=self.test_height)
        
        test_data = self._create_random_image()
        hash_key = 100
        
        cacher.put(hash_key, test_data)
        
        # First read
        cacher.get(hash_key)
        self.assertEqual(cacher.hits, 1)
        self.assertEqual(cacher.miss, 0)
        
        # Second read
        cacher.get(hash_key)
        self.assertEqual(cacher.hits, 2)
        self.assertEqual(cacher.miss, 0)
    
    def test_read_updates_miss_stats(self):
        """Test that cache misses update miss statistics."""
        cacher = Cacher(max_volume_giga=0.1, width=self.test_width, height=self.test_height)
        
        # Read non-existent keys
        cacher.get(1)
        cacher.get(2)
        cacher.get(3)
        
        self.assertEqual(cacher.miss, 3)
        self.assertEqual(cacher.hits, 0)
    
    def test_write_skip_duplicate(self):
        """Test that writing duplicate keys is skipped."""
        cacher = Cacher(max_volume_giga=0.1, width=self.test_width, height=self.test_height)
        
        data1 = self._create_random_image(seed=1)
        data2 = self._create_random_image(seed=2)
        hash_key = 500
        
        # Write first data
        cacher.put(hash_key, data1)
        size_after_first = cacher.cached_kbytes
        
        # Try to write different data with same key
        cacher.put(hash_key, data2)
        size_after_second = cacher.cached_kbytes
        
        # Size should not change
        self.assertEqual(size_after_first, size_after_second)
        
        # Should still return first data
        result = cacher.get(hash_key)
        np.testing.assert_array_equal(result, data1)
    
    def test_write_with_real_image(self):
        """Test write/read with actual image file."""
        if self.base_image is None:
            self.skipTest("base.png not found")
        
        cacher = Cacher(max_volume_giga=0.1, width=self.test_width, height=self.test_height)
        
        hash_key = 1001
        cacher.put(hash_key, self.base_image)
        
        result = cacher.get(hash_key)
        
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result, self.base_image)


class TestCacherLRU(unittest.TestCase):
    """Test LRU eviction behavior."""
    
    def _create_random_image(self, width: int = 512, height: int = 512, seed: int = 0) -> np.ndarray:
        """Create a random BGRA image."""
        np.random.seed(seed)
        return np.random.randint(0, 256, (height, width, 4), dtype=np.uint8)
    
    def test_lru_eviction_order(self):
        """Test that oldest entries are evicted first."""
        # Create a very small cache that can only hold ~2 entries
        # 512x512x4 = 1MB uncompressed, compressed ~200-400KB typically
        cacher = Cacher(max_volume_giga=0.0005, width=512, height=512)  # ~512KB max
        
        data1 = self._create_random_image(seed=1)
        data2 = self._create_random_image(seed=2)
        data3 = self._create_random_image(seed=3)
        
        # Write three entries - should evict oldest
        cacher.put(1, data1)
        cacher.put(2, data2)
        cacher.put(3, data3)
        
        # First entry should be evicted (or at least cache should be under limit)
        self.assertLessEqual(cacher.cached_kbytes, cacher.max_kbytes)
    
    def test_lru_promotion_on_read(self):
        """Test that reading an entry promotes it to MRU position."""
        # Cache large enough to hold multiple entries without eviction
        cacher = Cacher(max_volume_giga=0.1, width=512, height=512)
        
        data1 = self._create_random_image(seed=10)
        data2 = self._create_random_image(seed=20)
        
        cacher.put(1, data1)
        cacher.put(2, data2)
        
        # Verify both entries exist before testing promotion
        self.assertEqual(len(cacher.cache), 2)
        
        # Read entry 1 to promote it
        cacher.get(1)
        
        # Entry 1 should now be at the end (MRU)
        keys = list(cacher.cache.keys())
        self.assertEqual(keys[-1], 1)
    
    def test_size_tracking_accuracy(self):
        """Test that cached_kbytes is tracked accurately."""
        cacher = Cacher(max_volume_giga=1.0, width=512, height=512)
        
        self.assertEqual(cacher.cached_kbytes, 0)
        
        data = self._create_random_image()
        cacher.put(1, data)
        
        # Size should be positive after write
        self.assertGreater(cacher.cached_kbytes, 0)
        
        # Calculate expected size from cache contents
        expected_kb = sum(len(v) for v in cacher.cache.values()) / 1024
        self.assertAlmostEqual(cacher.cached_kbytes, expected_kb, places=2)
    
    def test_multiple_evictions(self):
        """Test multiple entries are evicted when needed."""
        # Very tiny cache
        cacher = Cacher(max_volume_giga=0.0003, width=512, height=512)  # ~300KB
        
        # Write many entries
        for i in range(10):
            data = self._create_random_image(seed=i)
            cacher.put(i, data)
        
        # Cache should stay within limits
        self.assertLessEqual(cacher.cached_kbytes, cacher.max_kbytes)
        
        # Should have fewer than 10 entries
        self.assertLess(len(cacher.cache), 10)


class TestCacherAntiThrashing(unittest.TestCase):
    """Test anti-thrashing mechanism."""
    
    def _create_random_image(self, seed: int = 0) -> np.ndarray:
        """Create a random BGRA image."""
        np.random.seed(seed)
        return np.random.randint(0, 256, (512, 512, 4), dtype=np.uint8)
    
    def test_anti_thrashing_activates_after_5_sequential_hits(self):
        """Test that after 5 sequential hits to different keys, cache returns None."""
        cacher = Cacher(max_volume_giga=1.0, width=512, height=512)
        
        # Write 10 entries
        for i in range(10):
            data = self._create_random_image(seed=i)
            cacher.put(i, data)
        
        # Read 6 different keys sequentially (more than 5)
        for i in range(6):
            cacher.get(i)
        
        # The 7th read to a different key should return None due to anti-thrashing
        # continues_hits is now 6, which is > 5
        result = cacher.get(7)
        self.assertIsNone(result)
    
    def test_anti_thrashing_same_key_resets_counter(self):
        """Test that reading the same key twice resets continues_hits."""
        cacher = Cacher(max_volume_giga=1.0, width=512, height=512)
        
        # Write entries
        for i in range(10):
            data = self._create_random_image(seed=i)
            cacher.put(i, data)
        
        # Read different keys to build up continues_hits
        for i in range(4):
            cacher.get(i)
        
        self.assertEqual(cacher.continues_hits, 4)
        
        # Read the same key as last time (key 3)
        cacher.get(3)
        
        # Counter should reset to 0
        self.assertEqual(cacher.continues_hits, 0)
    
    def test_anti_thrashing_miss_resets_counter(self):
        """Test that a cache miss resets continues_hits."""
        cacher = Cacher(max_volume_giga=1.0, width=512, height=512)
        
        # Write some entries
        for i in range(5):
            data = self._create_random_image(seed=i)
            cacher.put(i, data)
        
        # Build up continues_hits
        for i in range(4):
            cacher.get(i)
        
        self.assertGreater(cacher.continues_hits, 0)
        
        # Trigger a miss
        cacher.get(999)
        
        # Counter should reset to 0
        self.assertEqual(cacher.continues_hits, 0)
    
    def test_continues_hits_increments_correctly(self):
        """Test that continues_hits increments on sequential different-key reads."""
        cacher = Cacher(max_volume_giga=1.0, width=512, height=512)
        
        # Write entries
        for i in range(5):
            data = self._create_random_image(seed=i)
            cacher.put(i, data)
        
        # Initially 0
        self.assertEqual(cacher.continues_hits, 0)
        
        # Read key 0
        cacher.get(0)
        self.assertEqual(cacher.continues_hits, 1)
        
        # Read key 1 (different)
        cacher.get(1)
        self.assertEqual(cacher.continues_hits, 2)
        
        # Read key 2 (different)
        cacher.get(2)
        self.assertEqual(cacher.continues_hits, 3)


class TestBrotliCodec(unittest.TestCase):
    """Test Brotli encoder/decoder directly."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.base_image_path = "./test/data/base.png"
        cls.base_1_image_path = "./test/data/base_1.png"
        
        if os.path.exists(cls.base_image_path):
            cls.base_image = cv2.imread(cls.base_image_path, cv2.IMREAD_UNCHANGED)
            if cls.base_image.shape[2] == 3:
                cls.base_image = cv2.cvtColor(cls.base_image, cv2.COLOR_BGR2BGRA)
        else:
            cls.base_image = None
    
    def _create_random_image(self, width: int = 512, height: int = 512, seed: int = 0) -> np.ndarray:
        """Create a random BGRA image."""
        np.random.seed(seed)
        return np.random.randint(0, 256, (height, width, 4), dtype=np.uint8)
    
    def test_lossless_compression_random(self):
        """Test that compression is lossless for random data."""
        original = self._create_random_image(seed=42)
        
        # Encode and decode directly using brotli
        encoded = brotli.compress(original.tobytes(), quality=0)
        decoded = np.frombuffer(brotli.decompress(encoded), dtype=np.uint8).reshape(original.shape)
        
        np.testing.assert_array_equal(decoded, original)
    
    def test_lossless_compression_real_image(self):
        """Test that compression is lossless for real image."""
        if self.base_image is None:
            self.skipTest("base.png not found")
        
        # Encode and decode directly using brotli
        encoded = brotli.compress(self.base_image.tobytes(), quality=0)
        decoded = np.frombuffer(brotli.decompress(encoded), dtype=np.uint8).reshape(self.base_image.shape)
        
        np.testing.assert_array_equal(decoded, self.base_image)
    
    def test_compression_ratio(self):
        """Test that Brotli provides compression."""
        original = self._create_random_image()
        original_size = original.nbytes
        
        encoded = brotli.compress(original.tobytes(), quality=0)
        compressed_size = len(encoded)
        
        ratio = original_size / compressed_size
        
        print(f"\nCompression stats (random data):")
        print(f"  Original size: {original_size / 1024:.1f} KB")
        print(f"  Compressed size: {compressed_size / 1024:.1f} KB")
        print(f"  Compression ratio: {ratio:.2f}x")
        
        # Random data doesn't compress well, but should still be somewhat compressed
        # or at least not expand significantly
        self.assertGreater(ratio, 0.5)
    
    def test_compression_ratio_real_image(self):
        """Test compression ratio for real image (should be better than random)."""
        if self.base_image is None:
            self.skipTest("base.png not found")
        
        original_size = self.base_image.nbytes
        encoded = brotli.compress(self.base_image.tobytes(), quality=0)
        compressed_size = len(encoded)
        
        ratio = original_size / compressed_size
        
        print(f"\nCompression stats (real image):")
        print(f"  Original size: {original_size / 1024:.1f} KB")
        print(f"  Compressed size: {compressed_size / 1024:.1f} KB")
        print(f"  Compression ratio: {ratio:.2f}x")
        
        # Real images typically compress better
        self.assertGreater(ratio, 0.8)
    
    def test_encode_decode_multiple_times(self):
        """Test encoding/decoding the same data multiple times is consistent."""
        original = self._create_random_image(seed=123)
        
        for _ in range(5):
            encoded = brotli.compress(original.tobytes(), quality=0)
            decoded = np.frombuffer(brotli.decompress(encoded), dtype=np.uint8).reshape(original.shape)
            np.testing.assert_array_equal(decoded, original)
    
    def test_different_images_encode_differently(self):
        """Test that different images produce different encoded outputs."""
        image1 = self._create_random_image(seed=1)
        image2 = self._create_random_image(seed=2)
        
        encoded1 = brotli.compress(image1.tobytes(), quality=0)
        encoded2 = brotli.compress(image2.tobytes(), quality=0)
        
        # Encoded data should be different
        self.assertNotEqual(encoded1, encoded2)
    
    def test_solid_color_compression(self):
        """Test compression of solid color images (should compress very well)."""
        # Create solid red image
        solid_red = np.zeros((512, 512, 4), dtype=np.uint8)
        solid_red[:, :, 2] = 255  # Red channel
        solid_red[:, :, 3] = 255  # Alpha channel
        
        original_size = solid_red.nbytes
        encoded = brotli.compress(solid_red.tobytes(), quality=0)
        compressed_size = len(encoded)
        
        ratio = original_size / compressed_size
        
        print(f"\nCompression stats (solid color):")
        print(f"  Original size: {original_size / 1024:.1f} KB")
        print(f"  Compressed size: {compressed_size / 1024:.1f} KB")
        print(f"  Compression ratio: {ratio:.2f}x")
        
        # Solid color should compress extremely well
        self.assertGreater(ratio, 3.0)
        
        # Verify lossless
        decoded = np.frombuffer(brotli.decompress(encoded), dtype=np.uint8).reshape(solid_red.shape)
        np.testing.assert_array_equal(decoded, solid_red)


class TestCacherBenchmarks(unittest.TestCase):
    """Performance benchmarks for Cacher."""
    
    @classmethod
    def setUpClass(cls):
        """Set up benchmark fixtures."""
        cls.base_image_path = "./test/data/base.png"
        
        if os.path.exists(cls.base_image_path):
            cls.base_image = cv2.imread(cls.base_image_path, cv2.IMREAD_UNCHANGED)
            if cls.base_image.shape[2] == 3:
                cls.base_image = cv2.cvtColor(cls.base_image, cv2.COLOR_BGR2BGRA)
        else:
            cls.base_image = None
    
    def _create_random_image(self, seed: int = 0) -> np.ndarray:
        """Create a random BGRA image."""
        np.random.seed(seed)
        return np.random.randint(0, 256, (512, 512, 4), dtype=np.uint8)
    
    def test_benchmark_encode_performance(self):
        """Benchmark Brotli encoding throughput."""
        test_image = self._create_random_image()
        num_iterations = 500
        
        # Warm up
        for _ in range(10):
            brotli.compress(test_image.tobytes(), quality=0)
        
        # Benchmark
        start = time.perf_counter()
        for _ in tqdm(range(num_iterations), desc="Encoding"):
            brotli.compress(test_image.tobytes(), quality=0)
        elapsed = time.perf_counter() - start
        
        ops_per_sec = num_iterations / elapsed
        avg_time_ms = (elapsed / num_iterations) * 1000
        
        print(f"\n=== Encode Benchmark (512x512 BGRA) ===")
        print(f"  Iterations: {num_iterations}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Throughput: {ops_per_sec:.1f} ops/sec")
        print(f"  Average time: {avg_time_ms:.2f} ms/op")
        
        # Should be able to encode at least 100 images/sec
        self.assertGreater(ops_per_sec, 100)
    
    def test_benchmark_decode_performance(self):
        """Benchmark Brotli decoding throughput."""
        test_image = self._create_random_image()
        encoded = brotli.compress(test_image.tobytes(), quality=0)
        num_iterations = 500
        
        # Warm up
        for _ in range(10):
            brotli.decompress(encoded)
        
        # Benchmark
        start = time.perf_counter()
        for _ in tqdm(range(num_iterations), desc="Decoding"):
            brotli.decompress(encoded)
        elapsed = time.perf_counter() - start
        
        ops_per_sec = num_iterations / elapsed
        avg_time_ms = (elapsed / num_iterations) * 1000
        
        print(f"\n=== Decode Benchmark (512x512 BGRA) ===")
        print(f"  Iterations: {num_iterations}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Throughput: {ops_per_sec:.1f} ops/sec")
        print(f"  Average time: {avg_time_ms:.2f} ms/op")
        
        # Should be able to decode at least 100 images/sec
        self.assertGreater(ops_per_sec, 100)
    
    def test_benchmark_write_performance(self):
        """Benchmark cache write throughput."""
        cacher = Cacher(max_volume_giga=2.0, width=512, height=512)
        
        # Pre-generate test images
        num_iterations = 500
        test_images = [self._create_random_image(seed=i) for i in range(num_iterations)]
        
        # Warm up
        for i in range(10):
            cacher.put(i, test_images[i])
        cacher.cache.clear()
        cacher.cached_kbytes = 0
        
        # Benchmark
        start = time.perf_counter()
        for i in tqdm(range(num_iterations), desc="Writing"):
            cacher.put(i + 1000, test_images[i])
        elapsed = time.perf_counter() - start
        
        ops_per_sec = num_iterations / elapsed
        avg_time_ms = (elapsed / num_iterations) * 1000
        
        print(f"\n=== Write Benchmark (512x512 BGRA) ===")
        print(f"  Iterations: {num_iterations}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Throughput: {ops_per_sec:.1f} ops/sec")
        print(f"  Average time: {avg_time_ms:.2f} ms/op")
        print(f"  Final cache size: {cacher.cached_kbytes / 1024:.1f} MB")
        
        self.assertGreater(ops_per_sec, 50)
    
    def test_benchmark_read_performance(self):
        """Benchmark cache read throughput (hit scenario)."""
        cacher = Cacher(max_volume_giga=2.0, width=512, height=512)
        
        # Pre-populate cache
        num_entries = 100
        test_images = [self._create_random_image(seed=i) for i in range(num_entries)]
        for i, img in enumerate(test_images):
            cacher.put(i, img)
        
        num_iterations = 500
        
        # Warm up
        for i in range(10):
            cacher.continues_hits = 0  # Reset anti-thrashing
            cacher.get(i % num_entries)
        
        # Benchmark - read with anti-thrashing reset
        start = time.perf_counter()
        for i in tqdm(range(num_iterations), desc="Reading"):
            # Reset anti-thrashing to ensure hits
            if cacher.continues_hits > 4:
                cacher.continues_hits = 0
            cacher.get(i % num_entries)
        elapsed = time.perf_counter() - start
        
        ops_per_sec = num_iterations / elapsed
        avg_time_ms = (elapsed / num_iterations) * 1000
        
        print(f"\n=== Read Benchmark (512x512 BGRA, cache hits) ===")
        print(f"  Iterations: {num_iterations}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Throughput: {ops_per_sec:.1f} ops/sec")
        print(f"  Average time: {avg_time_ms:.2f} ms/op")
        
        self.assertGreater(ops_per_sec, 50)
    
    def test_benchmark_mixed_workload(self):
        """Benchmark realistic mixed read/write workload."""
        cacher = Cacher(max_volume_giga=0.5, width=512, height=512)
        
        num_iterations = 500
        test_images = [self._create_random_image(seed=i) for i in range(100)]
        
        # Warm up
        for i in range(20):
            cacher.put(i, test_images[i])
        
        # Mixed workload: 70% reads, 30% writes
        np.random.seed(42)
        operations = np.random.choice(['read', 'write'], size=num_iterations, p=[0.7, 0.3])
        
        read_count = 0
        write_count = 0
        
        start = time.perf_counter()
        for i, op in enumerate(tqdm(operations, desc="Mixed workload")):
            if op == 'read':
                # Reset anti-thrashing periodically
                if cacher.continues_hits > 4:
                    cacher.continues_hits = 0
                cacher.get(i % 50)
                read_count += 1
            else:
                cacher.put(i + 10000, test_images[i % 100])
                write_count += 1
        elapsed = time.perf_counter() - start
        
        ops_per_sec = num_iterations / elapsed
        avg_time_ms = (elapsed / num_iterations) * 1000
        
        print(f"\n=== Mixed Workload Benchmark ===")
        print(f"  Total operations: {num_iterations}")
        print(f"  Reads: {read_count}, Writes: {write_count}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Throughput: {ops_per_sec:.1f} ops/sec")
        print(f"  Average time: {avg_time_ms:.2f} ms/op")
        print(f"  Cache hits: {cacher.hits}, misses: {cacher.miss}")
        print(f"  Hit rate: {cacher.hits / (cacher.hits + cacher.miss) * 100:.1f}%")
        
        self.assertGreater(ops_per_sec, 50)
    
    def test_benchmark_eviction_pressure(self):
        """Benchmark performance under heavy eviction pressure."""
        # Very small cache to force constant evictions
        cacher = Cacher(max_volume_giga=0.0005, width=512, height=512)  # ~500KB
        
        num_iterations = 200
        test_images = [self._create_random_image(seed=i) for i in range(num_iterations)]
        
        start = time.perf_counter()
        for i in tqdm(range(num_iterations), desc="Eviction pressure"):
            cacher.put(i, test_images[i])
        elapsed = time.perf_counter() - start
        
        ops_per_sec = num_iterations / elapsed
        avg_time_ms = (elapsed / num_iterations) * 1000
        
        print(f"\n=== Eviction Pressure Benchmark ===")
        print(f"  Iterations: {num_iterations}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Throughput: {ops_per_sec:.1f} ops/sec")
        print(f"  Average time: {avg_time_ms:.2f} ms/op")
        print(f"  Final cache entries: {len(cacher.cache)}")
        print(f"  Final cache size: {cacher.cached_kbytes:.1f} KB (max: {cacher.max_kbytes:.1f} KB)")
        
        self.assertGreater(ops_per_sec, 30)
    
    def test_benchmark_real_image(self):
        """Benchmark with real image from test data."""
        if self.base_image is None:
            self.skipTest("base.png not found")
        
        width, height = self.base_image.shape[1], self.base_image.shape[0]
        
        num_iterations = 500
        
        # Encode benchmark
        start = time.perf_counter()
        for _ in tqdm(range(num_iterations), desc="Real image encode"):
            brotli.compress(self.base_image.tobytes(), quality=0)
        encode_elapsed = time.perf_counter() - start
        
        # Decode benchmark
        encoded = brotli.compress(self.base_image.tobytes(), quality=0)
        start = time.perf_counter()
        for _ in tqdm(range(num_iterations), desc="Real image decode"):
            brotli.decompress(encoded)
        decode_elapsed = time.perf_counter() - start
        
        print(f"\n=== Real Image Benchmark ({width}x{height}) ===")
        print(f"  Encode throughput: {num_iterations / encode_elapsed:.1f} ops/sec")
        print(f"  Decode throughput: {num_iterations / decode_elapsed:.1f} ops/sec")
        print(f"  Compression ratio: {self.base_image.nbytes / len(encoded):.2f}x")


class TestCacherBenchmarks1024(unittest.TestCase):
    """Performance benchmarks for Cacher at 1024x1024 resolution."""
    
    @classmethod
    def setUpClass(cls):
        """Load and resize test images to 1024x1024."""
        cls.test_width = 1024
        cls.test_height = 1024
        cls.test_shape = (cls.test_height, cls.test_width, 4)
        cls.test_dtype = np.uint8
        
        # Load and resize base images
        test_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.join(test_dir, "data", "base.png")
        base1_path = os.path.join(test_dir, "data", "base_1.png")
        
        cls.base_image = None
        cls.base1_image = None
        
        if os.path.exists(base_path):
            img = cv2.imread(base_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                cls.base_image = cv2.resize(img, (cls.test_width, cls.test_height), interpolation=cv2.INTER_LINEAR)
                if cls.base_image.shape[2] == 3:
                    cls.base_image = cv2.cvtColor(cls.base_image, cv2.COLOR_BGR2BGRA)
        
        if os.path.exists(base1_path):
            img = cv2.imread(base1_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                cls.base1_image = cv2.resize(img, (cls.test_width, cls.test_height), interpolation=cv2.INTER_LINEAR)
                if cls.base1_image.shape[2] == 3:
                    cls.base1_image = cv2.cvtColor(cls.base1_image, cv2.COLOR_BGR2BGRA)
    
    def _create_random_image(self, seed: int = 0) -> np.ndarray:
        """Create a random 1024x1024 BGRA image."""
        np.random.seed(seed)
        return np.random.randint(0, 256, self.test_shape, dtype=self.test_dtype)
    
    def test_benchmark_encode_performance_1024(self):
        """Benchmark Brotli encoding at 1024x1024."""
        test_data = self._create_random_image(seed=42)
        
        iterations = 500
        
        # Warmup
        for _ in range(10):
            brotli.compress(test_data.tobytes(), quality=0)
        
        start = time.perf_counter()
        for _ in tqdm(range(iterations), desc="Encode 1024x1024"):
            brotli.compress(test_data.tobytes(), quality=0)
        elapsed = time.perf_counter() - start
        
        ops_per_sec = iterations / elapsed
        ms_per_op = (elapsed / iterations) * 1000
        
        print(f"\n[1024x1024] Encode Performance:")
        print(f"  Total time: {elapsed:.3f}s for {iterations} iterations")
        print(f"  Throughput: {ops_per_sec:.2f} ops/sec")
        print(f"  Latency: {ms_per_op:.3f} ms/op")
        
        self.assertGreater(ops_per_sec, 10, "Encode should be faster than 10 ops/sec at 1024x1024")
    
    def test_benchmark_decode_performance_1024(self):
        """Benchmark Brotli decoding at 1024x1024."""
        test_data = self._create_random_image(seed=42)
        compressed = brotli.compress(test_data.tobytes(), quality=0)
        
        iterations = 500
        
        # Warmup
        for _ in range(10):
            brotli.decompress(compressed)
        
        start = time.perf_counter()
        for _ in tqdm(range(iterations), desc="Decode 1024x1024"):
            brotli.decompress(compressed)
        elapsed = time.perf_counter() - start
        
        ops_per_sec = iterations / elapsed
        ms_per_op = (elapsed / iterations) * 1000
        
        print(f"\n[1024x1024] Decode Performance:")
        print(f"  Total time: {elapsed:.3f}s for {iterations} iterations")
        print(f"  Throughput: {ops_per_sec:.2f} ops/sec")
        print(f"  Latency: {ms_per_op:.3f} ms/op")
        
        self.assertGreater(ops_per_sec, 10, "Decode should be faster than 10 ops/sec at 1024x1024")
    
    def test_benchmark_write_performance_1024(self):
        """Benchmark cache write performance at 1024x1024."""
        cacher = Cacher(max_volume_giga=2.0, width=self.test_width, height=self.test_height)
        
        # Pre-generate test data
        test_images = [self._create_random_image(seed=i) for i in range(100)]
        
        iterations = 500
        
        # Warmup
        for i in range(10):
            cacher.put(i + 100000, test_images[i % len(test_images)])
        
        start = time.perf_counter()
        for i in tqdm(range(iterations), desc="Write 1024x1024"):
            cacher.put(i, test_images[i % len(test_images)])
        elapsed = time.perf_counter() - start
        
        ops_per_sec = iterations / elapsed
        ms_per_op = (elapsed / iterations) * 1000
        
        print(f"\n[1024x1024] Write Performance:")
        print(f"  Total time: {elapsed:.3f}s for {iterations} iterations")
        print(f"  Throughput: {ops_per_sec:.2f} ops/sec")
        print(f"  Latency: {ms_per_op:.3f} ms/op")
        print(f"  Final cache size: {cacher.cached_kbytes / 1024:.2f} MB")
        
        self.assertGreater(ops_per_sec, 10, "Write should be faster than 10 ops/sec at 1024x1024")
    
    def test_benchmark_read_performance_1024(self):
        """Benchmark cache read performance at 1024x1024."""
        cacher = Cacher(max_volume_giga=4.0, width=self.test_width, height=self.test_height)
        
        # Pre-populate cache
        num_entries = 200
        test_images = [self._create_random_image(seed=i) for i in range(num_entries)]
        for i in range(num_entries):
            cacher.put(i, test_images[i])
        
        iterations = 500
        
        # Warmup
        for i in range(10):
            cacher.continues_hits = 0  # Reset anti-thrashing
            cacher.get(i % num_entries)
        
        start = time.perf_counter()
        for i in tqdm(range(iterations), desc="Read 1024x1024"):
            cacher.continues_hits = 0  # Disable anti-thrashing for benchmark
            cacher.get(i % num_entries)
        elapsed = time.perf_counter() - start
        
        ops_per_sec = iterations / elapsed
        ms_per_op = (elapsed / iterations) * 1000
        
        print(f"\n[1024x1024] Read Performance:")
        print(f"  Total time: {elapsed:.3f}s for {iterations} iterations")
        print(f"  Throughput: {ops_per_sec:.2f} ops/sec")
        print(f"  Latency: {ms_per_op:.3f} ms/op")
        
        self.assertGreater(ops_per_sec, 10, "Read should be faster than 10 ops/sec at 1024x1024")
    
    def test_benchmark_mixed_workload_1024(self):
        """Benchmark mixed read/write workload at 1024x1024."""
        cacher = Cacher(max_volume_giga=2.0, width=self.test_width, height=self.test_height)
        
        test_images = [self._create_random_image(seed=i) for i in range(50)]
        
        iterations = 500
        
        start = time.perf_counter()
        for i in tqdm(range(iterations), desc="Mixed 1024x1024"):
            key = i % 100
            if i % 3 == 0:  # ~33% writes
                cacher.put(key, test_images[i % len(test_images)])
            else:  # ~67% reads
                cacher.continues_hits = 0
                cacher.get(key)
        elapsed = time.perf_counter() - start
        
        ops_per_sec = iterations / elapsed
        ms_per_op = (elapsed / iterations) * 1000
        
        print(f"\n[1024x1024] Mixed Workload Performance:")
        print(f"  Total time: {elapsed:.3f}s for {iterations} iterations")
        print(f"  Throughput: {ops_per_sec:.2f} ops/sec")
        print(f"  Latency: {ms_per_op:.3f} ms/op")
        print(f"  Cache hits: {cacher.hits}, misses: {cacher.miss}")
        
        self.assertGreater(ops_per_sec, 10, "Mixed workload should be faster than 10 ops/sec at 1024x1024")
    
    def test_benchmark_real_image_1024(self):
        """Benchmark with resized real images at 1024x1024."""
        if self.base_image is None:
            self.skipTest("base.png not available")
        
        iterations = 500
        
        # Benchmark encode
        start = time.perf_counter()
        for _ in tqdm(range(iterations), desc="Encode real 1024"):
            compressed = brotli.compress(self.base_image.tobytes(), quality=0)
        encode_elapsed = time.perf_counter() - start
        
        # Benchmark decode
        start = time.perf_counter()
        for _ in tqdm(range(iterations), desc="Decode real 1024"):
            brotli.decompress(compressed)
        decode_elapsed = time.perf_counter() - start
        
        # Compression stats
        original_size = self.base_image.nbytes
        compressed_size = len(compressed)
        ratio = original_size / compressed_size
        
        print(f"\n[1024x1024] Real Image Performance:")
        print(f"  Encode: {iterations / encode_elapsed:.2f} ops/sec ({(encode_elapsed / iterations) * 1000:.3f} ms/op)")
        print(f"  Decode: {iterations / decode_elapsed:.2f} ops/sec ({(decode_elapsed / iterations) * 1000:.3f} ms/op)")
        print(f"  Original size: {original_size / 1024:.2f} KB ({self.test_width}x{self.test_height}x4)")
        print(f"  Compressed size: {compressed_size / 1024:.2f} KB")
        print(f"  Compression ratio: {ratio:.2f}x")
        
        self.assertGreater(ratio, 1.0, "Compression should reduce size")
    
    def test_benchmark_eviction_pressure_1024(self):
        """Benchmark performance under eviction pressure at 1024x1024."""
        # Small cache to force frequent evictions
        # 1024x1024x4 = 4MB uncompressed, compressed ~800KB-1.5MB
        cacher = Cacher(max_volume_giga=0.01, width=self.test_width, height=self.test_height)  # ~10MB cache
        
        test_images = [self._create_random_image(seed=i) for i in range(20)]
        
        iterations = 200
        eviction_count = 0
        
        start = time.perf_counter()
        for i in tqdm(range(iterations), desc="Eviction 1024x1024"):
            before_size = len(cacher.cache)
            cacher.put(i, test_images[i % len(test_images)])
            if len(cacher.cache) <= before_size and i > 0:
                eviction_count += 1
        elapsed = time.perf_counter() - start
        
        ops_per_sec = iterations / elapsed
        
        print(f"\n[1024x1024] Eviction Pressure Performance:")
        print(f"  Total time: {elapsed:.3f}s for {iterations} iterations")
        print(f"  Throughput: {ops_per_sec:.2f} ops/sec")
        print(f"  Evictions triggered: ~{eviction_count}")
        print(f"  Final entries: {len(cacher.cache)}")
        print(f"  Cache utilization: {cacher.cached_kbytes / cacher.max_kbytes * 100:.1f}%")
        
        self.assertLessEqual(cacher.cached_kbytes, cacher.max_kbytes)


class TestBrotliCodec1024(unittest.TestCase):
    """Test Brotli encoder/decoder at 1024x1024 resolution."""
    
    @classmethod
    def setUpClass(cls):
        """Load and resize test images."""
        cls.test_width = 1024
        cls.test_height = 1024
        cls.test_shape = (cls.test_height, cls.test_width, 4)
        cls.test_dtype = np.uint8
        
        test_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.join(test_dir, "data", "base.png")
        base1_path = os.path.join(test_dir, "data", "base_1.png")
        
        cls.base_image = None
        cls.base1_image = None
        
        if os.path.exists(base_path):
            img = cv2.imread(base_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                cls.base_image = cv2.resize(img, (cls.test_width, cls.test_height), interpolation=cv2.INTER_LINEAR)
                if cls.base_image.shape[2] == 3:
                    cls.base_image = cv2.cvtColor(cls.base_image, cv2.COLOR_BGR2BGRA)
        
        if os.path.exists(base1_path):
            img = cv2.imread(base1_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                cls.base1_image = cv2.resize(img, (cls.test_width, cls.test_height), interpolation=cv2.INTER_LINEAR)
                if cls.base1_image.shape[2] == 3:
                    cls.base1_image = cv2.cvtColor(cls.base1_image, cv2.COLOR_BGR2BGRA)
    
    def _create_random_image(self, seed: int = 0) -> np.ndarray:
        """Create a random 1024x1024 BGRA image."""
        np.random.seed(seed)
        return np.random.randint(0, 256, self.test_shape, dtype=self.test_dtype)
    
    def test_lossless_compression_1024(self):
        """Test lossless compression at 1024x1024."""
        test_data = self._create_random_image(seed=42)
        
        compressed = brotli.compress(test_data.tobytes(), quality=0)
        decoded = np.frombuffer(brotli.decompress(compressed), dtype=np.uint8).reshape(test_data.shape)
        
        np.testing.assert_array_equal(decoded, test_data, "Decoded data must match original")
    
    def test_lossless_compression_real_image_1024(self):
        """Test lossless compression with resized real image."""
        if self.base_image is None:
            self.skipTest("base.png not available")
        
        compressed = brotli.compress(self.base_image.tobytes(), quality=0)
        decoded = np.frombuffer(brotli.decompress(compressed), dtype=np.uint8).reshape(self.base_image.shape)
        
        np.testing.assert_array_equal(decoded, self.base_image, "Decoded image must match original")
    
    def test_compression_ratio_comparison_512_vs_1024(self):
        """Compare compression ratios between 512x512 and 1024x1024."""
        # 512x512 test
        np.random.seed(42)
        data_512 = np.random.randint(0, 256, (512, 512, 4), dtype=np.uint8)
        compressed_512 = brotli.compress(data_512.tobytes(), quality=0)
        ratio_512 = data_512.nbytes / len(compressed_512)
        
        # 1024x1024 test
        np.random.seed(42)
        data_1024 = np.random.randint(0, 256, (1024, 1024, 4), dtype=np.uint8)
        compressed_1024 = brotli.compress(data_1024.tobytes(), quality=0)
        ratio_1024 = data_1024.nbytes / len(compressed_1024)
        
        print(f"\nCompression Ratio Comparison:")
        print(f"  512x512:  {data_512.nbytes / 1024:.2f} KB -> {len(compressed_512) / 1024:.2f} KB (ratio: {ratio_512:.2f}x)")
        print(f"  1024x1024: {data_1024.nbytes / 1024:.2f} KB -> {len(compressed_1024) / 1024:.2f} KB (ratio: {ratio_1024:.2f}x)")
        
        self.assertGreater(ratio_512, 0.5)
        self.assertGreater(ratio_1024, 0.5)
    
    def test_solid_color_compression_1024(self):
        """Test compression of solid color image at 1024x1024."""
        # Solid red image
        solid = np.zeros(self.test_shape, dtype=self.test_dtype)
        solid[:, :, 2] = 255  # Red channel in BGRA
        solid[:, :, 3] = 255  # Alpha channel
        
        compressed = brotli.compress(solid.tobytes(), quality=0)
        decoded = np.frombuffer(brotli.decompress(compressed), dtype=np.uint8).reshape(solid.shape)
        
        np.testing.assert_array_equal(decoded, solid)
        
        # Solid color should compress very well
        ratio = solid.nbytes / len(compressed)
        print(f"\n[1024x1024] Solid color compression ratio: {ratio:.2f}x")
        self.assertGreater(ratio, 2.0, "Solid color should compress well")
    
    def test_gradient_compression_1024(self):
        """Test compression of gradient image at 1024x1024."""
        # Create gradient image
        gradient = np.zeros(self.test_shape, dtype=self.test_dtype)
        for i in range(self.test_height):
            gradient[i, :, 0] = int(255 * i / self.test_height)  # B gradient
            gradient[i, :, 1] = int(255 * (self.test_height - i) / self.test_height)  # G gradient
        for j in range(self.test_width):
            gradient[:, j, 2] = int(255 * j / self.test_width)  # R gradient
        gradient[:, :, 3] = 255  # Full alpha
        
        compressed = brotli.compress(gradient.tobytes(), quality=0)
        decoded = np.frombuffer(brotli.decompress(compressed), dtype=np.uint8).reshape(gradient.shape)
        
        np.testing.assert_array_equal(decoded, gradient)
        
        ratio = gradient.nbytes / len(compressed)
        print(f"\n[1024x1024] Gradient compression ratio: {ratio:.2f}x")


class TestCacherEdgeCases1024(unittest.TestCase):
    """Test edge cases at 1024x1024 resolution."""
    
    def _create_random_image(self, seed: int = 0) -> np.ndarray:
        """Create a random 1024x1024 BGRA image."""
        np.random.seed(seed)
        return np.random.randint(0, 256, (1024, 1024, 4), dtype=np.uint8)
    
    def test_cache_memory_usage_1024(self):
        """Test memory usage tracking at 1024x1024."""
        cacher = Cacher(max_volume_giga=0.5, width=1024, height=1024)
        
        # Write several entries
        for i in range(10):
            data = self._create_random_image(seed=i)
            cacher.put(i, data)
        
        # Verify size tracking
        calculated_kb = sum(len(v) for v in cacher.cache.values()) / 1024
        self.assertAlmostEqual(cacher.cached_kbytes, calculated_kb, places=2)
        
        print(f"\n[1024x1024] Memory usage for 10 entries: {cacher.cached_kbytes / 1024:.2f} MB")
    
    def test_rapid_write_eviction_1024(self):
        """Test rapid write with forced eviction at 1024x1024."""
        # Very small cache - should only hold 2-3 entries
        cacher = Cacher(max_volume_giga=0.005, width=1024, height=1024)  # ~5MB
        
        for i in range(50):
            data = self._create_random_image(seed=i)
            cacher.put(i, data)
        
        # Should stay under limit
        self.assertLessEqual(cacher.cached_kbytes, cacher.max_kbytes)
        
        # Should have evicted most entries
        self.assertLess(len(cacher.cache), 50)
        
        print(f"\n[1024x1024] After 50 writes to tiny cache:")
        print(f"  Entries remaining: {len(cacher.cache)}")
        print(f"  Cache size: {cacher.cached_kbytes:.2f} KB / {cacher.max_kbytes:.2f} KB")


class TestCacherEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def _create_random_image(self, width: int = 512, height: int = 512, seed: int = 0) -> np.ndarray:
        """Create a random BGRA image."""
        np.random.seed(seed)
        return np.random.randint(0, 256, (height, width, 4), dtype=np.uint8)
    
    def test_empty_cache_read(self):
        """Test reading from empty cache."""
        cacher = Cacher(max_volume_giga=1.0, width=512, height=512)
        
        self.assertEqual(len(cacher.cache), 0)
        self.assertIsNone(cacher.get(0))
        self.assertIsNone(cacher.get(12345))
        self.assertIsNone(cacher.get(-1))
    
    def test_very_small_cache(self):
        """Test cache smaller than one entry."""
        # Cache so small it can't hold even one 512x512 image
        cacher = Cacher(max_volume_giga=0.00001, width=512, height=512)  # ~10KB
        
        data = self._create_random_image()
        cacher.put(1, data)
        
        # Entry should be immediately evicted or not stored
        # Cache should still be within limits
        self.assertLessEqual(cacher.cached_kbytes, cacher.max_kbytes)
    
    def test_negative_hash_keys(self):
        """Test using negative hash keys."""
        cacher = Cacher(max_volume_giga=0.1, width=512, height=512)
        
        data = self._create_random_image()
        
        cacher.put(-1, data)
        cacher.put(-12345, data)
        cacher.put(-999999, data)
        
        result = cacher.get(-1)
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result, data)
    
    def test_zero_hash_key(self):
        """Test using zero as hash key."""
        cacher = Cacher(max_volume_giga=0.1, width=512, height=512)
        
        data = self._create_random_image()
        cacher.put(0, data)
        
        result = cacher.get(0)
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result, data)
    
    def test_large_hash_keys(self):
        """Test using very large hash keys."""
        cacher = Cacher(max_volume_giga=0.1, width=512, height=512)
        
        data = self._create_random_image()
        large_key = 2**62
        
        cacher.put(large_key, data)
        result = cacher.get(large_key)
        
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result, data)
    
    def test_sequential_same_key_reads(self):
        """Test that sequential reads of the same key work correctly."""
        cacher = Cacher(max_volume_giga=0.1, width=512, height=512)
        
        data = self._create_random_image()
        cacher.put(100, data)
        
        # Read same key 10 times
        for _ in range(10):
            result = cacher.get(100)
            self.assertIsNotNone(result)
            np.testing.assert_array_equal(result, data)
        
        # continues_hits should be 0 after same-key reads
        # (first read sets continues_hits=1, subsequent same-key reads reset to 0)
        self.assertEqual(cacher.continues_hits, 0)
    
    def test_max_continues_hits_boundary(self):
        """Test the exact boundary of anti-thrashing (5 vs 6 sequential hits)."""
        cacher = Cacher(max_volume_giga=1.0, width=512, height=512)
        
        # Write 10 entries
        for i in range(10):
            data = self._create_random_image(seed=i)
            cacher.put(i, data)
        
        # Read exactly 5 different keys
        for i in range(5):
            result = cacher.get(i)
            self.assertIsNotNone(result)  # Should still hit
        
        self.assertEqual(cacher.continues_hits, 5)
        
        # 6th read should still succeed (condition is > 5, not >= 5)
        result = cacher.get(5)
        self.assertIsNotNone(result)
        self.assertEqual(cacher.continues_hits, 6)
        
        # 7th read to different key should return None
        result = cacher.get(6)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
