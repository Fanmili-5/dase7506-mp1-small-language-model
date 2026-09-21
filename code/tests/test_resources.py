import unittest

from scripts.peak_memory import peak_process_memory


class ResourceTests(unittest.TestCase):
    def test_peak_memory_is_available_and_monotonic(self):
        before = peak_process_memory()
        allocation = bytearray(1024 * 1024)
        allocation[::4096] = b"x" * len(allocation[::4096])
        after = peak_process_memory()
        self.assertGreater(before["peak_rss_bytes"], 0)
        self.assertGreaterEqual(after["peak_rss_bytes"], before["peak_rss_bytes"])
        self.assertIn("memory_measurement", after)


if __name__ == "__main__":
    unittest.main()
