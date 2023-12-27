from unittest import TestCase

from mbr import MBRConfig


class MBRConfigTestCase(TestCase):

    def test_default_config(self):
        config = MBRConfig()
        self.assertEqual(config.num_samples, 10)
        self.assertEqual(config.num_references, 10)
