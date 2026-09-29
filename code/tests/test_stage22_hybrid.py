import unittest
from scripts.screen_stage22_hybrid import WEIGHTS


class Stage22HybridTests(unittest.TestCase):
    def test_fixed_grid_is_ordered_and_contains_prior_optimum(self):
        self.assertEqual(tuple(sorted(set(WEIGHTS))),WEIGHTS)
        self.assertIn(.1,WEIGHTS)
        self.assertEqual(WEIGHTS[0],0)
        self.assertLess(WEIGHTS[-1],1)


if __name__=="__main__": unittest.main()
