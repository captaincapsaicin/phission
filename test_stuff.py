import unittest


class TestNucNormMin(unittest.TestCase):

    def test_min(self):
        m = [[-1, 1],
             [0, 1]]
        # we expect rank to be minimized
        m_complete_expected = [[-1, 1],
                               [-1, 1]]
        m_complete = complete(m)
        self.assertEqual(m_complete, m_complete_expected)
