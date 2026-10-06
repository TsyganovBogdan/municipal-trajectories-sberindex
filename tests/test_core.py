"""Meaningful checks of edge cases, metric conventions and time handling."""
import unittest
import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score
from src.metrics import graph_indices, s_dbw
from src.pipeline import knn_graph, align_labels, build_features


class CoreTests(unittest.TestCase):
    def test_disconnected_cliques(self):
        a = np.zeros((6, 6))
        a[:3, :3] = 1
        a[3:, 3:] = 1
        np.fill_diagonal(a, 0)
        m = graph_indices(sparse.csr_matrix(a), np.array([0, 0, 0, 1, 1, 1]))
        self.assertAlmostEqual(m['AVI'], 1)
        self.assertAlmostEqual(m['AVU'], 0)
        self.assertAlmostEqual(m['MQ'], .5)

    def test_complete_graph_partition(self):
        a = np.ones((4, 4)) - np.eye(4)
        m = graph_indices(sparse.csr_matrix(a), np.array([0, 0, 1, 1]))
        self.assertAlmostEqual(m['AVI'], 1 / 3)
        self.assertAlmostEqual(m['AVU'], 1)
        self.assertAlmostEqual(m['MQ'], -1 / 6)

    def test_weight_scale_invariance(self):
        rng = np.random.default_rng(4)
        a = rng.uniform(size=(8, 8)); a = (a + a.T) / 2
        np.fill_diagonal(a, 0)
        labels = np.repeat([0, 1], 4)
        first = graph_indices(sparse.csr_matrix(a), labels)
        second = graph_indices(sparse.csr_matrix(11 * a), labels)
        for k in first:
            self.assertAlmostEqual(first[k], second[k])

    def test_knn_symmetric_no_self_loops(self):
        x = np.random.default_rng(3).normal(size=(20, 4))
        a = knn_graph(cdist(x, x), 4)
        self.assertEqual((a != a.T).nnz, 0)
        self.assertEqual(np.count_nonzero(a.diagonal()), 0)
        self.assertTrue(np.all(a.data > 0))
        self.assertTrue(np.all(np.diff(a.indptr) >= 4))

    def test_missing_road_distances_not_edges(self):
        d = np.array([[0, 1, np.inf], [1, 0, np.inf], [np.inf, np.inf, 0]])
        a = knn_graph(d, 2)
        self.assertEqual(a[2].nnz, 0)

    def test_label_permutation_is_not_a_transition(self):
        a = np.array([0, 0, 1, 1, 2, 2])
        b = np.array([7, 7, 9, 9, 3, 3])
        np.testing.assert_array_equal(a, align_labels(a, b))

    def test_split_remains_visible(self):
        a = np.array([0, 0, 0, 0, 1, 1])
        b = np.array([4, 4, 5, 5, 6, 6])
        aligned = align_labels(a, b)
        self.assertEqual(len(np.unique(aligned)), 3)
        self.assertLess(adjusted_rand_score(a, aligned), 1)

    def test_density_index_distinguishes_mixed_partition(self):
        rng = np.random.default_rng(5)
        x = np.r_[rng.normal(-4, .2, size=(100, 2)), rng.normal(4, .2, size=(100, 2))]
        good = np.repeat([0, 1], 100)
        bad = np.arange(200) % 2
        self.assertLess(s_dbw(x, good), s_dbw(x, bad))

    def test_future_data_cannot_change_past_features(self):
        rng = np.random.default_rng(7)
        levels = rng.uniform(100, 200, size=(24, 30, 6))
        shares = rng.dirichlet(np.ones(6), size=(24, 30))
        months = [f'{y}-{m:02d}' for y in [2023, 2024] for m in range(1, 13)]
        fit = ['2024-01', '2024-03', '2024-06']
        first = build_features(levels, shares, months, fit)[2]
        modified = levels.copy(); modified[-3:] *= 10
        shares2 = shares.copy(); shares2[-3:] = shares2[-3:, :, ::-1]
        second = build_features(modified, shares2, months, fit)[2]
        np.testing.assert_allclose(first[:9], second[:9])


if __name__ == '__main__':
    unittest.main()
