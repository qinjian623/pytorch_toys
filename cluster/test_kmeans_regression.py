import torch

from cluster.kmeans import kmeans, l2_distance


def test_kmeans_handles_single_element_cluster():
    obs = torch.tensor([
        [0.0, 0.0],
        [10.0, 10.0],
        [11.0, 11.0],
    ])

    torch.manual_seed(0)
    centers, _ = kmeans(obs, k=2, iter=1)

    assignments = l2_distance(obs, centers).argmin(dim=1)
    counts = torch.bincount(assignments, minlength=2)

    assert (counts == 1).any()
