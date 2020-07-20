import torch
import warnings


def cosine_distance(obs, centers):
    obs_norm = obs / obs.norm(dim=1, keepdim=True)
    centers_norm = centers / centers.norm(dim=1, keepdim=True)
    cos = torch.matmul(obs_norm, centers_norm.transpose(1, 0))
    return 1 - cos


def l2_distance(obs, centers):
    dis = ((obs.unsqueeze(dim=1) - centers.unsqueeze(dim=0)) ** 2.0).sum(dim=-1).squeeze()
    return dis


def _kmeans_batch(obs: torch.Tensor,
                  k: int,
                  distance_function,
                  batch_size=0,
                  thresh=1e-5,
                  norm_center=False):
    # k x D
    centers = obs[torch.randperm(obs.size(0))[:k]].clone()
    history_distances = [float('inf')]
    if batch_size == 0:
        batch_size = obs.shape[0]
    while True:
        # (N x D, k x D) -> N x k
        segs = torch.split(obs, batch_size)
        seg_center_dis = []
        seg_center_ids = []
        for seg in segs:
            distances = distance_function(seg, centers)
            center_dis, center_ids = distances.min(dim=1)
            seg_center_ids.append(center_ids)
            seg_center_dis.append(center_dis)

        obs_center_dis_mean = torch.cat(seg_center_dis).mean()
        obs_center_ids = torch.cat(seg_center_ids)
        history_distances.append(obs_center_dis_mean.item())
        diff = history_distances[-2] - history_distances[-1]
        if diff < thresh:
            if diff < 0:
                warnings.warn("Distance diff < 0, distances: " + ", ".join(map(str, history_distances)))
            break
        for i in range(k):
            obs_id_in_cluster_i = obs_center_ids == i
            if obs_id_in_cluster_i.sum() == 0:
                continue
            obs_in_cluster = obs.index_select(0, obs_id_in_cluster_i.nonzero().squeeze())
            c = obs_in_cluster.mean(dim=0)
            if norm_center:
                c /= c.norm()
            centers[i] = c
    return centers, history_distances[-1]


def kmeans(obs: torch.Tensor,
           k: int,
           distance_function=l2_distance,
           iter=20,
           batch_size=0,
           thresh=1e-5,
           norm_center=False):
    """
        Performs k-means on a set of observation vectors forming k clusters.

        Parameters
        ----------
        obs : torch.Tensor
           Each row of the M by N array is an observation vector.

        k : int
           The number of centroids to generate. A code is assigned to
           each centroid, which is also the row index of the centroid
           in the code_book matrix generated.

           The initial k centroids are chosen by randomly selecting
           observations from the observation matrix.

        distance_function : function, optional
           The function to calculate distances between observations and centroids.
           Default value: l2_distance

        iter : int, optional
           The number of times to run k-means, returning the codebook
           with the lowest distortion. This parameter does not represent the
           number of iterations of the k-means algorithm.

        batch_size : int, optional
           Batch size of observations to calculate distances, if your GPU memory can NOT handle all observations.
           Default value is 0, which will send all observations into distance_function.

        thresh : float, optional
           Terminates the k-means algorithm if the change in
           distortion since the last k-means iteration is less than
           or equal to thresh.

        norm_center : False, optional
           Whether to normalize the centroids while updating every centroid.

        Returns
        -------
        best_centers : torch.Tensor
           A k by N array of k centroids. The i'th centroid
           codebook[i] is represented with the code i. The centroids
           and codes generated represent the lowest distortion seen,
           not necessarily the globally minimal distortion.

        best_distance : float
           The mean distance between the observations passed and the best centroids generated.
        """
    best_distance = float("inf")
    best_centers = None
    for i in range(iter):
        if batch_size == 0:
            batch_size == obs.shape[0]
        centers, distance = _kmeans_batch(obs, k,
                                          norm_center=norm_center,
                                          distance_function=distance_function,
                                          batch_size=batch_size,
                                          thresh=thresh)
        if distance < best_distance:
            best_centers = centers
            best_distance = distance
    return best_centers, best_distance
