from math import ceil

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


def kmeans(obs: torch.Tensor, k: int,
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


def product_quantization(data, sub_vector_size, k, **kwargs):
    centers = []
    for i in range(0, data.shape[1], sub_vector_size):
        sub_data = data[:, i:i + sub_vector_size]
        sub_centers, _ = kmeans(sub_data, k=k, **kwargs)
        centers.append(sub_centers)
    return centers


def data_to_pq(data, centers):
    assert (len(centers) > 0)
    assert (data.shape[1] == sum([cb.shape[1] for cb in centers]))

    m = len(centers)
    sub_size = centers[0].shape[1]
    ret = torch.zeros(data.shape[0], m,
                      dtype=torch.uint8,
                      device=data.device)
    for idx, sub_vec in enumerate(torch.split(data, sub_size, dim=1)):
        dis = l2_distance(sub_vec, centers[idx])
        ret[:, idx] = dis.argmin(dim=1).to(dtype=torch.uint8)
    return ret


def train_product_quantization(data, sub_vector_size, k, **kwargs):
    center_list = product_quantization(data, sub_vector_size, k, **kwargs)
    pq_data = data_to_pq(data, center_list)
    return pq_data, center_list


def _gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w * h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G


def pq_distance_book(pq_centers):
    assert (len(pq_centers) > 0)

    pq = torch.zeros(len(pq_centers),
                     len(pq_centers[0]),
                     len(pq_centers[0]),
                     device=pq_centers[0].device)
    for ci, center in enumerate(pq_centers):
        for i in range(len(center)):
            dis = l2_distance(center[i:i + 1, :], center)
            pq[ci, i] = dis
    return pq


def asymmetric_table(query, centers):
    m = len(centers)
    sub_size = centers[0].shape[1]
    ret = torch.zeros(
        query.shape[0], m, centers[0].shape[0],
        device=query.device)
    assert (query.shape[1] == sum([cb.shape[1] for cb in centers]))
    for i, offset in enumerate(range(0, query.shape[1], sub_size)):
        sub_query = query[:, offset: offset + sub_size]
        ret[:, i, :] = l2_distance(sub_query, centers[i])
    return ret


def asymmetric_distance_slow(asymmetric_tab, pq_data):
    ret = torch.zeros(asymmetric_tab.shape[0], pq_data.shape[0])
    for i in range(asymmetric_tab.shape[0]):
        for j in range(pq_data.shape[0]):
            dis = 0
            for k in range(pq_data.shape[1]):
                sub_dis = asymmetric_tab[i, k, pq_data[j, k].item()]
                dis += sub_dis
            ret[i, j] = dis
    return ret


def asymmetric_distance(asymmetric_tab, pq_data):
    pq_db = pq_data.long()
    dd = [torch.index_select(asymmetric_tab[:, i, :], 1, pq_db[:, i]) for i in range(pq_data.shape[1])]
    return sum(dd)


def pq_distance(obj, centers, pq_disbook):
    ret = torch.zeros(obj.shape[0], centers.shape[0])
    for obj_idx, o in enumerate(obj):
        for ct_idx, c in enumerate(centers):
            for i, (oi, ci) in enumerate(zip(o, c)):
                ret[obj_idx, ct_idx] += pq_disbook[i, oi.item(), ci.item()]
    return ret
