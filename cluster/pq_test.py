import torch
import numpy as np
import time

from cluster.kmeans import product_quantization, l2_distance, asymmetric_table, data_to_pq, asymmetric_distance


def main():
    torch.set_num_threads(16)
    dim = 32
    subv_size = 4
    num_centers = 256
    query_size = 100
    db_size = 10000

    db = np.random.randn(db_size, dim)
    training_data = torch.from_numpy(db)

    if torch.cuda.is_available():
        training_data = training_data.cuda()

    codebook = product_quantization(
        training_data,
        subv_size,
        k=num_centers, iter=2,
        batch_size=4096*10)

    centers = torch.stack(codebook).cpu().numpy()
    print(centers.shape)

    if torch.cuda.is_available():
        codebook = [i.cuda() for i in codebook]

    pq_data = data_to_pq(training_data, codebook)

    query = torch.randn(query_size, dim)
    if torch.cuda.is_available():
        query = query.cuda()

    db = training_data[:db_size]
    pq_db = pq_data[:db_size]
    gts = torch.cat(
        [l2_distance(batch, db).argmin(dim=1) for batch in torch.split(query, 64)])

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    s = time.time()

    tb = asymmetric_table(query, codebook)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    e = time.time()
    print("Precomputing: {}".format(e - s))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    s = time.time()

    distances = []
    for batch in torch.split(pq_db, 2048):
        distances.append(asymmetric_distance(tb, batch))
    distances = torch.cat(distances, dim=1)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    e = time.time()
    print("Distancing: {}".format(e - s))

    tk = 32
    _, ids = distances.topk(tk, dim=1, largest=False)

    hit = 0
    for gt, id in zip(gts, ids):
        if gt in id:
            hit += 1
    import math
    print("Recall: {} [M={}, bits={} @ top {}]".format(hit / query_size, dim//subv_size, math.log2(num_centers), tk))


if __name__ == '__main__':
    main()
