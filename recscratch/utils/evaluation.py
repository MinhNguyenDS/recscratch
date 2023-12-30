import numpy as np

def mean_reciprocal_rank(rs):
    rs = (np.atleast_1d(np.asarray(r)).nonzero()[0] for r in rs)
    rs = np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])
    rs = np.array(rs)
    return rs

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0
    return dcg_at_k(r, k) / dcg_max

if __name__ == "__main__":
   ## Test function ##
    y_true = [0., 1., 2.]
    y_pred = [3., 0., 2.]

    relevance = []
    for i in range(len(y_true)):
      if y_true[i] == y_pred[i]:
        relevance.append(1)
      else: relevance.append(0)

    print ("MRR", mean_reciprocal_rank(relevance))
    print ("DCG@10", dcg_at_k(relevance, 10))
    print ("nDCG@10", ndcg_at_k(relevance, 10))

