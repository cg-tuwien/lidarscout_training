import typing

import numpy as np
import trimesh
from torch.jit import ignore
from scipy.spatial import KDTree as ScipyKDTree
from pykdtree.kdtree import KDTree as PyKDTree


def make_kdtree(pts: np.ndarray, lib: typing.Literal['scipy', 'pykdtree'] = 'pykdtree'):

    # old reliable
    def _make_kdtree_scipy(pts_np: np.ndarray):
        # otherwise KDTree construction may run out of recursions
        import sys
        leaf_size = 1000
        sys.setrecursionlimit(int(max(1000, round(pts_np.shape[0] / leaf_size))))
        _kdtree = ScipyKDTree(pts_np, leaf_size)
        return _kdtree

    # a lot slower than scipy
    # def _make_kdtree_sklearn(pts: np.ndarray):
    #     from sklearn.neighbors import NearestNeighbors
    #     nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=workers).fit(pts)
    #     # indices_batch: np.ndarray = nbrs.kneighbors(pts_query_np, return_distance=False)

    # fastest even without multiprocessing
    def _make_kdtree_pykdtree(pts_np: np.ndarray):
        _kdtree = PyKDTree(pts_np, leafsize=10)
        return _kdtree

    if lib == 'scipy':
        kdtree = _make_kdtree_scipy(pts)
    elif lib == 'pykdtree':
        kdtree = _make_kdtree_pykdtree(pts)
    else:
        raise NotImplementedError('Unknown kdtree lib: {}'.format(lib))
    return kdtree


def query_kdtree(kdtree: typing.Union[ScipyKDTree, PyKDTree],
                 pts_query: np.ndarray, k: int, sqr_dists=False, **kwargs):
    # sqr_dists: some speed-up if True but distorted distances

    if isinstance(kdtree, ScipyKDTree):
        kdtree = typing.cast(ScipyKDTree, kdtree)
        nn_dists, nn_ids = kdtree.query(x=pts_query, k=k, workers=kwargs.get('workers', -1))
        if not sqr_dists:
            nn_dists = nn_dists ** 2
    elif isinstance(kdtree, PyKDTree):
        kdtree = typing.cast(PyKDTree, kdtree)
        nn_dists, nn_ids = kdtree.query(pts_query, k=k, sqr_dists=sqr_dists)
    else:
        raise NotImplementedError('Unknown kdtree type: {}'.format(type(kdtree)))
    return nn_dists, nn_ids


def query_ball_kdtree(kdtree: typing.Union[ScipyKDTree, PyKDTree],
                      pts_query: np.ndarray, r: float, **kwargs):
    # sqr_dists: some speed-up if True but distorted distances

    if isinstance(kdtree, ScipyKDTree):
        kdtree = typing.cast(ScipyKDTree, kdtree)
        nn_ids = kdtree.query_ball_point(
            x=pts_query, r=r, p=kwargs.get('p', 2),
            workers=kwargs.get('workers', -1), return_sorted=kwargs.get('return_sorted', True))
    elif isinstance(kdtree, PyKDTree):
        raise NotImplementedError('PyKDTree does not support ball query')
    else:
        raise NotImplementedError('Unknown kdtree type: {}'.format(type(kdtree)))
    return nn_ids


def query_ball_kdtree_batched(
        kdtree: typing.Union[ScipyKDTree, PyKDTree],
        pts_query: np.ndarray, r: float, batch_size: int = 500, **kwargs) -> typing.List[np.ndarray]:
    """
    Same result as query_ball_kdtree, but issues the query in batches instead of one all-at-once
    call. A single query_ball_point() call covering many query points against a dense tree can
    itself be a large memory bottleneck inside scipy -- that memory is spent before any Python-
    side postprocessing runs, so batching *how the result is consumed* doesn't help; only
    batching the query call itself does. This hit a real crash (~46-49GB) on a 35M-point/10k-
    query shape (see source/dataloaders/local_points_cache.py's build_local_point_index_cache
    docstring for the original diagnosis) and, unbatched, the exact same class of crash recurred
    in predict/reconstruction mode (source/dataloaders/ipes_data_loader.py), which builds its
    query points on the fly and so can't go through that module's precomputed-cache path at all.
    Bounds peak memory to roughly one batch's worth of query results, independent of the total
    number of local-point hits across all query points.
    """
    n = pts_query.shape[0]
    result: typing.List[np.ndarray] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        id_lists = query_ball_kdtree(kdtree=kdtree, pts_query=pts_query[start:end], r=r, **kwargs)
        result.extend(np.asarray(ids) for ids in id_lists)
        del id_lists
    return result


@ignore  # can't compile kdtree
def kdtree_query_oneshot(pts: np.ndarray, pts_query: np.ndarray, k: int, sqr_dists=False, **kwargs):
    # sqr_dists: True: some speed-up but distorted distances
    kdtree = make_kdtree(pts)
    nn_dists, nn_ids = query_kdtree(kdtree=kdtree, pts_query=pts_query, k=k, sqr_dists=sqr_dists, **kwargs)
    return nn_dists, nn_ids
