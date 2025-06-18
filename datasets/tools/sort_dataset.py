import os

import numpy as np
import trimesh

# in_dir = 'datasets/laz_minimal/bins/ca_13/unsorted'
in_dir = 'datasets/laz_minimal/bins/swisssurface3d/unsorted'
# out_dir = 'datasets/laz_minimal/bins/ca_13'
out_dir = 'datasets/laz_minimal/bins/swisssurface3d'

num_query_pts = 10_000
res = 64

# find all bins in in_dir
bins = [f for f in os.listdir(in_dir) if f.endswith('.bin')]

for bin in bins:
    # read
    dt = np.dtype('3f8, (64,64)f4')
    # dt = np.dtype([('f0', np.float64, (num_query_pts, 3)), ('f1', np.float32, (num_query_pts, res, res))])
    hm_data = np.fromfile(file=os.path.join(in_dir, bin), dtype=dt)

    query_pts = hm_data['f0']
    hm = hm_data['f1']

    # get 70 percentile of query_pts x
    x = query_pts[:, 0]
    x_70 = np.percentile(x, 70)

    # get ids of lower part in beginning, rest in end
    ids = np.arange(num_query_pts)
    lower_ids = ids[x < x_70]
    upper_ids = ids[x >= x_70]

    # re-assemble query_pts and hm
    query_pts_new = np.concatenate([query_pts[upper_ids], query_pts[lower_ids]])
    hm_new = np.concatenate([hm[upper_ids], hm[lower_ids]])

    # write query_pts to ply for checking
    # trimesh.Trimesh(vertices=query_pts_new).export(bin.replace('.bin', '_query_pts_check.ply'))

    # write to bin
    new_bin = os.path.join(out_dir, bin)
    # hm_data_new = np.array([query_pts_new, hm_new], dtype=dt)
    hm_data_new = np.empty(num_query_pts, dtype=dt)
    hm_data_new['f0'] = query_pts_new
    hm_data_new['f1'] = hm_new
    hm_data_new.tofile(new_bin)

    # read new bin to check
    # hm_data_new_check = np.fromfile(file=new_bin, dtype=dt)
    # print(hm_data_new_check['f0'].shape)
    # print(hm_data_new_check['f1'].shape)
    # print(hm_data_new_check['f0'])
    # diff_f0 = hm_data_new_check['f0'] == query_pts_new
    # diff_f1 = hm_data_new_check['f1'] == hm_new
    # pass
