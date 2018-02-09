import sys
# for use on morganstanley.cl.cam.ac.uk
sys.path.append("/local/scratch/js2173/pytorch/Selectively-Retexuring-Subimages/submodules/pix2pixHD") # access submodules
sys.path = [p for p in sys.path if not p.startswith('/local/scratch') or p.startswith('/local/scratch/js2173')]

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
import numpy as np
import os

opt = TrainOptions().parse()
opt.nThreads = 1
opt.batchSize = 1 
opt.serial_batches = True 
opt.no_flip = True
opt.instance_feat = True
opt.name = 'label2city_512p_feat'
opt.dataroot = '/local/scratch/js2173/pytorch/Selectively-Retexuring-Subimages/submodules/pix2pixHD/datasets/cityscapes/'
opt.checkpoints_dir = '/local/scratch/js2173/pytorch/Selectively-Retexuring-Subimages/submodules/pix2pixHD/checkpoints/'
opt.results = '/local/scratch/js2173/pytorch/Selectively-Retexuring-Subimages/submodules/pix2pixHD/results/'

name = 'features'
save_path = os.path.join(opt.checkpoints_dir, opt.name)

# O(N^3) furthest-first clustering
# intended to explore outliers/extremes of feature space
def furthest_first_clustering(data, n_clusters, dist=np.linalg.norm):
    assert len(data) >= n_clusters, "Can't create more clusters than there are points"
    initial = np.random.randint(0, data.shape[0], size=(1))[0]
    centers = np.array([data[initial]])
    def min_dist(centers, new_point):
        d = dist(centers[0] - new_point)
        for c in centers[1:]:
            x = dist(c - point)
            if x < d:
                d = x
        return d
    for i in range(n_clusters-1):
        max_dist_found, maximizing_point = -1, -1
        # select another point in data which maximizes minimum distance to all previous centers
        for point in data:
            closest_cluster_dist = min_dist(centers, point)
            if closest_cluster_dist > max_dist_found:
                max_dist_found = closest_cluster_dist
                maximizing_point = point
        centers = np.vstack([centers, maximizing_point])
    return centers



saved_features_name = os.path.join(save_path, name+'.npy')
# dictionary mapping from class ID => numpy ndarray
features = np.load(saved_features_name).item()
n_clusters = opt.n_clusters

centers = {}
for label in range(opt.label_nc):
	feat = features[label]
	feat = feat[feat[:,-1] > 0.5, :-1]		
	if feat.shape[0]:
		n_clusters = min(feat.shape[0], opt.n_clusters)
        points = furthest_first_clustering(feat, n_clusters)
        centers[label] = points
save_name = os.path.join(save_path, name + '_extrema_%03d.npy' % opt.n_clusters)
np.save(save_name, centers)
print('saving to %s' % save_name)