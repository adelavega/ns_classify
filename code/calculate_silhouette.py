from sklearn.metrics import silhouette_score
import joblib
from scipy.cluster.hierarchy import fcluster

distances = joblib.load('/projects/delavega/clustering/results/bootstrap/whole_brain_PCA_dist_min100_full.kl')
scaled_distances = joblib.load('/projects/delavega/clustering/results/bootstrap/whole_brain_PCA_dist_min100_scaled_full.pkl')

regions = [20, 40, 60, 80, 100, 120, 140, 160, 180]
ss = 20000


# kmeans_labels = joblib.load('/projects/delavega/clustering/results/bootstrap/kmeans/labels_wholebrain_raw_full.pkl') +  joblib.load(
#     '/projects/delavega/clustering/results/bootstrap/kmeans/labels_wholebrain_raw_full_100_180.pkl')
# kmeans_scaled_labels = joblib.load('/projects/delavega/clustering/results/bootstrap/kmeans/labels_wholebrain_scaled_full.pkl') + joblib.load(
#     '/projects/delavega/clustering/results/bootstrap/kmeans/labels_wholebrain_scaled_full_100_180.pkl')

# silhouette_k = [['kmeans', 'original', n_reg, silhouette_score(distances, labels, sample_size=ss)] for n_reg, labels in kmeans_labels]

# silhouette_k_scaled = [['kmeans', 'scaled', n_reg, silhouette_score(scaled_distances, labels, sample_size=ss)] for n_reg, labels in kmeans_scaled_labels]
# joblib.dump(silhouette_k_scaled + silhouette_k, '/projects/delavega/clustering/results/bootstrap/kmeans_silhouette.pkl')


Z_raw = joblib.load('/projects/delavega/clustering/results/bootstrap/hierarchical/Z_ward_wholebrain_full.pkl')
ward_labels = [[region, fcluster(Z_raw, region, 'maxclust')] for region in regions]

Z_scaled = joblib.load('/projects/delavega/clustering/results/bootstrap/hierarchical/Z_ward_wholebrain_scaled_full.pkl')
ward_scaled_labels = [[region, fcluster(Z_scaled, region, 'maxclust')] for region in regions]

silhouette_w = [['ward', 'original', n_reg, silhouette_score(distances, labels, sample_size=ss)] for n_reg, labels in ward_labels]

silhouette_w_scaled = [['ward', 'scaled', n_reg, silhouette_score(scaled_distances, labels, sample_size=ss)] for n_reg, labels in ward_scaled_labels]

joblib.dump(silhouette_w + silhouette_w_scaled, '/projects/delavega/clustering/results/bootstrap/ward_silhouette.pkl')
