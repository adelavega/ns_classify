d <- read.csv('v6_dist_scaled.csv')
dm <- as.matrix(d)[, 2:101]
library(fastcluster)
clust <- hclust.vector(dm, method='ward')
saveRDS(clust, '/projects/delavega/clustering/results/hierarchical/Rclust_wb.R')
