#pragma once

#include <vector>

void launch_find_nearest(
    int blocks,
    int threads,
    const float* points,
    int* point_affiliation,
    const float* centroids,
    int dim,
    int num_clusters,
    int dataset_size
);

void launch_eval_centroids(
    int blocks,
    int threads,
    const float* points,
    const int* point_affiliation,
    float* partial_centroids,
    int dim,
    int num_clusters,
    int dataset_size
);

void launch_reduce_all(
    int blocks,
    int threads,
    float* data,
    int vector_dim,
    int total_vectors,
    int vector_step,
    int sum_per_thread
);

void launch_update_centroids(
    int blocks,
    int threads,
    float* centroids,
    const float* sums,
    int dim,
    int num_clusters
);

std::vector<std::vector<float>> k_means_clustering(
    const std::vector<std::vector<float>>& points,
    int k,
    const std::vector<std::vector<float>>& initial_centroids,
    int max_iterations
);
