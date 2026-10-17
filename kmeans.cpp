#include "kmeans.h"

#include <cuda_runtime.h>
#include <vector>
#include <cmath>

std::vector<std::vector<float>> k_means_clustering(
    const std::vector<std::vector<float>>& points,
    int k,
    const std::vector<std::vector<float>>& initial_centroids,
    int max_iterations
) {
    int dim = points[0].size();
    int num_clusters = initial_centroids.size();
    int dataset_size = points.size();

    std::vector<float> flattened_points;
    flattened_points.reserve(dataset_size * dim);
    for (const auto& v : points) {
        flattened_points.insert(flattened_points.end(), v.begin(), v.end());
    }

    std::vector<float> flattened_centroids;
    flattened_centroids.reserve(num_clusters * dim);
    for (const auto& v : initial_centroids) {
        flattened_centroids.insert(flattened_centroids.end(), v.begin(), v.end());
    }

    float *d_points, *d_centroids, *d_partial_centroids;
    int *d_point_affiliation;

    cudaMalloc(&d_points,            dataset_size * dim * sizeof(float));
    cudaMalloc(&d_centroids,         num_clusters * dim * sizeof(float));
    cudaMalloc(&d_point_affiliation, dataset_size * sizeof(int));

    cudaMemcpy(d_points,    flattened_points.data(),    dataset_size * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, flattened_centroids.data(), num_clusters * dim * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (dataset_size + threads - 1) / threads;

    int vector_dim = num_clusters * (dim + 1);   // one block's partial
    size_t partial_bytes = (size_t)blocks * vector_dim * sizeof(float);
    cudaMalloc(&d_partial_centroids, partial_bytes);

    int sum_per_thread = 2;
    int update_blocks = (num_clusters * dim + threads - 1) / threads;

    for (int epoch = 0; epoch < max_iterations; ++epoch) {
        cudaMemset(d_partial_centroids, 0, partial_bytes);

        launch_find_nearest(
            blocks, threads,
            d_points, d_point_affiliation, d_centroids,
            dim, num_clusters, dataset_size
        );

        launch_eval_centroids(
            blocks, threads,
            d_points, d_point_affiliation, d_partial_centroids,
            dim, num_clusters, dataset_size
        );

        int vector_step = sum_per_thread;
        while ((vector_step / sum_per_thread) < blocks) {
            int active_vectors = (blocks + vector_step - 1) / vector_step;
            int reduce_blocks  = (active_vectors + threads - 1) / threads;

            launch_reduce_all(
                reduce_blocks, threads,
                d_partial_centroids, vector_dim, blocks,
                vector_step, sum_per_thread
            );

            vector_step *= sum_per_thread;
        }

        launch_update_centroids(
            update_blocks, threads,
            d_centroids, d_partial_centroids, dim, num_clusters
        );
    }

    cudaMemcpy(flattened_centroids.data(), d_centroids,
               num_clusters * dim * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_point_affiliation);
    cudaFree(d_partial_centroids);

    std::vector<std::vector<float>> result(num_clusters, std::vector<float>(dim));
    for (int c = 0; c < num_clusters; ++c) {
        for (int d = 0; d < dim; ++d) {
            float v = flattened_centroids[c * dim + d];
            result[c][d] = std::round(v * 10000.0f) / 10000.0f;
        }
    }

    return result;
}
