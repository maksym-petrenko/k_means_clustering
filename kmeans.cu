#include "kmeans.h"

#include <cuda_runtime.h>
#include <cfloat>

__global__ void find_nearest(
    const float* points,
    int* point_affiliation,
    const float* centroids,
    int dim,
    int num_clusters,
    int dataset_size
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= dataset_size) return;

    int cluster = 0;
    float best = FLT_MAX;

    for (int c = 0; c < num_clusters; ++c) {
        float d_sqr = 0.0f;
        for (int d = 0; d < dim; ++d) {
            float diff = points[index * dim + d] - centroids[c * dim + d];
            d_sqr += diff * diff;
        }
        if (d_sqr < best) {
            best = d_sqr;
            cluster = c;
        }
    }

    point_affiliation[index] = cluster;
}

__global__ void eval_centroids(
    const float* points,
    const int* point_affiliation,
    float* partial_centroids,
    int dim,
    int num_clusters,
    int dataset_size
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= dataset_size) return;

    int stride = dim + 1;
    int cluster = point_affiliation[index];
    float* slot = partial_centroids + (blockIdx.x * num_clusters + cluster) * stride;

    for (int d = 0; d < dim; ++d) {
        atomicAdd(&slot[d], points[index * dim + d]);
    }
    atomicAdd(&slot[dim], 1.0f);
}

__global__ void reduce_all(
    float* data,
    int vector_dim,
    int total_vectors,
    int vector_step,
    int sum_per_thread
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int target_vector = index * vector_step;

    if (target_vector >= total_vectors) return;

    int source_step = vector_step / sum_per_thread;

    for (int entry = 0; entry < vector_dim; ++entry) {
        int local_idx = target_vector * vector_dim + entry;

        for (int vect = 1; vect < sum_per_thread; ++vect) {
            int source_vector = target_vector + vect * source_step;

            if (source_vector < total_vectors) {
                data[local_idx] += data[source_vector * vector_dim + entry];
            }
        }
    }
}

__global__ void update_centroids(
    float* centroids,
    const float* sums,
    int dim,
    int num_clusters
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= num_clusters * dim) return;

    int c = index / dim;
    int d = index % dim;
    int stride = dim + 1;

    float count = sums[c * stride + dim];
    if (count > 0.0f) {
        centroids[index] = sums[c * stride + d] / count;
    }
}

void launch_find_nearest(
    int blocks,
    int threads,
    const float* points,
    int* point_affiliation,
    const float* centroids,
    int dim,
    int num_clusters,
    int dataset_size
) {
    find_nearest<<<blocks, threads>>>(
        points, point_affiliation, centroids,
        dim, num_clusters, dataset_size
    );
}

void launch_eval_centroids(
    int blocks,
    int threads,
    const float* points,
    const int* point_affiliation,
    float* partial_centroids,
    int dim,
    int num_clusters,
    int dataset_size
) {
    eval_centroids<<<blocks, threads>>>(
        points, point_affiliation, partial_centroids,
        dim, num_clusters, dataset_size
    );
}

void launch_reduce_all(
    int blocks,
    int threads,
    float* data,
    int vector_dim,
    int total_vectors,
    int vector_step,
    int sum_per_thread
) {
    reduce_all<<<blocks, threads>>>(
        data, vector_dim, total_vectors,
        vector_step, sum_per_thread
    );
}

void launch_update_centroids(
    int blocks,
    int threads,
    float* centroids,
    const float* sums,
    int dim,
    int num_clusters
) {
    update_centroids<<<blocks, threads>>>(
        centroids, sums, dim, num_clusters
    );
}
