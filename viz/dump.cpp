#include "kmeans.h"

#include <cstdio>
#include <iostream>
#include <vector>

int main() {
    int n, k, iterations;
    std::cin >> n >> k >> iterations;

    std::vector<std::vector<float>> points(n, std::vector<float>(2));
    for (auto& p : points) std::cin >> p[0] >> p[1];

    std::vector<std::vector<float>> initial_centroids(k, std::vector<float>(2));
    for (auto& c : initial_centroids) std::cin >> c[0] >> c[1];

    for (int i = 0; i <= iterations; ++i) {
        auto centroids = k_means_clustering(points, k, initial_centroids, i);
        for (const auto& c : centroids) {
            std::printf("%d %f %f\n", i, c[0], c[1]);
        }
    }

    return 0;
}
