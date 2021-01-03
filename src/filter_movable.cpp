#include <iostream>
#include "estimation.hpp"
#include "utils.hpp"
#include "pointmatcher/PointMatcher.h"

int main(int argc, const char *argv[]) {
    std::string root, config;
    if (validateArgs(argc, argv, root, config) != 0) {
        return 1;
    }
    std::cout << "Loading map..." << std::endl;
    DP map = DP::load(root + "map/map.ply");
    std::cout << "Finished loading map" << std::endl;

    std::cout << "Filtering on movable probability..." << std::endl;
    uint movable_row = map.getDescriptorStartingRow("movable");
    uint feat_dim = map.features.rows();
    uint desc_dim = map.descriptors.rows();
    uint j = 0;
    for (uint i = 0; i < map.features.cols(); ++i) {
        if (map.descriptors(movable_row, i) < 0.75) {
            map.features.block(0, j, feat_dim, 1) = map.features.block(0, i, feat_dim, 1);
            map.descriptors.block(0, j, desc_dim, 1) = map.descriptors.block(0, i, desc_dim, 1);
            j++;
        }
    }
    map.conservativeResize(j);
    map.removeDescriptor("movable");

    std::cout << "Saving the filtered map..." << std::endl;
    map.save(root + "map/map_no_movable.ply");
}
