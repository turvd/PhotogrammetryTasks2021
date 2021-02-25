#pragma once

#include <cstdint>
#include <vector>

namespace phg {

    uint64_t xorshift64(uint64_t *state);

    void randomSample(std::vector<int> &dst, int max_id, int sample_size, uint64_t *state);

}
