#include <Eigen/Dense>
#include <string>
#include <cstdarg>

//-------------------------------------------------------------------------------------------
//
// VoxKey (HUGUES THOMAS)
// ******
//
//  Here we define a struct that will be used as key in our hash map. It contains 3 integers.
//  Then we specialize the std::hash function for this class.
//
//-------------------------------------------------------------------------------------------

class VoxKey {
public:
    int x;
    int y;
    int z;

    VoxKey() { x = 0; y = 0; z = 0; }
    VoxKey(int x0, int y0, int z0) { x = x0; y = y0; z = z0; }

    bool operator==(const VoxKey& other) const
    {
        return (x == other.x && y == other.y && z == other.z);
    }
};

inline VoxKey operator + (const VoxKey A, const VoxKey B) {
    return VoxKey(A.x + B.x, A.y + B.y, A.z + B.z);
}

void hash_combine(std::size_t& seed) { }

// Simple utility function to combine hashtables
template <typename T, typename... Rest>
void hash_combine(std::size_t& seed, const T& v, Rest... rest) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    hash_combine(seed, rest...);
}

// Specialization of std::hash function
namespace std{

template <>
struct hash<VoxKey>{
    std::size_t operator()(const VoxKey& k) const {
        std::size_t ret = 0;
        hash_combine(ret, k.x, k.y, k.z);
        return ret;
    }
};
}  // namespace std
