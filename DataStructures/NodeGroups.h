#ifndef GPUMESH_NODEGROUPS
#define GPUMESH_NODEGROUPS

#include <stddef.h>
#include <vector>


#ifndef uint
typedef unsigned int uint;
#endif // uint

class Mesh;
class MeshTopo;


/// The layout of the vertex buffer is as follows :
///      FixedX : Xth node of the fixed nodes group
///      IndX_BoundY : Yth boundary node of the Xth independent group
///      IndX_SubY : Yth subsurface node of the Xth independent group
///      IndX_InterY : Yth interior node of the Xth independent group
///
/// [ Fixed0 Fixed1 Fixed2 ...
///   ... Ind0_Bound0 Ind0_Bound1 Ind0_Bound2 ...
///   ... Ind1_Bound1 Ind1_Bound1 Ind1_Bound2 ...
///   ... Ind2_Bound1 Ind2_Bound1 Ind2_Bound2 ...
///   ... ...
///   ... Ind0_Sub0 Ind0_Sub1 Ind0_Sub2 ...
///   ... Ind1_Sub1 Ind1_Sub1 Ind1_Sub2 ...
///   ... Ind2_Sub1 Ind2_Sub1 Ind2_Sub2 ...
///   ... ...
///   ... Ind0_Inter0 Ind0_Inter1 Ind0_Inter2 ...
///   ... Ind1_Inter1 Ind1_Inter1 Ind1_Inter2 ...
///   ... Ind2_Inter1 Ind2_Inter1 Ind2_Inter2 ... ]

class NodeGroups
{
public:
    NodeGroups();

    struct Range
    {
        Range();

        size_t begin;
        size_t end;
    };

    struct GpuDispatch
    {
        GpuDispatch();

        // Sequence in gpuGroupsBuffer
        // allocated for this group.
        size_t gpuBufferBase;
        size_t gpuBufferSize;

        // Number of workgroup (or blocks) needed to
        // process all nodes in this independent group
        size_t workgroupCount;
    };

    struct ParallelGroup
    {
        Range boundaryRange;
        Range subsurfaceRange;
        Range interiorRange;

        GpuDispatch gpuDispatch;

        std::vector<uint> undispatchedNodes;
        std::vector<std::vector<uint>> allDispatchedNodes;
        std::vector<std::vector<uint>> cpuOnlyDispatchedNodes;
    };

    size_t count();

    size_t cpuWorkgroupSize() const;
    void setCpuWorkgroupSize(size_t size);

    size_t gpuWorkgroupSize() const;
    void setGpuWorkgroupSize(size_t size);

    const Range& fixedNodes() const;
    const Range& boundaryNodes() const;
    const Range& subsurfaceNodes() const;
    const Range& interiorNodes() const;

    const std::vector<uint>& serialGroup() const;
    const std::vector<uint>& gpuGroupsBuffer() const;
    const std::vector<ParallelGroup>& parallelGroups() const;


    void clear();
    void shrink_to_fit();

    void build(Mesh& mesh);


private:

    const int NO_GROUP = -2;
    const int UNSET_GROUP = -1;

    const int UNSET_TYPE = -1;
    const int FIXED_TYPE = 0;
    const int BOUND_TYPE = 1;
    const int SUBSU_TYPE = 2;
    const int INTER_TYPE = 3;


    /// @brief Independent vertex groups compilation
    /// Compiles independent vertex groups that is used by parallel smoothing
    /// algorithms to ensure that no two _adjacent_ vertices are moved at the
    /// same time. Independent vertices are vertices that do not share a common
    /// element. This is more strict than prohibiting edge existance.
    ///
    /// A simple graph coloring scheme is used to generate the groups. The
    /// algorithm works well with connected and highly disconnected graphs and
    /// show a linear complexity in either case : O(n*d), where _n_ is the
    /// number of vertices and _d_ is the 'mean' vertex degree.
    void determineTypes(const Mesh& mesh, std::vector<int>& types);
    void determineGroups(const Mesh& mesh, std::vector<int>& groups);    
    void determinePositions(Mesh& mesh,
            const std::vector<int> &types,
            const std::vector<int> &groups,
            std::vector<int> &positions);

    void clusterNodes(Mesh& mesh,
            std::vector<int> &types,
            std::vector<int> &groups,
            std::vector<int> &positions);

    void dispatchCpuWorkgroups();

    void dispatchGpuWorkgroups();

    static bool isMovableBound(const MeshTopo& topo);


    size_t _cpuWorkgroupSize;
    size_t _gpuWorkgroupSize;

    // Fixed nodes are uploaded once and
    // never move across smoothing passes
    Range _fixedNodes;

    // Boundary nodes can only be moved on the CPU side
    // They must by uploaded to the GPU side each time
    // an independent group is processed
    Range _boundaryNodes;

    // Subsurface nodes are neighbors of boundary and interior nodes
    // They will be moved on the GPU side, but must be uploaded to
    // to the CPU side each time they are moved.
    Range _subsurfaceNodes;

    // Interior nodes are move on the the GPU side.
    // They come back on the CPU side only when the smoothing temrinates.
    Range _interiorNodes;


    // Contains all the movable nodes
    // MUST be processed in a serial fashion
    std::vector<uint> _serialGroup;

    // Contains all the nodes processed on the GPU
    // GpuDispatch referes to sections of this buffer
    std::vector<uint> _gpuGroupsBuffer;

    // All the independent patch that can be processed
    // individually in a parallel fashion, on the GPU and CPU
    std::vector<ParallelGroup> _parallelGroups;
};



// IMPLEMENTATION //
inline size_t NodeGroups::cpuWorkgroupSize() const
{
    return _cpuWorkgroupSize;
}

inline size_t NodeGroups::gpuWorkgroupSize() const
{
    return _gpuWorkgroupSize;
}

inline size_t NodeGroups::count()
{
    return _parallelGroups.size();
}

inline const NodeGroups::Range& NodeGroups::fixedNodes() const
{
    return _fixedNodes;
}

inline const NodeGroups::Range& NodeGroups::boundaryNodes() const
{
    return _boundaryNodes;
}

inline const NodeGroups::Range& NodeGroups::subsurfaceNodes() const
{
    return _subsurfaceNodes;
}

inline const NodeGroups::Range& NodeGroups::interiorNodes() const
{
    return _interiorNodes;
}

inline const std::vector<uint>& NodeGroups::serialGroup() const
{
    return _serialGroup;
}

inline const std::vector<uint>& NodeGroups::gpuGroupsBuffer() const
{
    return _gpuGroupsBuffer;
}

inline const std::vector<NodeGroups::ParallelGroup>& NodeGroups::parallelGroups() const
{
    return _parallelGroups;
}

#endif // GPUMESH_NODEGROUPS
