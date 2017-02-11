#include "NodeGroups.h"

#include <set>
#include <iostream>
#include <algorithm>

#include <CellarWorkbench/Misc/Log.h>

#include "Mesh.h"
#include "Boundaries/Constraints/AbstractConstraint.h"

using namespace cellar;


const int NodeGroups::NO_GROUP = -2;
const int NodeGroups::UNSET_GROUP = -1;

const int NodeGroups::UNSET_TYPE = -1;
const int NodeGroups::FIXED_TYPE = 0;
const int NodeGroups::BOUND_TYPE = 1;
const int NodeGroups::SUBSU_TYPE = 2;
const int NodeGroups::INTER_TYPE = 3;


NodeGroups::Range::Range() :
    begin(0),
    end(0)
{

}

NodeGroups::GpuDispatch::GpuDispatch() :
    gpuBufferBase(0),
    gpuBufferSize(0),
    workgroupCount(0)
{

}

NodeGroups::NodeGroups() :
    _cpuWorkgroupSize(1),
    _gpuWorkgroupSize(1)
{

}


void NodeGroups::setCpuWorkgroupSize(size_t size)
{
    _cpuWorkgroupSize = size;
    dispatchCpuWorkgroups();
    dispatchGpuWorkgroups();
}

void NodeGroups::setGpuWorkgroupSize(size_t size)
{
    _gpuWorkgroupSize = size;
    dispatchGpuWorkgroups();
}

void NodeGroups::clear()
{
    _fixedNodes = Range();
    _boundaryNodes = Range();
    _subsurfaceNodes = Range();
    _interiorNodes = Range();

    _nodeVector.clear();
    _nodeVector.shrink_to_fit();

    _gpuGroupsBuffer.clear();
    _gpuGroupsBuffer.shrink_to_fit();

    _parallelGroups.clear();
    _parallelGroups.shrink_to_fit();
}

void NodeGroups::shrink_to_fit()
{
    _nodeVector.shrink_to_fit();

    _gpuGroupsBuffer.shrink_to_fit();

    _parallelGroups.shrink_to_fit();
    for(ParallelGroup& group : _parallelGroups)
    {
        group.undispatchedNodes.shrink_to_fit();

        group.allDispatchedNodes.shrink_to_fit();
        for(std::vector<uint>& dispatch : group.allDispatchedNodes)
            dispatch.shrink_to_fit();

        group.cpuOnlyDispatchedNodes.shrink_to_fit();
        for(std::vector<uint>& dispatch : group.cpuOnlyDispatchedNodes)
            dispatch.shrink_to_fit();
    }
}

void NodeGroups::build(Mesh& mesh)
{
    clear();

    std::vector<int> types;
    determineTypes(mesh, types);

    std::vector<int> groups;
    determineGroups(mesh, groups);

    std::vector<int> positions;
    determinePositions(mesh, types, groups, positions);

    clusterNodes(mesh, types, groups, positions);

    // Dispatch workgroups
    dispatchCpuWorkgroups();
    dispatchGpuWorkgroups();
}

void NodeGroups::determineTypes(const Mesh& mesh, std::vector<int>& types)
{
    size_t vertCount = mesh.verts.size();
    types = std::vector<int> (vertCount, UNSET_TYPE);
    const std::vector<MeshTopo>& topos = mesh.topos;

    ///////////////////
    // Classify tets //
    ///////////////////
    const std::vector<MeshTet>& tets = mesh.tets;
    std::vector<bool> boundingTets(tets.size(), false);
    for(size_t eId=0; eId < tets.size(); ++eId)
    {
        const MeshTet& elem = tets[eId];
        if(isMovableBound(topos[elem.v[0]]) ||
           isMovableBound(topos[elem.v[1]]) ||
           isMovableBound(topos[elem.v[2]]) ||
           isMovableBound(topos[elem.v[3]]))
        {
            boundingTets[eId] = true;
        }
    }

    ///////////////////
    // Classify pris //
    ///////////////////
    const std::vector<MeshPri>& pris = mesh.pris;
    std::vector<bool> boundingPris(pris.size(), false);
    for(size_t eId=0; eId < pris.size(); ++eId)
    {
        const MeshPri& elem = pris[eId];
        if(isMovableBound(topos[elem.v[0]]) ||
           isMovableBound(topos[elem.v[1]]) ||
           isMovableBound(topos[elem.v[2]]) ||
           isMovableBound(topos[elem.v[3]]) ||
           isMovableBound(topos[elem.v[4]]) ||
           isMovableBound(topos[elem.v[5]]))
        {
            boundingPris[eId] = true;
        }
    }

    ///////////////////
    // Classify hexs //
    ///////////////////
    const std::vector<MeshHex>& hexs = mesh.hexs;
    std::vector<bool> boundingHexs(hexs.size(), false);
    for(size_t eId=0; eId < hexs.size(); ++eId)
    {
        const MeshHex& elem = hexs[eId];
        if(isMovableBound(topos[elem.v[0]]) ||
           isMovableBound(topos[elem.v[1]]) ||
           isMovableBound(topos[elem.v[2]]) ||
           isMovableBound(topos[elem.v[3]]) ||
           isMovableBound(topos[elem.v[4]]) ||
           isMovableBound(topos[elem.v[5]]) ||
           isMovableBound(topos[elem.v[6]]) ||
           isMovableBound(topos[elem.v[7]]))
        {
            boundingHexs[eId] = true;
        }
    }


    ////////////////////
    // Classify verts //
    ////////////////////
    for(size_t vId=0; vId < vertCount; ++vId)
    {
        const MeshTopo& topo = topos[vId];
        if(topo.snapToBoundary->isFixed())
        {
            types[vId] = FIXED_TYPE;
        }
        else if(topo.snapToBoundary->isConstrained())
        {
            types[vId] = BOUND_TYPE;
        }
        else
        {
            bool isSubsurface = false;
            for(const MeshNeigElem& ne : topo.neighborElems)
            {
                if((ne.type == MeshTet::ELEMENT_TYPE && boundingTets[ne.id]) ||
                   (ne.type == MeshPri::ELEMENT_TYPE && boundingPris[ne.id]) ||
                   (ne.type == MeshHex::ELEMENT_TYPE && boundingHexs[ne.id]))
                {
                    isSubsurface = true;
                    break;
                }
            }

            if(isSubsurface)
                types[vId] = SUBSU_TYPE;
            else
                types[vId] = INTER_TYPE;
        }
    }
}

void NodeGroups::determineGroups(const Mesh& mesh, std::vector<int>& groups)
{
    size_t vertCount = mesh.verts.size();
    const std::vector<MeshTet>& tets = mesh.tets;
    const std::vector<MeshPri>& pris = mesh.pris;
    const std::vector<MeshHex>& hexs = mesh.hexs;
    const std::vector<MeshTopo>& topos = mesh.topos;

    size_t seekStart = 0;
    std::set<uint> existingGroups;
    std::vector<size_t> nextNodes;
    groups = std::vector<int> (vertCount, NO_GROUP);
    while(nextNodes.size() < vertCount)
    {
        size_t firstNode = nextNodes.size();
        for(size_t vId=seekStart; vId < vertCount; ++vId)
        {
            ++seekStart;
            if(groups[vId] == NO_GROUP)
            {
                groups[vId] = UNSET_GROUP;
                nextNodes.push_back(vId);
                break;
            }
        }

        for(int v=firstNode; v < nextNodes.size(); ++v)
        {
            uint vId = nextNodes[v];
            const MeshTopo& topo = topos[vId];
            std::set<uint> availableGroups = existingGroups;

            for(size_t e=0; e < topo.neighborElems.size(); ++e)
            {
                const MeshNeigElem& neigElem = topo.neighborElems[e];
                if(neigElem.type == MeshTet::ELEMENT_TYPE)
                {
                    const MeshTet& elem = tets[neigElem.id];
                    for(size_t n=0; n < MeshTet::VERTEX_COUNT; ++n)
                    {
                        int& group = groups[elem.v[n]];
                        if(group == NO_GROUP)
                        {
                            group = UNSET_GROUP;
                            nextNodes.push_back(elem.v[n]);
                        }
                        else if(group != UNSET_GROUP)
                        {
                            availableGroups.erase(group);
                        }
                    }
                }
                else if(neigElem.type == MeshPri::ELEMENT_TYPE)
                {
                    const MeshPri& elem = pris[neigElem.id];
                    for(size_t n=0; n < MeshPri::VERTEX_COUNT; ++n)
                    {
                        int& group = groups[elem.v[n]];
                        if(group == NO_GROUP)
                        {
                            group = UNSET_GROUP;
                            nextNodes.push_back(elem.v[n]);
                        }
                        else if(group != UNSET_GROUP)
                        {
                            availableGroups.erase(group);
                        }
                    }
                }
                else if(neigElem.type == MeshHex::ELEMENT_TYPE)
                {
                    const MeshHex& elem = hexs[neigElem.id];
                    for(size_t n=0; n < MeshHex::VERTEX_COUNT; ++n)
                    {
                        int& group = groups[elem.v[n]];
                        if(group == NO_GROUP)
                        {
                            group = UNSET_GROUP;
                            nextNodes.push_back(elem.v[n]);
                        }
                        else if(group != UNSET_GROUP)
                        {
                            availableGroups.erase(group);
                        }
                    }
                }
            }

            int group;
            if(availableGroups.empty())
            {
                group = existingGroups.size();
                existingGroups.insert(group);
            }
            else
            {
                group = *availableGroups.begin();
            }

            groups[vId] = group;
        }
    }
}

void NodeGroups::determinePositions(Mesh& mesh,
        const std::vector<int> &types,
        const std::vector<int> &groups,
        std::vector<int>& positions)
{
    const auto& topos = mesh.topos;
    size_t vertCount = mesh.verts.size();

    //////////////////
    // Build ranges //
    //////////////////
    for(size_t vId=0; vId < vertCount; ++vId)
    {
        if(types[vId] == FIXED_TYPE)
        {
            ++_fixedNodes.end;
        }
        else  if(types[vId] == BOUND_TYPE)
        {
            ++_boundaryNodes.end;
        }
        else  if(types[vId] == SUBSU_TYPE)
        {
            ++_subsurfaceNodes.end;
        }
        else  if(types[vId] == INTER_TYPE)
        {
            ++_interiorNodes.end;
        }
    }

    _boundaryNodes.begin += _fixedNodes.end;
    _boundaryNodes.end += _fixedNodes.end;

    _subsurfaceNodes.begin += _boundaryNodes.end;
    _subsurfaceNodes.end += _boundaryNodes.end;

    _interiorNodes.begin += _subsurfaceNodes.end;
    _interiorNodes.end += _subsurfaceNodes.end;

    assert(_interiorNodes.end == vertCount);


    //////////////////////
    // Cluster vertices //
    //////////////////////
    std::vector<int> indices(vertCount);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&] (int a, int b) {
        if(types[a] == types[b])
        {
            if(types[a] == FIXED_TYPE)
            {
                return false;
            }
            else
            {
                if(groups[a] == groups[b])
                {
                    // Inside type-group sets, nodes are sorted by their
                    // number of neighbor elements. This enables us to balance
                    // CPU and GPU dispatches. CPU dispatches needs a uniform
                    // distribution of neighborhood sizes. GPU dispatches prefer
                    // workgroups(blocks) with of nodes with equal number of neighbors

                    // See dispatchCpuWorkgroups() and dispatchGpuWorkgroups()
                    // to know how dispatches are built based on this sorting
                    return topos[a].neighborElems.size() <
                            topos[b].neighborElems.size();
                }
                else
                {
                    return groups[a] < groups[b];
                }
            }
        }
        else
        {
            return types[a] < types[b];
        }
    });

    positions.resize(indices.size());
    for(size_t vId=0; vId < vertCount; ++vId)
        positions[indices[vId]] = vId;
}

void NodeGroups::clusterNodes(Mesh& mesh,
        std::vector<int>& types,
        std::vector<int>& groups,
        std::vector<int>& positions)
{
    size_t vertCount = mesh.verts.size();
    std::vector<MeshTet>& tets = mesh.tets;
    std::vector<MeshPri>& pris = mesh.pris;
    std::vector<MeshHex>& hexs = mesh.hexs;
    std::vector<MeshVert>& verts = mesh.verts;
    std::vector<MeshTopo>& topos = mesh.topos;

    for(MeshTet& tet : tets)
    {
        tet.v[0] = positions[tet.v[0]];
        tet.v[1] = positions[tet.v[1]];
        tet.v[2] = positions[tet.v[2]];
        tet.v[3] = positions[tet.v[3]];
    }

    for(MeshPri& pri : pris)
    {
        pri.v[0] = positions[pri.v[0]];
        pri.v[1] = positions[pri.v[1]];
        pri.v[2] = positions[pri.v[2]];
        pri.v[3] = positions[pri.v[3]];
        pri.v[4] = positions[pri.v[4]];
        pri.v[5] = positions[pri.v[5]];
    }

    for(MeshHex& hex : hexs)
    {
        hex.v[0] = positions[hex.v[0]];
        hex.v[1] = positions[hex.v[1]];
        hex.v[2] = positions[hex.v[2]];
        hex.v[3] = positions[hex.v[3]];
        hex.v[4] = positions[hex.v[4]];
        hex.v[5] = positions[hex.v[5]];
        hex.v[6] = positions[hex.v[6]];
        hex.v[7] = positions[hex.v[7]];
    }

    for(MeshTopo& topo : topos)
    {
        for(MeshNeigVert& vN : topo.neighborVerts)
            vN.v = positions[vN.v];
    }



    int lastType = UNSET_TYPE;
    int lastGroup = UNSET_GROUP;
    for(size_t vId = 0; vId < vertCount; ++vId)
    {
        // Find the node that should be placed here
        while(positions[vId] != vId)
        {
            size_t nId = positions[vId];
            std::swap(verts[vId], verts[nId]);
            std::swap(topos[vId], topos[nId]);
            std::swap(types[vId], types[nId]);
            std::swap(groups[vId], groups[nId]);
            std::swap(positions[vId], positions[nId]);
        }

        int type = types[vId];
        if(type != lastType)
        {
            assert(lastType < type);
            lastType = type;
            lastGroup = -1;
        }

        if(type > FIXED_TYPE)
        {
            _nodeVector.push_back(vId);


            int gId = groups[vId];
            if(gId != lastGroup)
            {
                assert(lastGroup < gId);
                lastGroup = gId;

                if(_parallelGroups.size() <= gId)
                    _parallelGroups.resize(gId+1);

                ParallelGroup& currGroup = _parallelGroups[gId];
                currGroup.gpuDispatch.gpuBufferBase = _gpuGroupsBuffer.size();

                if(type == BOUND_TYPE)
                {
                    currGroup.boundaryRange.begin = vId;
                    currGroup.boundaryRange.end = vId;
                }
                else if(type == SUBSU_TYPE)
                {
                    currGroup.subsurfaceRange.begin = vId;
                    currGroup.subsurfaceRange.end = vId;
                }
                else if(type == INTER_TYPE)
                {
                    currGroup.interiorRange.begin = vId;
                    currGroup.interiorRange.end = vId;
                }
            }

            ParallelGroup& group = _parallelGroups[gId];
            group.undispatchedNodes.push_back(vId);

            if(type == BOUND_TYPE)
            {
                ++group.boundaryRange.end;
            }
            else if(type == SUBSU_TYPE)
            {
                ++group.subsurfaceRange.end;
            }
            else if(type == INTER_TYPE)
            {
                ++group.interiorRange.end;
            }
        }
    }


    // Fill-in GPU dispatches
    for(ParallelGroup& group : _parallelGroups)
    {
        size_t cpuGroupSize = group.boundaryRange.end - group.boundaryRange.begin;
        size_t gpuGroupSize = group.undispatchedNodes.size() - cpuGroupSize;

        // This threshold makes sure small dispatch aren't sent to the GPU
        // Small dispatch may produce more latency that parallelism mar recover
        size_t gpuGroupSizeThreshold = 0;
        if(gpuGroupSize >= gpuGroupSizeThreshold)
        {
            group.gpuDispatch.gpuBufferBase =
                    _gpuGroupsBuffer.size();
            group.gpuDispatch.gpuBufferSize =
                    gpuGroupSize;

            _gpuGroupsBuffer.insert(_gpuGroupsBuffer.end(),
                group.undispatchedNodes.begin() + cpuGroupSize,
                group.undispatchedNodes.end());
        }
        else
        {
            getLog().postMessage(new Message('D', false, "GPU dispatch of " +
                std::to_string(gpuGroupSize) + " skipped", "NodeGroups"));

            group.gpuDispatch.gpuBufferBase =
                    _gpuGroupsBuffer.size();
            group.gpuDispatch.gpuBufferSize =
                    0;
        }
    }
}

void NodeGroups::dispatchCpuWorkgroups()
{
    for(ParallelGroup& group : _parallelGroups)
    {
        group.allDispatchedNodes.clear();
        group.allDispatchedNodes.resize(_cpuWorkgroupSize);
        size_t allGroupSize = group.undispatchedNodes.size();
        size_t maxDispatchSize = glm::ceil(double(allGroupSize) / _cpuWorkgroupSize);
        for(size_t w=0; w < _cpuWorkgroupSize; ++w)
        {
            // Nodes are sorted according to their number of neighbor elements
            // This distribution pattern makes sure CPU dispatches are well balanced

            group.allDispatchedNodes[w].reserve(maxDispatchSize);
            for(size_t v = w; v < allGroupSize; v += _cpuWorkgroupSize)
            {
                group.allDispatchedNodes[w].push_back(
                    group.undispatchedNodes[v]);
            }
        }
    }
}

void NodeGroups::dispatchGpuWorkgroups()
{
    for(ParallelGroup& group : _parallelGroups)
    {
        double gpuGroupSize = group.gpuDispatch.gpuBufferSize;
        group.gpuDispatch.workgroupCount =
            glm::ceil(gpuGroupSize / _gpuWorkgroupSize);

        group.cpuOnlyDispatchedNodes.clear();
        group.cpuOnlyDispatchedNodes.resize(_cpuWorkgroupSize);

        size_t cpuGroupSize = group.boundaryRange.end -
                group.boundaryRange.begin;

        if(group.gpuDispatch.gpuBufferSize == 0)
        {
            cpuGroupSize = group.undispatchedNodes.size();
        }

        size_t maxDispatchSize = glm::ceil(double(cpuGroupSize) / _cpuWorkgroupSize);
        for(size_t w=0; w < _cpuWorkgroupSize; ++w)
        {
            // Nodes are sorted according to their number of neighbor elements
            // This distribution pattern makes sure CPU dispatches are well balanced

            group.cpuOnlyDispatchedNodes[w].reserve(maxDispatchSize);
            for(size_t v = w; v < cpuGroupSize; v += _cpuWorkgroupSize)
            {
                group.cpuOnlyDispatchedNodes[w].push_back(
                    group.undispatchedNodes[v]);
            }
        }
    }
}

bool NodeGroups::isMovableBound(const MeshTopo& topo)
{
    if(topo.snapToBoundary->isConstrained() &&
       !topo.snapToBoundary->isFixed())
    {
        return true;
    }

    return false;
}
