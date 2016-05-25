#ifndef GPUMESH_BARTTOPOLOGIST
#define GPUMESH_BARTTOPOLOGIST

#include "AbstractTopologist.h"

class MeshTri;
class MeshTopo;


class BatrTopologist : public AbstractTopologist
{
public:
    BatrTopologist();

    virtual ~BatrTopologist();


    virtual bool needTopologicalModifications(
            int vertRelocationPassCount,
            const Mesh& mesh) const override;

    virtual void restructureMesh(
            Mesh& mesh,
            const MeshCrew& crew) const override;

    virtual void printOptimisationParameters(
            const Mesh& mesh,
            OptimizationPlot& plot) const override;


protected:    
    virtual size_t edgeSplitMerge(
            Mesh& mesh,
            const MeshCrew& crew) const;

    virtual size_t faceSwapping(
            Mesh& mesh,
            const MeshCrew& crew) const;

    virtual size_t edgeSwapping(
            Mesh& mesh,
            const MeshCrew& crew) const;

private:
    template<typename T, typename V>
    static bool popOut(std::vector<T>& vec, const V& val);

    template<typename T, typename V>
    static bool popOut(std::vector<T>& vec, const std::vector<V>& val);

    template<typename T, typename V>
    static bool make_union(std::vector<T>& vec, const V& val);

    void buildVertNeighborhood(Mesh& mesh, uint vId) const;

    void findRing(
            const Mesh& mesh,
            uint vId, uint nId,
            std::vector<uint>& ringVerts,
            std::vector<uint>& ringElems) const;

    void findExclusiveElems(
            const Mesh& mesh, uint vId,
            const std::vector<uint>& ringElems,
            std::vector<uint>& exElems) const;

    void trimVerts(
            Mesh& mesh,
            const std::vector<bool>& aliveVerts) const;

    void trimTets(
            Mesh& mesh,
            const std::vector<bool>& aliveTets) const;

    bool cureBoundaries(
            Mesh& mesh,
            std::vector<uint>& vertsToVerifiy,
            std::vector<bool>& aliveVerts,
            std::vector<uint>& deadVerts,
            std::vector<bool>& aliveElems,
            std::vector<uint>& deadElems) const;

    void printRing(const Mesh& mesh, uint vId, uint nId) const;

    bool validateMesh(const Mesh& mesh,
                      const std::vector<bool>& aliveTets,
                      const std::vector<bool>& aliveVerts) const;


    struct RingConfig
    {
        RingConfig(uint rotCount, const std::vector<MeshTri>& tris) :
            rotCount(rotCount), tris(tris) {}
        std::vector<MeshTri> tris;
        uint rotCount;
    };

    std::vector<std::vector<RingConfig>> _ringConfigDictionary;
    size_t _refinementCoarseningMaxPassCount;
    size_t _globalLoopMaxPassCount;
    double _minAcceptableGenQuality;
};

#endif // GPUMESH_BARTTOPOLOGIST
