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
    virtual bool edgeSplitMerge(
            Mesh& mesh,
            const MeshCrew& crew) const;

    virtual bool faceSwapping(
            Mesh& mesh,
            const MeshCrew& crew) const;

    virtual bool edgeSwapping(
            Mesh& mesh,
            const MeshCrew& crew) const;

private:
    template<typename T, typename V>
    static bool popOut(std::vector<T>& vec, const V& val);

    template<typename T, typename V>
    static bool popOut(std::vector<T>& vec, const std::vector<V>& val);

    template<typename T, typename V>
    static bool make_union(std::vector<T>& vec, const V& val);

    template<typename T>
    void findRing(
            const std::vector<T>& tets,
            const std::vector<MeshTopo>& topos,
            uint vId, uint nId,
            std::vector<uint>& ringVerts,
            std::vector<uint>& ringElems) const;

    void findExclusiveElems(
            const Mesh& mesh, uint vId,
            const std::vector<uint>& ringElems,
            std::vector<uint>& exElems) const;

    void findExclusiveVerts(
            const Mesh& mesh,
            uint vId, uint nId,
            const std::vector<uint>& ringVerts,
            std::vector<uint>& exVerts) const;


    struct RingConfig
    {
        RingConfig(uint rotCount, const std::vector<MeshTri>& tris) :
            rotCount(rotCount), tris(tris) {}
        std::vector<MeshTri> tris;
        uint rotCount;
    };

    std::vector<std::vector<RingConfig>> _ringConfigDictionary;
};

#endif // GPUMESH_BARTTOPOLOGIST
