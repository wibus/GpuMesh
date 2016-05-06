#ifndef GPUMESH_BARTTOPOLOGIST
#define GPUMESH_BARTTOPOLOGIST

#include "AbstractTopologist.h"

class TriangularBoundary;


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
    template<typename T>
    static bool popOut(std::vector<T>& vec, const T& val);

    template<typename T, typename V>
    static bool popOut(std::vector<T>& vec, const std::vector<V>& val);

    template<typename T>
    static bool intersects(const std::vector<T>& a, const std::vector<T>& b);

    void findRing(
            const TriangularBoundary& bounds,
            const Mesh& mesh,
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
};

#endif // GPUMESH_BARTTOPOLOGIST
