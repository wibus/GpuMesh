#ifndef GPUMESH_BARTTOPOLOGIST
#define GPUMESH_BARTTOPOLOGIST

#include <set>

#include "AbstractTopologist.h"

class MeshTet;
typedef std::pair<uint, uint> LocalEdge;


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
    virtual void edgeSplitMerge(
            Mesh& mesh,
            const MeshCrew& crew) const;

    virtual void faceSwapping(
            Mesh& mesh,
            const MeshCrew& crew) const;

    virtual void edgeSwapping(
            Mesh& mesh,
            const MeshCrew& crew) const;


    // Helper methods
    void buildBoundaryEdgeSet(
            std::vector<MeshTet>& tets,
            std::set<LocalEdge>& edges) const;
};

#endif // GPUMESH_BARTTOPOLOGIST
