#ifndef GPUMESH_VOLUMEEDGEEVALUATOR
#define GPUMESH_VOLUMEEDGEEVALUATOR

#include "AbstractEvaluator.h"


class VolumeEdgeEvaluator : public AbstractEvaluator
{
public:
    VolumeEdgeEvaluator();
    virtual ~VolumeEdgeEvaluator();


    virtual double tetrahedronQuality(const Mesh& mesh, const MeshTet& tet) const override;

    virtual double prismQuality(const Mesh& mesh, const MeshPri& pri) const override;

    virtual double hexahedronQuality(const Mesh& mesh, const MeshHex& hex) const override;


protected:
    double volumeEdgeRatio(
            const glm::dvec3 ev[],
            size_t tetCount,
            const MeshTet tets[],
            size_t edgeCount,
            const MeshEdge edges[]) const;
};

#endif // GPUMESH_VOLUMEEDGEEVALUATOR
