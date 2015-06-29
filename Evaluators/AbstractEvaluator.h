#ifndef GPUMESH_ABSTRACTEVALUATOR
#define GPUMESH_ABSTRACTEVALUATOR

#include "DataStructures/Mesh.h"


class AbstractEvaluator
{
public:
    AbstractEvaluator();
    virtual ~AbstractEvaluator();


    virtual double tetrahedronQuality(const Mesh& mesh, const MeshTet& tet) const = 0;

    virtual double hexahedronQuality(const Mesh& mesh, const MeshHex& hex) const = 0;

    virtual double prismQuality(const Mesh& mesh, const MeshPri& pri) const = 0;

    virtual void evaluateMeshQuality(
            const Mesh& mesh,
            double& qualityMean,
            double& qualityVar,
            double& minQuality) = 0;
};

#endif // GPUMESH_ABSTRACTEVALUATOR
