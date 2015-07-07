#ifndef GPUMESH_INSPHEREEVALUATOR
#define GPUMESH_INSPHEREEVALUATOR

#include "AbstractEvaluator.h"


class InsphereEvaluator : public AbstractEvaluator
{
public:
    InsphereEvaluator();
    virtual ~InsphereEvaluator();


    virtual double tetrahedronQuality(const Mesh& mesh, const MeshTet& tet) const override;

    virtual double hexahedronQuality(const Mesh& mesh, const MeshHex& hex) const override;

    virtual double prismQuality(const Mesh& mesh, const MeshPri& pri) const override;

    virtual void evaluateCpuMeshQuality(
            const Mesh& mesh,
            double& minQuality,
            double& qualityMean) override;
};

#endif // GPUMESH_INSPHEREEVALUATOR
