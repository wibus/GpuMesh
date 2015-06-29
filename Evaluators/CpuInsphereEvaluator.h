#ifndef GPUMESH_CPUINSPHEREEVALUATOR
#define GPUMESH_CPUINSPHEREEVALUATOR

#include "AbstractEvaluator.h"


class CpuInsphereEvaluator : public AbstractEvaluator
{
public:
    CpuInsphereEvaluator();
    virtual ~CpuInsphereEvaluator();


    virtual double tetrahedronQuality(const Mesh& mesh, const MeshTet& tet) const override;

    virtual double hexahedronQuality(const Mesh& mesh, const MeshHex& hex) const override;

    virtual double prismQuality(const Mesh& mesh, const MeshPri& pri) const override;

    virtual void evaluateMeshQuality(
            const Mesh& mesh,
            double& qualityMean,
            double& qualityVar,
            double& minQuality) override;
};

#endif // GPUMESH_CPUINSPHEREEVALUATOR
