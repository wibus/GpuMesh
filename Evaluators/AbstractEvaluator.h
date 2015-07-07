#ifndef GPUMESH_ABSTRACTEVALUATOR
#define GPUMESH_ABSTRACTEVALUATOR

#include <CellarWorkbench/GL/GlProgram.h>

#include "DataStructures/Mesh.h"


class AbstractEvaluator
{
public:
    AbstractEvaluator(const std::string& shapeMeasuresShader);
    virtual ~AbstractEvaluator();


    virtual double tetrahedronQuality(const Mesh& mesh, const MeshTet& tet) const = 0;

    virtual double hexahedronQuality(const Mesh& mesh, const MeshHex& hex) const = 0;

    virtual double prismQuality(const Mesh& mesh, const MeshPri& pri) const = 0;

    virtual void evaluateCpuMeshQuality(
            const Mesh& mesh,
            double& minQuality,
            double& qualityMean) = 0;

    virtual void evaluateGpuMeshQuality(
            const Mesh& mesh,
            double& minQuality,
            double& qualityMean);

protected:
    virtual void initializeProgram();

    bool _initialized;
    std::string _shapeMeasuresShader;
    cellar::GlProgram _evaluatorProgram;

    static const double MAX_INTEGER_VALUE;
    static const double MIN_QUALITY_PRECISION_DENOM;
};

#endif // GPUMESH_ABSTRACTEVALUATOR
