#ifndef GPUMESH_GPUEVALUATOR
#define GPUMESH_GPUEVALUATOR

#include <CellarWorkbench/GL/GlProgram.h>

#include "CpuInsphereEvaluator.h"


class GpuEvaluator : public CpuInsphereEvaluator
{
public:
    GpuEvaluator();
    virtual ~GpuEvaluator();

    virtual void evaluateMeshQuality(
            const Mesh& mesh,
            double& qualityMean,
            double& qualityVar,
            double& minQuality) override;


protected:
    virtual void initializeProgram();

    bool _initialized;
    cellar::GlProgram _evaluatorProgram;

    static const double MAX_INTEGER_VALUE;
    static const double MIN_QUALITY_PRECISION_DENOM;
};

#endif // GPUMESH_GPUEVALUATOR
