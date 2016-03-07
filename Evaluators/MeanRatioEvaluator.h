#ifndef GPUMESH_MEANRATIOEVALUATOR
#define GPUMESH_MEANRATIOEVALUATOR

#include "AbstractEvaluator.h"


class MeanRatioEvaluator : public AbstractEvaluator
{
public:
    MeanRatioEvaluator();
    virtual ~MeanRatioEvaluator();

    virtual double tetQuality(
            const AbstractSampler& sampler,
            const AbstractMeasurer& measurer,
            const glm::dvec3 vp[],
            const MeshTet& tet) const override;

    virtual double priQuality(
            const AbstractSampler& sampler,
            const AbstractMeasurer& measurer,
            const glm::dvec3 vp[],
            const MeshPri& pri) const override;

    virtual double hexQuality(
            const AbstractSampler& sampler,
            const AbstractMeasurer& measurer,
            const glm::dvec3 vp[],
            const MeshHex& hex) const override;

protected:
    virtual double cornerQuality(const glm::dmat3& Fk) const;
};

#endif // GPUMESH_MEANRATIOEVALUATOR
