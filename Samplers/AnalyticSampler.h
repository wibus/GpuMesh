#ifndef GPUMESH_ANALYTICSAMPLER
#define GPUMESH_ANALYTICSAMPLER

#include "AbstractSampler.h"


class AnalyticSampler : public AbstractSampler
{
public:
    AnalyticSampler();
    virtual ~AnalyticSampler();


    virtual bool isMetricWise() const override;


    virtual void setReferenceMesh(
            const Mesh& mesh,
            int density) override;

    virtual Metric metricAt(
            const glm::dvec3& position) const override;


    virtual void releaseDebugMesh() override;
    virtual const Mesh& debugMesh() override;


protected:


private:
    std::shared_ptr<Mesh> _debugMesh;
};

#endif // GPUMESH_ANALYTICSAMPLER
