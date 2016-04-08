#ifndef GPUMESH_DUMMYSAMPLER
#define GPUMESH_DUMMYSAMPLER

#include "AbstractSampler.h"


class DummySampler : public AbstractSampler
{
public:
    DummySampler();
    virtual ~DummySampler();


    virtual bool isMetricWise() const override;


    virtual void setReferenceMesh(
            const Mesh& mesh) override;

    virtual Metric metricAt(
            const glm::dvec3& position,
            uint cacheId) const override;


    virtual void releaseDebugMesh() override;
    virtual const Mesh& debugMesh() override;


protected:


private:
    std::shared_ptr<Mesh> _debugMesh;
};

#endif // GPUMESH_DUMMYSAMPLER
