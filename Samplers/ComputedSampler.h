#ifndef GPUMESH_COMPUTEDSAMPLER
#define GPUMESH_COMPUTEDSAMPLER

#include <GL3/gl3w.h>

#include "LocalSampler.h"


class ComputedSampler : public LocalSampler
{
public:
    ComputedSampler();
    virtual ~ComputedSampler();


    virtual void setReferenceMesh(
            const Mesh& mesh) override;

    virtual void setComputedMetrics(
            const Mesh& mesh,
            const std::vector<MeshMetric>& metrics);
};

#endif // GPUMESH_COMPUTEDSAMPLER
