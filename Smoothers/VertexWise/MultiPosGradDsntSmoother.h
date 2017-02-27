#ifndef GPUMESH_MULTIPOSGRADDSNTSMOOTHER
#define GPUMESH_MULTIPOSGRADDSNTSMOOTHER

#include "GradientDescentSmoother.h"

class MultiPosGradDsntSmoother : public GradientDescentSmoother
{
public:
    MultiPosGradDsntSmoother();
    virtual ~MultiPosGradDsntSmoother();

    virtual void smoothMeshGlsl(
            Mesh& mesh,
            const MeshCrew& crew) override;

    virtual void smoothMeshCuda(
            Mesh& mesh,
            const MeshCrew& crew) override;

protected:
    virtual bool verifyMeshForGpuLimitations(
            const Mesh& mesh) const;

    virtual std::string glslLauncher() const override;

    virtual NodeGroups::GpuDispatcher glslDispatcher() const override;

    virtual NodeGroups::GpuDispatcher cudaDispatcher() const override;

protected:
};

#endif // GPUMESH_MULTIPOSGRADDSNTSMOOTHER
