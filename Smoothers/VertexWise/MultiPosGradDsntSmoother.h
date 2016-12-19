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

    virtual void launchCudaKernel(
            const NodeGroups::GpuDispatch& dispatch) override;

    virtual std::string glslLauncher() const override;

    virtual size_t nodesPerBlock() const override;

protected:
    static const int POSITION_THREAD_COUNT;
    static const int NODE_THREAD_COUNT;
};

#endif // GPUMESH_MULTIPOSGRADDSNTSMOOTHER
