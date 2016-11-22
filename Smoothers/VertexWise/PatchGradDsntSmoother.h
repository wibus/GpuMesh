#ifndef GPUMESH_PATCHGRADDSNTSMOOTHER
#define GPUMESH_PATCHGRADDSNTSMOOTHER

#include "GradientDescentSmoother.h"

class PatchGradDsntSmoother : public GradientDescentSmoother
{
public:
    PatchGradDsntSmoother();
    virtual ~PatchGradDsntSmoother();

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
            const NodeGroups::GpuDispatch& dispatch);

    virtual void setVertexProgramUniforms(
            const Mesh& mesh,
            cellar::GlProgram& program) override;

    virtual void printOptimisationParameters(
            const Mesh& mesh,
            OptimizationImpl& plotImpl) const override;

    virtual void smoothVertices(
            Mesh& mesh,
            const MeshCrew& crew,
            const std::vector<uint>& vIds) override;

    virtual std::string glslLauncher() const override;

    virtual glm::ivec3 layoutWorkgroups(
            const NodeGroups::GpuDispatch& dispatch) const override;

};

#endif // GPUMESH_PATCHGRADDSNTSMOOTHER
