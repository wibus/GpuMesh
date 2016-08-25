#ifndef GPUMESH_SPAWNSEARCHSMOOTHER
#define GPUMESH_SPAWNSEARCHSMOOTHER

#include <vector>

#include "AbstractVertexWiseSmoother.h"


class SpawnSearchSmoother : public AbstractVertexWiseSmoother
{
public:
    SpawnSearchSmoother();
    virtual ~SpawnSearchSmoother();

    virtual void smoothMeshGlsl(
            Mesh& mesh,
            const MeshCrew& crew) override;

    virtual void smoothMeshCuda(
            Mesh& mesh,
            const MeshCrew& crew) override;

protected:
    virtual bool verifyMeshForGpuLimitations(
            const Mesh& mesh) const;

    virtual void lauchCudaKernel(
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

private:
    static const int PROPOSITION_COUNT;

    GLuint _offsetsSsbo;
};

#endif // GPUMESH_SPAWNSEARCHSMOOTHER

