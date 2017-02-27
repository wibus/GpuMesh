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

    virtual NodeGroups::GpuDispatcher glslDispatcher() const override;

    virtual NodeGroups::GpuDispatcher cudaDispatcher() const override;

private:
    static const int SPAWN_COUNT;

    GLuint _offsetsSsbo;
};

#endif // GPUMESH_SPAWNSEARCHSMOOTHER

