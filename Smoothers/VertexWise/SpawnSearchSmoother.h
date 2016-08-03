#ifndef GPUMESH_SPAWNSEARCHSMOOTHER
#define GPUMESH_SPAWNSEARCHSMOOTHER

#include <vector>

#include "AbstractVertexWiseSmoother.h"


class SpawnSearchSmoother : public AbstractVertexWiseSmoother
{
public:
    SpawnSearchSmoother();
    virtual ~SpawnSearchSmoother();

protected:
    virtual void setVertexProgramUniforms(
            const Mesh& mesh,
            cellar::GlProgram& program) override;

    virtual void printOptimisationParameters(
            const Mesh& mesh,
            OptimizationPlot& plot) const override;

    virtual void smoothVertices(
            Mesh& mesh,
            const MeshCrew& crew,
            const std::vector<uint>& vIds) override;

    virtual std::string glslLauncher() const override;

    virtual glm::ivec3 layoutWorkgroups(
            const NodeGroups::GpuDispatch& dispatch) const override;

private:
    static const int PROPOSITION_COUNT;
    std::vector<glm::dvec4> _offsets;

    GLuint _offsetsSsbo;
};

#endif // GPUMESH_SPAWNSEARCHSMOOTHER

