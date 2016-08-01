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

private:
    static const int PROPOSITION_COUNT;
    std::vector<glm::dvec3> _offsets;
};

#endif // GPUMESH_SPAWNSEARCHSMOOTHER

