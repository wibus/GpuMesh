#ifndef GPUMESH_SPRINGLAPLACESMOOTHER
#define GPUMESH_SPRINGLAPLACESMOOTHER


#include "AbstractVertexWiseSmoother.h"


class SpringLaplaceSmoother : public AbstractVertexWiseSmoother
{
public:
    SpringLaplaceSmoother();
    virtual ~SpringLaplaceSmoother();

protected:
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
};

#endif // GPUMESH_SPRINGLAPLACESMOOTHER
