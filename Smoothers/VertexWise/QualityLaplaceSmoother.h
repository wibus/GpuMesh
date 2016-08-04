#ifndef GPUMESH_QUALITYLAPLACESMOOTHER
#define GPUMESH_QUALITYLAPLACESMOOTHER


#include "AbstractVertexWiseSmoother.h"


class QualityLaplaceSmoother : public AbstractVertexWiseSmoother
{
public:
    QualityLaplaceSmoother();
    virtual ~QualityLaplaceSmoother();

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

#endif // GPUMESH_QUALITYLAPLACESMOOTHER
