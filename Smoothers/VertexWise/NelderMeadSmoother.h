#ifndef GPUMESH_LOCALNELDERMEADOPTIMISATIONSMOOTHER
#define GPUMESH_LOCALNELDERMEADOPTIMISATIONSMOOTHER


#include "AbstractVertexWiseSmoother.h"


class NelderMeadSmoother : public AbstractVertexWiseSmoother
{
public:
    NelderMeadSmoother();
    virtual ~NelderMeadSmoother();


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
};

#endif // GPUMESH_LOCALNELDERMEADOPTIMISATIONSMOOTHER
