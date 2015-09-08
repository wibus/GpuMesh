#ifndef GPUMESH_LOCALOPTIMISATIONSMOOTHER
#define GPUMESH_LOCALOPTIMISATIONSMOOTHER


#include "AbstractVertexWiseSmoother.h"


class LocalOptimisationSmoother : public AbstractVertexWiseSmoother
{
public:
    LocalOptimisationSmoother();
    virtual ~LocalOptimisationSmoother();

protected:
    virtual void smoothVertices(
            Mesh& mesh,
            AbstractEvaluator& evaluator,
            const std::vector<uint>& vIds) override;

protected:
    virtual void setVertexProgramUniforms(
            const Mesh& mesh,
            cellar::GlProgram& program);

private:
    int _securityCycleCount;
};

#endif // GPUMESH_LOCALOPTIMISATIONSMOOTHER
