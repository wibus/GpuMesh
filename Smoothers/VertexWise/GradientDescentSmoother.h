#ifndef GPUMESH_LOCALGRADIENTDESCENTOPTIMISATIONSMOOTHER
#define GPUMESH_LOCALGRADIENTDESCENTOPTIMISATIONSMOOTHER


#include "AbstractVertexWiseSmoother.h"


class GradientDescentSmoother : public AbstractVertexWiseSmoother
{
protected:
    GradientDescentSmoother(
        const std::vector<std::string>& smoothShaders,
        const installCudaFct& installCuda,
        const launchCudaKernelFct& launchCudaKernel);

public:
    GradientDescentSmoother();
    virtual ~GradientDescentSmoother();


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

#endif // GPUMESH_LOCALGRADIENTDESCENTOPTIMISATIONSMOOTHER
