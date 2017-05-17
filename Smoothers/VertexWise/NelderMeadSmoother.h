#ifndef GPUMESH_LOCALNELDERMEADOPTIMISATIONSMOOTHER
#define GPUMESH_LOCALNELDERMEADOPTIMISATIONSMOOTHER


#include "AbstractVertexWiseSmoother.h"


class NelderMeadSmoother : public AbstractVertexWiseSmoother
{
protected:
    NelderMeadSmoother(
        const std::vector<std::string>& smoothShaders,
        const installCudaFct& installCuda,
        const launchCudaKernelFct& launchCudaKernel);

public:
    NelderMeadSmoother();
    virtual ~NelderMeadSmoother();


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

#endif // GPUMESH_LOCALNELDERMEADOPTIMISATIONSMOOTHER
