#ifndef GPUMESH_LOCALOPTIMISATIONSMOOTHER
#define GPUMESH_LOCALOPTIMISATIONSMOOTHER


#include "AbstractVertexWiseSmoother.h"


class LocalOptimisationSmoother : public AbstractVertexWiseSmoother
{
public:
    LocalOptimisationSmoother();
    virtual ~LocalOptimisationSmoother();


protected:
    virtual void setVertexProgramUniforms(
            const Mesh& mesh,
            cellar::GlProgram& program);

    virtual void printSmoothingParameters(
            const Mesh& mesh,
            OptimizationPlot& plot) const override;

    virtual void smoothVertices(
            Mesh& mesh,
            const MeshCrew& crew,
            const std::vector<uint>& vIds) override;


private:
    int _securityCycleCount;
    double _localSizeToNodeShift;
};

#endif // GPUMESH_LOCALOPTIMISATIONSMOOTHER
