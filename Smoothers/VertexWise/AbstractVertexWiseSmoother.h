#ifndef GPUMESH_ABSTRACTVERTEXWISESMOOTHER
#define GPUMESH_ABSTRACTVERTEXWISESMOOTHER

#include "../AbstractSmoother.h"


class AbstractVertexWiseSmoother : public AbstractSmoother
{
protected:
    AbstractVertexWiseSmoother(
            const std::vector<std::string>& smoothShaders);

public:
    ~AbstractVertexWiseSmoother();


    virtual void smoothMeshSerial(
            Mesh& mesh,
            const MeshCrew& crew) override;

    virtual void smoothMeshThread(
            Mesh& mesh,
            const MeshCrew& crew) override;

    virtual void smoothMeshGlsl(
            Mesh& mesh,
            const MeshCrew& crew) override;


protected:
    virtual void initializeProgram(
            Mesh& mesh,
            const MeshCrew& crew) override;

    virtual void setVertexProgramUniforms(
            const Mesh& mesh,
            cellar::GlProgram& program);

    virtual void printSmoothingParameters(
            const Mesh& mesh,
            OptimizationPlot& plot) const override;

    virtual void smoothVertices(
            Mesh& mesh,
            const MeshCrew& crew,
            const std::vector<uint>& vIds) = 0;


private:
    static const size_t WORKGROUP_SIZE;

    bool _initialized;
    std::string _modelBoundsShader;
    std::string _discretizationShader;
    std::string _measureShader;
    std::string _evaluationShader;
    std::vector<std::string> _smoothShaders;
    cellar::GlProgram _vertSmoothProgram;
};

#endif // GPUMESH_ABSTRACTVERTEXWISESMOOTHER
