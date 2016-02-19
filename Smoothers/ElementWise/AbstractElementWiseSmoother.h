#ifndef GPUMESH_ABSTRACTELEMENTWISESMOOTHER
#define GPUMESH_ABSTRACTELEMENTWISESMOOTHER

#include "../AbstractSmoother.h"

class IVertexAccum;


class AbstractElementWiseSmoother : public AbstractSmoother
{
protected:
    AbstractElementWiseSmoother(
            const std::vector<std::string>& smoothShaders,
            const installCudaFct installCuda);

public:
    ~AbstractElementWiseSmoother();


    virtual void smoothMeshSerial(
            Mesh& mesh,
            const MeshCrew& crew) override;

    virtual void smoothMeshThread(
            Mesh& mesh,
            const MeshCrew& crew) override;

    virtual void smoothMeshGlsl(
            Mesh& mesh,
            const MeshCrew& crew) override;

    virtual void smoothMeshCuda(
            Mesh& mesh,
            const MeshCrew& crew) override;


protected:
    virtual void initializeProgram(
            Mesh& mesh,
            const MeshCrew& crew) override;

    virtual void setElementProgramUniforms(
            const Mesh& mesh,
            cellar::GlProgram& program);

    virtual void setVertexProgramUniforms(
            const Mesh& mesh,
            cellar::GlProgram& program);

    virtual void printSmoothingParameters(
            const Mesh& mesh,
            OptimizationPlot& plot) const override;

    virtual void updateVertexPositions(
            Mesh& mesh,
            const MeshCrew& crew,
            const std::vector<uint>& vIds);

    virtual void smoothTets(
            Mesh& mesh,
            const MeshCrew& crew,
            size_t first,
            size_t last) = 0;

    virtual void smoothPris(
            Mesh& mesh,
            const MeshCrew& crew,
            size_t first,
            size_t last) = 0;

    virtual void smoothHexs(
            Mesh& mesh,
            const MeshCrew& crew,
            size_t first,
            size_t last) = 0;

protected:
    static const size_t WORKGROUP_SIZE;

    bool _initialized;
    std::string _modelBoundsShader;
    std::string _samplingShader;
    std::string _measureShader;
    std::string _evaluationShader;
    std::vector<std::string> _smoothShaders;
    cellar::GlProgram _elemSmoothProgram;
    cellar::GlProgram _vertUpdateProgram;

    IVertexAccum** _vertexAccums;
    GLuint _accumSsbo;
};

#endif // GPUMESH_ABSTRACTELEMENTWISESMOOTHER
