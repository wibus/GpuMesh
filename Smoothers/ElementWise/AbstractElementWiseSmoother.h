#ifndef GPUMESH_ABSTRACTELEMENTWISESMOOTHER
#define GPUMESH_ABSTRACTELEMENTWISESMOOTHER

#include "../AbstractSmoother.h"

class IVertexAccum;


class AbstractElementWiseSmoother : public AbstractSmoother
{
protected:
    AbstractElementWiseSmoother(
            const std::vector<std::string>& smoothShaders);

public:
    ~AbstractElementWiseSmoother();


    virtual void smoothMeshSerial(
            Mesh& mesh,
            AbstractEvaluator& evaluator) override;

    virtual void smoothMeshThread(
            Mesh& mesh,
            AbstractEvaluator& evaluator) override;

    virtual void smoothMeshGlsl(
            Mesh& mesh,
            AbstractEvaluator& evaluator) override;


protected:
    virtual void initializeProgram(
            Mesh& mesh,
            AbstractEvaluator& evaluator) override;

    virtual void setElementProgramUniforms(
            const Mesh& mesh,
            cellar::GlProgram& program);

    virtual void setVertexProgramUniforms(
            const Mesh& mesh,
            cellar::GlProgram& program);

    virtual void updateVertexPositions(
            Mesh& mesh,
            AbstractEvaluator& evaluator,
            const std::vector<uint>& vIds);

    virtual void smoothTets(
            Mesh& mesh,
            AbstractEvaluator& evaluator,
            size_t first,
            size_t last) = 0;

    virtual void smoothPris(
            Mesh& mesh,
            AbstractEvaluator& evaluator,
            size_t first,
            size_t last) = 0;

    virtual void smoothHexs(
            Mesh& mesh,
            AbstractEvaluator& evaluator,
            size_t first,
            size_t last) = 0;

protected:
    static const size_t WORKGROUP_SIZE;

    bool _initialized;
    std::string _modelBoundsShader;
    std::string _shapeMeasureShader;
    std::vector<std::string> _smoothShaders;
    cellar::GlProgram _elemSmoothProgram;
    cellar::GlProgram _vertUpdateProgram;

    IVertexAccum** _vertexAccums;
    GLuint _accumSsbo;
};

#endif // GPUMESH_ABSTRACTELEMENTWISESMOOTHER
