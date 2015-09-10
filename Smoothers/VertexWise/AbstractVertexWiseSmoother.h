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
            AbstractEvaluator& evaluator) override;

    virtual void smoothMeshThread(
            Mesh& mesh,
            AbstractEvaluator& evaluator) override;

    virtual void smoothMeshGlsl(
            Mesh& mesh,
            AbstractEvaluator& evaluator) override;


protected:
    virtual void printImplParameters(
            const Mesh& mesh,
            const AbstractEvaluator& evaluator,
            OptimizationImpl& implementation) const override;

    virtual void initializeProgram(
            Mesh& mesh,
            AbstractEvaluator& evaluator) override;

    virtual void setVertexProgramUniforms(
            const Mesh& mesh,
            cellar::GlProgram& program);

    virtual void smoothVertices(
            Mesh& mesh,
            AbstractEvaluator& evaluator,
            const std::vector<uint>& vIds) = 0;




private:
    static const size_t WORKGROUP_SIZE;

    bool _initialized;
    std::string _modelBoundsShader;
    std::string _shapeMeasureShader;
    std::vector<std::string> _smoothShaders;
    cellar::GlProgram _vertSmoothProgram;
};

#endif // GPUMESH_ABSTRACTVERTEXWISESMOOTHER
