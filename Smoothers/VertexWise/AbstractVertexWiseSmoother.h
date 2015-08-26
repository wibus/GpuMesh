#ifndef GPUMESH_ABSTRACTVERTEXWISESMOOTHER
#define GPUMESH_ABSTRACTVERTEXWISESMOOTHER

#include "../AbstractSmoother.h"


class AbstractVertexWiseSmoother : public AbstractSmoother
{
protected:
    AbstractVertexWiseSmoother(const std::vector<std::string>& smoothShaders);

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
    virtual void initializeProgram(
            Mesh& mesh,
            AbstractEvaluator& evaluator) override;

    virtual void smoothVertices(
            Mesh& mesh,
            AbstractEvaluator& evaluator,
            size_t first,
            size_t last,
            bool synchronize) = 0;


private:
    bool _initialized;
    std::string _modelBoundsShader;
    std::string _shapeMeasureShader;
    std::vector<std::string> _smoothShaders;
    cellar::GlProgram _smoothingProgram;

    static const size_t WORKGROUP_SIZE;
};

#endif // GPUMESH_ABSTRACTVERTEXWISESMOOTHER
