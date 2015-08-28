#ifndef GPUMESH_ABSTRACTELEMENTWISESMOOTHER
#define GPUMESH_ABSTRACTELEMENTWISESMOOTHER

#include "../AbstractSmoother.h"

class IVertexAccum;


class AbstractElementWiseSmoother : public AbstractSmoother
{
protected:
    AbstractElementWiseSmoother(const std::vector<std::string>& smoothShaders);

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

    virtual void updateVertexPositions(
            Mesh& mesh,
            AbstractEvaluator& evaluator,
            size_t first,
            size_t last);

    virtual void smoothTets(
            Mesh& mesh,
            AbstractEvaluator& evaluator,
            size_t first,
            size_t last,
            bool synchronize) = 0;

    virtual void smoothPris(
            Mesh& mesh,
            AbstractEvaluator& evaluator,
            size_t first,
            size_t last,
            bool synchronize) = 0;

    virtual void smoothHexs(
            Mesh& mesh,
            AbstractEvaluator& evaluator,
            size_t first,
            size_t last,
            bool synchronize) = 0;

protected:
    static const size_t WORKGROUP_SIZE;

    IVertexAccum** _vertexAccums;
};

#endif // GPUMESH_ABSTRACTELEMENTWISESMOOTHER
