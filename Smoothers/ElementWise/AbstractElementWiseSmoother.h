#ifndef GPUMESH_ABSTRACTELEMENTWISESMOOTHER
#define GPUMESH_ABSTRACTELEMENTWISESMOOTHER

#include "../AbstractSmoother.h"


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
            AbstractEvaluator& evaluator) = 0;

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

private:
    static const size_t WORKGROUP_SIZE;
};

#endif // GPUMESH_ABSTRACTELEMENTWISESMOOTHER
