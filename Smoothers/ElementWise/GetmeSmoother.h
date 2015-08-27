#ifndef GPUMESH_GETMESMOOTHER
#define GPUMESH_GETMESMOOTHER


#include "AbstractElementWiseSmoother.h"


class IVertexAccum
{
public:
    virtual ~IVertexAccum() {}
    virtual void add(const glm::dvec3 pos, double weight) = 0;
    virtual bool assignAverage(glm::dvec3& pos) const = 0;
};


class GetmeSmoother : public AbstractElementWiseSmoother
{
public:
    GetmeSmoother();
    virtual ~GetmeSmoother();


    virtual void smoothMeshSerial(
            Mesh& mesh,
            AbstractEvaluator& evaluator) override;

    virtual void smoothMeshThread(
            Mesh& mesh,
            AbstractEvaluator& evaluator) override;

protected:    
    virtual void updateVertexPositions(
            Mesh& mesh,
            AbstractEvaluator& evaluator) override;

    virtual void smoothTets(
            Mesh& mesh,
            AbstractEvaluator& evaluator,
            size_t first,
            size_t last,
            bool synchronize) override;

    virtual void smoothPris(
            Mesh& mesh,
            AbstractEvaluator& evaluator,
            size_t first,
            size_t last,
            bool synchronize) override;

    virtual void smoothHexs(
            Mesh& mesh,
            AbstractEvaluator& evaluator,
            size_t first,
            size_t last,
            bool synchronize) override;

private:
    double _lambda;
    IVertexAccum** _vertexAccums;
};

#endif // GPUMESH_GETMESMOOTHER
