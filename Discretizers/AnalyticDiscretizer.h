#ifndef GPUMESH_ANALYTICDISCRETIZER
#define GPUMESH_ANALYTICDISCRETIZER

#include "AbstractDiscretizer.h"


class AnalyticDiscretizer : public AbstractDiscretizer
{
public:
    AnalyticDiscretizer();
    virtual ~AnalyticDiscretizer();


    virtual bool isMetricWise() const override;


    virtual void discretize(
            const Mesh& mesh,
            int density) override;

    virtual Metric metricAt(
            const glm::dvec3& position) const override;


    virtual void releaseDebugMesh() override;
    virtual const Mesh& debugMesh() override;


protected:


private:
    std::shared_ptr<Mesh> _debugMesh;
};

#endif // GPUMESH_ANALYTICDISCRETIZER
