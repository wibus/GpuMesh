#ifndef GPUMESH_DUMMYDISCRETIZER
#define GPUMESH_DUMMYDISCRETIZER

#include "AbstractDiscretizer.h"


class DummyDiscretizer : public AbstractDiscretizer
{
public:
    DummyDiscretizer();
    virtual ~DummyDiscretizer();


    virtual bool isMetricWise() const;


    virtual void installPlugIn(
            const Mesh& mesh,
            cellar::GlProgram& program) const override;

    virtual void uploadUniforms(
            const Mesh& mesh,
            cellar::GlProgram& program) const override;


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

#endif // GPUMESH_DUMMYDISCRETIZER
