#ifndef GPUMESH_DUMMYDISCRETIZER
#define GPUMESH_DUMMYDISCRETIZER

#include "AbstractDiscretizer.h"


class DummyDiscretizer : public AbstractDiscretizer
{
public:
    DummyDiscretizer();
    virtual ~DummyDiscretizer();

    virtual void discretize(
            const Mesh& mesh,
            const glm::ivec3& gridSize) override;

    virtual Metric metric(
            const glm::dvec3& position) const override;

    virtual double distance(
            const glm::dvec3& a,
            const glm::dvec3& b) const override;


    virtual void installPlugIn(
            const Mesh& mesh,
            cellar::GlProgram& program) const override;

    virtual void uploadPlugInUniforms(
            const Mesh& mesh,
            cellar::GlProgram& program) const override;

    virtual void releaseDebugMesh() override;
    virtual std::shared_ptr<Mesh> debugMesh() override;


private:
    std::shared_ptr<Mesh> _debugMesh;
};

#endif // GPUMESH_DUMMYDISCRETIZER
