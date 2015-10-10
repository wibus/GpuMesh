#ifndef GPUMESH_MESHCREW
#define GPUMESH_MESHCREW

#include <memory>

namespace cellar
{
    class GlProgram;
}

class Mesh;
class AbstractDiscretizer;
class AbstractEvaluator;
class AbstractMeasurer;
class AbstractSmoother;


class MeshCrew
{

public:
    MeshCrew();
    MeshCrew(const MeshCrew& crew) = delete;
    MeshCrew& operator = (const MeshCrew& crew) = delete;
    ~MeshCrew() = default;

    AbstractDiscretizer& discretizer();
    AbstractEvaluator& evaluator();
    AbstractMeasurer& measurer();

    const AbstractDiscretizer& discretizer() const;
    const AbstractEvaluator& evaluator() const;
    const AbstractMeasurer& measurer() const;

    void setDiscretizer(const std::shared_ptr<AbstractDiscretizer>& discretizer);
    void setEvaluator(const std::shared_ptr<AbstractEvaluator>& evaluator);
    void setMeasurer(const std::shared_ptr<AbstractMeasurer>& measurer);

    void installPlugIns(const Mesh& mesh, cellar::GlProgram& program) const;
    void uploadUniforms(const Mesh& mesh, cellar::GlProgram& program) const;

private:
    std::shared_ptr<AbstractDiscretizer> _discretizer;
    std::shared_ptr<AbstractEvaluator> _evaluator;
    std::shared_ptr<AbstractMeasurer> _measurer;
};

#endif // GPUMESH_MESHCREW
