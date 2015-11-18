#ifndef GPUMESH_MESHCREW
#define GPUMESH_MESHCREW

#include <memory>

#include "OptionMap.h"

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

    void initialize(const Mesh& mesh);
    void terminate();
    bool initialized() const;

    void setDiscretizer(const Mesh& mesh, const std::shared_ptr<AbstractDiscretizer>& discretizer);
    void setEvaluator(const Mesh& mesh, const std::shared_ptr<AbstractEvaluator>& evaluator);

    void installPlugins(const Mesh& mesh, cellar::GlProgram& program) const;
    void setPluginUniforms(const Mesh& mesh, cellar::GlProgram& program) const;
    void setupPluginExecution(const Mesh& mesh, const cellar::GlProgram& program) const;


private:
    void reinitCrew(const Mesh& mesh);

    OptionMap<std::shared_ptr<AbstractMeasurer>> _availableMeasurers;

    std::shared_ptr<AbstractDiscretizer> _discretizer;
    std::shared_ptr<AbstractEvaluator> _evaluator;
    std::shared_ptr<AbstractMeasurer> _measurer;
    bool _isInitialized;
};

#endif // GPUMESH_MESHCREW
