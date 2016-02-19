#ifndef GPUMESH_MESHCREW
#define GPUMESH_MESHCREW

#include <memory>

#include "OptionMap.h"

namespace cellar
{
    class GlProgram;
}

class Mesh;
class AbstractSampler;
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

    AbstractSampler& sampler();
    AbstractEvaluator& evaluator();
    AbstractMeasurer& measurer();

    const AbstractSampler& sampler() const;
    const AbstractEvaluator& evaluator() const;
    const AbstractMeasurer& measurer() const;

    void initialize(const Mesh& mesh);
    void terminate();
    bool initialized() const;

    void setSampler(const Mesh& mesh, const std::shared_ptr<AbstractSampler>& sampler);
    void setEvaluator(const Mesh& mesh, const std::shared_ptr<AbstractEvaluator>& evaluator);

    void installPlugins(const Mesh& mesh, cellar::GlProgram& program) const;
    void setPluginUniforms(const Mesh& mesh, cellar::GlProgram& program) const;
    void setupPluginExecution(const Mesh& mesh, const cellar::GlProgram& program) const;


private:
    void reinitCrew(const Mesh& mesh);

    OptionMap<std::shared_ptr<AbstractMeasurer>> _availableMeasurers;

    std::shared_ptr<AbstractSampler> _sampler;
    std::shared_ptr<AbstractEvaluator> _evaluator;
    std::shared_ptr<AbstractMeasurer> _measurer;
    bool _isInitialized;
};

#endif // GPUMESH_MESHCREW
