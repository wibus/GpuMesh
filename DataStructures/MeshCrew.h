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
class AbstractTopologist;


class MeshCrew
{

public:
    MeshCrew();
    MeshCrew(const MeshCrew& crew) = delete;
    MeshCrew& operator = (const MeshCrew& crew) = delete;
    ~MeshCrew() = default;

    AbstractSampler& sampler();
    AbstractMeasurer& measurer();
    AbstractEvaluator& evaluator();
    AbstractTopologist& topologist();

    const AbstractSampler& sampler() const;
    const AbstractMeasurer& measurer() const;
    const AbstractEvaluator& evaluator() const;
    const AbstractTopologist& topologist() const;

    std::shared_ptr<AbstractSampler> samplerPtr() const;

    void initialize(const Mesh& mesh);
    void terminate();
    bool initialized() const;

    void setSampler(const Mesh& mesh, const std::shared_ptr<AbstractSampler>& sampler);
    void setEvaluator(const Mesh& mesh, const std::shared_ptr<AbstractEvaluator>& evaluator);

    void installPlugins(const Mesh& mesh, cellar::GlProgram& program) const;
    void setPluginGlslUniforms(const Mesh& mesh, cellar::GlProgram& program) const;
    void setPluginCudaUniforms(const Mesh& mesh) const;

    void updateGlslData(const Mesh& mesh) const;
    void updateCudaData(const Mesh& mesh) const;

    void clearGlslMemory(const Mesh& mesh) const;
    void clearCudaMemory(const Mesh& mesh) const;


private:
    void reinitCrew(const Mesh& mesh);

    OptionMap<std::shared_ptr<AbstractMeasurer>> _availableMeasurers;

    std::shared_ptr<AbstractSampler> _sampler;
    std::shared_ptr<AbstractMeasurer> _measurer;
    std::shared_ptr<AbstractEvaluator> _evaluator;
    std::shared_ptr<AbstractTopologist> _topologist;
    bool _isInitialized;
};

#endif // GPUMESH_MESHCREW
