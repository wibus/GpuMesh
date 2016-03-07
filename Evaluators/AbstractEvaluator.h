#ifndef GPUMESH_ABSTRACTEVALUATOR
#define GPUMESH_ABSTRACTEVALUATOR

#include <CellarWorkbench/GL/GlProgram.h>

#include "DataStructures/Mesh.h"
#include "DataStructures/OptionMap.h"

class AbstractSampler;
class AbstractMeasurer;
class MeshCrew;


class AbstractEvaluator
{
protected:
    typedef void (*installCudaFct)(void);
    AbstractEvaluator(const std::string& shapeMeasuresShader,
                      const installCudaFct installCuda);

public:
    virtual ~AbstractEvaluator();


    virtual std::string evaluationShader() const;

    virtual OptionMapDetails availableImplementations() const;

    virtual void initialize(
            const Mesh& mesh,
            const MeshCrew& crew);


    // GLSL Plug-in interface
    virtual void installPlugin(
            const Mesh& mesh,
            cellar::GlProgram& program) const;

    virtual void setPluginUniforms(
            const Mesh& mesh,
            cellar::GlProgram& program) const;

    virtual void setupPluginExecution(
            const Mesh& mesh,
            const cellar::GlProgram& program) const;



    virtual double tetQuality(
            const Mesh& mesh,
            const AbstractSampler& sampler,
            const AbstractMeasurer& measurer,
            const MeshTet& tet) const;
    virtual double tetQuality(
            const AbstractSampler& sampler,
            const AbstractMeasurer& measurer,
            const glm::dvec3 vp[],
            const MeshTet& tet) const = 0;

    virtual double priQuality(
            const Mesh& mesh,
            const AbstractSampler& sampler,
            const AbstractMeasurer& measurer,
            const MeshPri& pri) const;
    virtual double priQuality(
            const AbstractSampler& sampler,
            const AbstractMeasurer& measurer,
            const glm::dvec3 vp[],
            const MeshPri& pri) const = 0;

    virtual double hexQuality(
            const Mesh& mesh,
            const AbstractSampler& sampler,
            const AbstractMeasurer& measurer,
            const MeshHex& hex) const;
    virtual double hexQuality(
            const AbstractSampler& sampler,
            const AbstractMeasurer& measurer,
            const glm::dvec3 vp[],
            const MeshHex& hex) const = 0;

    virtual double patchQuality(
            const Mesh& mesh,
            const AbstractSampler& sampler,
            const AbstractMeasurer& measurer,
            size_t vId) const;


    virtual bool assessMeasureValidy(
            const AbstractSampler& sampler,
            const AbstractMeasurer& measurer);

    virtual void evaluateMesh(
            const Mesh& mesh,
            const AbstractSampler& sampler,
            const AbstractMeasurer& measurer,
            double& minQuality,
            double& qualityMean,
            const std::string& implementationName) const;

    virtual void evaluateMeshQualitySerial(
            const Mesh& mesh,
            const AbstractSampler& sampler,
            const AbstractMeasurer& measurer,
            double& minQuality,
            double& qualityMean) const;

    virtual void evaluateMeshQualityThread(
            const Mesh& mesh,
            const AbstractSampler& sampler,
            const AbstractMeasurer& measurer,
            double& minQuality,
            double& qualityMean) const;

    virtual void evaluateMeshQualityGlsl(
            const Mesh& mesh,
            const AbstractSampler& sampler,
            const AbstractMeasurer& measurer,
            double& minQuality,
            double& qualityMean) const;

    virtual void evaluateMeshQualityCuda(
            const Mesh& mesh,
            const AbstractSampler& sampler,
            const AbstractMeasurer& measurer,
            double& minQuality,
            double& qualityMean) const;

    virtual void benchmark(
            const Mesh& mesh,
            const AbstractSampler& sampler,
            const AbstractMeasurer& measurer,
            const std::map<std::string, int>& cycleCounts);


protected:
    virtual void accumulatePatchQuality(
            double& patchQuality,
            double& patchWeight,
            double elemQuality) const;

    virtual double finalizePatchQuality(
            double patchQuality,
            double patchWeight) const;


    static const size_t WORKGROUP_SIZE;
    static const size_t POLYHEDRON_TYPE_COUNT;
    static const size_t MAX_GROUP_PARTICIPANTS;

    static const double VALIDITY_EPSILON;
    static const double MAX_INTEGER_VALUE;
    static const double MIN_QUALITY_PRECISION_DENOM;
    static const double MAX_QUALITY_VALUE;

    static const glm::dmat3 Fr_TET_INV;
    static const glm::dmat3 Fr_PRI_INV;
    static const glm::dmat3 Fr_HEX_INV;

    GLuint _qualSsbo;
    std::string _samplingShader;
    std::string _measureShader;
    std::string _evaluationShader;
    cellar::GlProgram _evaluationProgram;
    installCudaFct _installCuda;

    typedef std::function<void(const Mesh&,
                               const AbstractSampler&,
                               const AbstractMeasurer&,
                               double&,
                               double&)> ImplementationFunc;
    OptionMap<ImplementationFunc> _implementationFuncs;
};

#endif // GPUMESH_ABSTRACTEVALUATOR
