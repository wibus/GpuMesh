#ifndef GPUMESH_ABSTRACTEVALUATOR
#define GPUMESH_ABSTRACTEVALUATOR

#include <CellarWorkbench/GL/GlProgram.h>

#include "DataStructures/Mesh.h"
#include "DataStructures/OptionMap.h"

class AbstractDiscretizer;
class AbstractMeasurer;


class AbstractEvaluator
{
public:
    AbstractEvaluator(const std::string& shapeMeasuresShader);
    virtual ~AbstractEvaluator();


    virtual std::string evaluationShader() const;

    virtual OptionMapDetails availableImplementations() const;

    virtual void initialize(
            const Mesh& mesh,
            const AbstractDiscretizer& discretizer,
            const AbstractMeasurer& measurer);


    // GLSL Plug-in interface
    virtual void installPlugIn(
            const Mesh& mesh,
            cellar::GlProgram& program) const;

    virtual void uploadUniforms(
            const Mesh& mesh,
            cellar::GlProgram& program) const;


    virtual double tetQuality(
            const Mesh& mesh,
            const AbstractDiscretizer& discretizer,
            const AbstractMeasurer& measurer,
            const MeshTet& tet) const;
    virtual double tetQuality(
            const AbstractDiscretizer& discretizer,
            const AbstractMeasurer& measurer,
            const glm::dvec3 vp[]) const = 0;

    virtual double priQuality(
            const Mesh& mesh,
            const AbstractDiscretizer& discretizer,
            const AbstractMeasurer& measurer,
            const MeshPri& pri) const;
    virtual double priQuality(
            const AbstractDiscretizer& discretizer,
            const AbstractMeasurer& measurer,
            const glm::dvec3 vp[]) const = 0;

    virtual double hexQuality(
            const Mesh& mesh,
            const AbstractDiscretizer& discretizer,
            const AbstractMeasurer& measurer,
            const MeshHex& hex) const;
    virtual double hexQuality(
            const AbstractDiscretizer& discretizer,
            const AbstractMeasurer& measurer,
            const glm::dvec3 vp[]) const = 0;

    virtual double patchQuality(
            const Mesh& mesh,
            const AbstractDiscretizer& discretizer,
            const AbstractMeasurer& measurer,
            size_t vId) const;


    virtual bool assessMeasureValidy(
            const AbstractDiscretizer& discretizer,
            const AbstractMeasurer& measurer);

    virtual void evaluateMesh(
            const Mesh& mesh,
            const AbstractDiscretizer& discretizer,
            const AbstractMeasurer& measurer,
            double& minQuality,
            double& qualityMean,
            const std::string& implementationName) const;

    virtual void evaluateMeshQualitySerial(
            const Mesh& mesh,
            const AbstractDiscretizer& discretizer,
            const AbstractMeasurer& measurer,
            double& minQuality,
            double& qualityMean) const;

    virtual void evaluateMeshQualityThread(
            const Mesh& mesh,
            const AbstractDiscretizer& discretizer,
            const AbstractMeasurer& measurer,
            double& minQuality,
            double& qualityMean) const;

    virtual void evaluateMeshQualityGlsl(
            const Mesh& mesh,
            const AbstractDiscretizer& discretizer,
            const AbstractMeasurer& measurer,
            double& minQuality,
            double& qualityMean) const;

    virtual void benchmark(
            const Mesh& mesh,
            const AbstractDiscretizer& discretizer,
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
    std::string _discretizationShader;
    std::string _measureShader;
    std::string _evaluationShader;
    cellar::GlProgram _evaluationProgram;

    typedef std::function<void(const Mesh&,
                               const AbstractDiscretizer&,
                               const AbstractMeasurer&,
                               double&,
                               double&)> ImplementationFunc;
    OptionMap<ImplementationFunc> _implementationFuncs;
};

#endif // GPUMESH_ABSTRACTEVALUATOR
