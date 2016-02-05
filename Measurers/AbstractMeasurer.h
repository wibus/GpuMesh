#ifndef GPUMESH_ABSTRACTMEASURER
#define GPUMESH_ABSTRACTMEASURER

#include <string>
#include <vector>

#include <GLM/glm.hpp>

namespace cellar
{
    class GlProgram;
}

class Mesh;
struct MeshTet;
struct MeshPri;
struct MeshHex;
class AbstractEvaluator;
class AbstractDiscretizer;


class AbstractMeasurer
{
protected:
    typedef void (*installCudaFct)(void);
    AbstractMeasurer(const std::string& name,
                     const std::string& shader,
                     const installCudaFct installCuda);

public:
    virtual ~AbstractMeasurer();


    virtual void initialize();


    // GLSL Plug-in interface
    virtual std::string measureShader() const;

    virtual void installPlugin(
            const Mesh& mesh,
            cellar::GlProgram& program) const;

    virtual void setPluginUniforms(
            const Mesh& mesh,
            cellar::GlProgram& program) const;

    virtual void setupPluginExecution(
            const Mesh& mesh,
            const cellar::GlProgram& program) const;


    // Distances
    virtual double riemannianDistance(
            const AbstractDiscretizer& discretizer,
            const glm::dvec3& a,
            const glm::dvec3& b) const = 0;

    virtual glm::dvec3 riemannianSegment(
            const AbstractDiscretizer& discretizer,
            const glm::dvec3& a,
            const glm::dvec3& b) const = 0;


    // Volumes
    virtual double tetVolume(
            const Mesh& mesh,
            const AbstractDiscretizer& discretizer,
            const MeshTet& tet) const;
    virtual double tetVolume(
            const AbstractDiscretizer& discretizer,
            const glm::dvec3 vp[]) const = 0;

    virtual double priVolume(
            const Mesh& mesh,
            const AbstractDiscretizer& discretizer,
            const MeshPri& pri) const;
    virtual double priVolume(
            const AbstractDiscretizer& discretizer,
            const glm::dvec3 vp[]) const = 0;

    virtual double hexVolume(
            const Mesh& mesh,
            const AbstractDiscretizer& discretizer,
            const MeshHex& hex) const;
    virtual double hexVolume(
            const AbstractDiscretizer& discretizer,
            const glm::dvec3 vp[]) const = 0;



    virtual double computeLocalElementSize(
            const Mesh& mesh,
            const AbstractDiscretizer& discretizer,
            size_t vId) const;


    virtual glm::dvec3 computeVertexEquilibrium(
            const Mesh& mesh,
            const AbstractDiscretizer& discretizer,
            size_t vId) const = 0;

private:
    std::string _measureName;
    std::string _measureShader;
    installCudaFct _installCuda;
};

#endif // GPUMESH_ABSTRACTMEASURER
