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
class AbstractSampler;


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


    // Distances
    virtual double riemannianDistance(
            const AbstractSampler& sampler,
            const glm::dvec3& a,
            const glm::dvec3& b,
            uint& cachedRefTet) const = 0;

    virtual glm::dvec3 riemannianSegment(
            const AbstractSampler& sampler,
            const glm::dvec3& a,
            const glm::dvec3& b,
            uint& cachedRefTet) const = 0;


    // Volumes    
    static double tetEuclideanVolume(
            const Mesh& mesh,
            const MeshTet& tet);
    static double tetEuclideanVolume(
            const glm::dvec3 vp[]);

    virtual double tetVolume(
            const Mesh& mesh,
            const AbstractSampler& sampler,
            const MeshTet& tet) const;
    virtual double tetVolume(
            const AbstractSampler& sampler,
            const glm::dvec3 vp[],
            const MeshTet& tet) const = 0;

    virtual double priVolume(
            const Mesh& mesh,
            const AbstractSampler& sampler,
            const MeshPri& pri) const;
    virtual double priVolume(
            const AbstractSampler& sampler,
            const glm::dvec3 vp[],
            const MeshPri& pri) const = 0;

    virtual double hexVolume(
            const Mesh& mesh,
            const AbstractSampler& sampler,
            const MeshHex& hex) const;
    virtual double hexVolume(
            const AbstractSampler& sampler,
            const glm::dvec3 vp[],
            const MeshHex& hex) const = 0;



    virtual double computeLocalElementSize(
            const Mesh& mesh,
            uint vId) const;


    virtual glm::dvec3 computeVertexEquilibrium(
            const Mesh& mesh,
            const AbstractSampler& sampler,
            uint vId) const = 0;

private:
    std::string _measureName;
    std::string _measureShader;
    installCudaFct _installCuda;
};

#endif // GPUMESH_ABSTRACTMEASURER
