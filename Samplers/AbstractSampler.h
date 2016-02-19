#ifndef GPUMESH_ABSTRACTSAMPLER
#define GPUMESH_ABSTRACTSAMPLER

#include <memory>
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
struct MeshVert;

typedef glm::dmat3 Metric;


class AbstractSampler
{
protected:
    typedef void (*installCudaFct)(void);
    AbstractSampler(const std::string& name,
                        const std::string& shader,
                        const installCudaFct installCuda);

public:
    virtual ~AbstractSampler();


    virtual bool isMetricWise() const = 0;

    virtual void initialize();


    // GLSL Plug-in interface
    virtual std::string samplingShader() const;

    virtual void installPlugin(
            const Mesh& mesh,
            cellar::GlProgram& program) const;

    virtual void setPluginUniforms(
            const Mesh& mesh,
            cellar::GlProgram& program) const;

    virtual void setupPluginExecution(
            const Mesh& mesh,
            const cellar::GlProgram& program) const;


    virtual void setReferenceMesh(
            const Mesh& mesh,
            int density) = 0;

    virtual Metric metricAt(
            const glm::dvec3& position) const = 0;


    // Debug mesh
    virtual void releaseDebugMesh() = 0;
    virtual const Mesh& debugMesh() = 0;


protected:
    // Give mesh's provided metric
    Metric vertMetric(const Mesh& mesh, unsigned int vId) const;
    Metric vertMetric(const glm::dvec3& position) const;

    // Interpolate the metric given two samples and a mix ratio
    Metric interpolateMetrics(const Metric& m1, const Metric& m2, double a) const;

    // Classic bounding box computation
    void boundingBox(const Mesh& mesh,
                     glm::dvec3& minBounds,
                     glm::dvec3& maxBounds) const;


    static bool tetParams(
            const std::vector<MeshVert>& verts,
            const MeshTet& tet,
            const glm::dvec3& p,
            double coor[4]);

private:
    std::string _samplingName;
    std::string _samplingShader;
    std::string _baseShader;
    installCudaFct _installCuda;
};

#endif // GPUMESH_ABSTRACTSAMPLER
