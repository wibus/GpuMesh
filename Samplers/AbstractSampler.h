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

class LocalSampler;

typedef glm::dmat3 MeshMetric;


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

    virtual bool useComputedMetric() const = 0;


    double scaling() const;
    double scalingSqr() const;
    double scalingCube() const;
    virtual void setScaling(double scaling);

    double aspectRatio() const;
    virtual void setAspectRatio(double ratio);

    int discretizationDepth() const;
    void setDiscretizationDepth(int depth);


    // GPU Plug-in interface
    virtual std::string samplingShader() const;

    virtual void installPlugin(
            const Mesh& mesh,
            cellar::GlProgram& program) const;

    virtual void setPluginGlslUniforms(
            const Mesh& mesh,
            const cellar::GlProgram& program) const;

    virtual void setPluginCudaUniforms(
            const Mesh& mesh) const;

    virtual void updateGlslData(const Mesh& mesh) const;

    virtual void updateCudaData(const Mesh& mesh) const;

    virtual void clearGlslMemory(const Mesh& mesh) const;

    virtual void clearCudaMemory(const Mesh& mesh) const;


    virtual void updateAnalyticalMetric(
            const Mesh& mesh) = 0;    

    virtual void updateComputedMetric(
            const Mesh& mesh,
            const std::shared_ptr<LocalSampler>& sampler) = 0;

    virtual MeshMetric metricAt(
            const glm::dvec3& position,
            uint& cachedRefTet) const = 0;


    // Debug mesh
    virtual void releaseDebugMesh() = 0;
    virtual const Mesh& debugMesh() = 0;


protected:
    // Give mesh's provided metric
    MeshMetric vertMetric(const Mesh& mesh, unsigned int vId) const;
    MeshMetric vertMetric(const glm::dvec3& position) const;

    // Interpolate the metric given two samples and a mix ratio
    MeshMetric interpolateMetrics(const MeshMetric& m1, const MeshMetric& m2, double a) const;

    // Classic bounding box computation
    void boundingBox(const Mesh& mesh,
                     glm::dvec3& minBounds,
                     glm::dvec3& maxBounds) const;

private:
    double _scaling;
    double _scaling2;
    double _scaling3;
    double _aspectRatio;
    int _discretizationDepth;
    std::string _samplingName;
    std::string _samplingShader;
    std::string _baseShader;
    installCudaFct _installCuda;
};



// IMPLEMENTATION //
inline double AbstractSampler::scaling() const
{
    return _scaling;
}

inline double AbstractSampler::scalingSqr() const
{
    return _scaling2;
}

inline double AbstractSampler::scalingCube() const
{
    return _scaling3;
}

inline double AbstractSampler::aspectRatio() const
{
    return _aspectRatio;
}

inline int AbstractSampler::discretizationDepth() const
{
    return _discretizationDepth;
}

#endif // GPUMESH_ABSTRACTSAMPLER
