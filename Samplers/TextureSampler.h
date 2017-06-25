#ifndef GPUMESH_TEXTURESAMPLER
#define GPUMESH_TEXTURESAMPLER

#include <GL3/gl3w.h>

#include "AbstractSampler.h"

class TextureGrid;


class TextureSampler : public AbstractSampler
{
protected:
    TextureSampler(const std::string& name);

public:
    TextureSampler();
    virtual ~TextureSampler();


    virtual bool isMetricWise() const override;

    virtual bool useComputedMetric() const override;


    virtual void setPluginGlslUniforms(
            const Mesh& mesh,
            const cellar::GlProgram& program) const override;

    virtual void setPluginCudaUniforms(
            const Mesh& mesh) const override;


    virtual void updateGlslData(const Mesh& mesh) const override;

    virtual void updateCudaData(const Mesh& mesh) const override;

    virtual void clearGlslMemory(const Mesh& mesh) const override;

    virtual void clearCudaMemory(const Mesh& mesh) const override;


    virtual void updateAnalyticalMetric(
            const Mesh& mesh) override;

    virtual void updateComputedMetric(
            const Mesh& mesh,
            const std::shared_ptr<LocalSampler>& sampler) override;

protected:
    void buildGrid(
            const Mesh& mesh,
            LocalSampler& sampler);


public:
    virtual MeshMetric metricAt(
            const glm::dvec3& position,
            uint& cachedRefTet) const override;


    virtual void releaseDebugMesh() override;
    virtual const Mesh& debugMesh() override;


protected:
    glm::ivec3 cellId(
            const TextureGrid& grid,
            const glm::dvec3& vertPos) const;

    void meshGrid(TextureGrid& grid, Mesh& mesh);


private:
    std::unique_ptr<TextureGrid> _grid;
    std::shared_ptr<Mesh> _debugMesh;
    glm::mat4 _transform;

    mutable GLuint _topLineTex;
    mutable GLuint _sideTriTex;

    std::vector<glm::dvec3> _gaussP;
    std::vector<double> _gaussW;
};

#endif // GPUMESH_TEXTURESAMPLER
