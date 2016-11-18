#ifndef GPUMESH_TEXTURESAMPLER
#define GPUMESH_TEXTURESAMPLER

#include <GL3/gl3w.h>

#include "AbstractSampler.h"

class TextureGrid;


class TextureSampler : public AbstractSampler
{
public:
    TextureSampler();
    virtual ~TextureSampler();


    virtual bool isMetricWise() const override;


    virtual void setPluginGlslUniforms(
            const Mesh& mesh,
            const cellar::GlProgram& program) const override;

    virtual void setPluginCudaUniforms(
            const Mesh& mesh) const override;


    virtual void updateGlslData(const Mesh& mesh) const override;

    virtual void updateCudaData(const Mesh& mesh) const override;

    virtual void clearGlslMemory(const Mesh& mesh) const override;

    virtual void clearCudaMemory(const Mesh& mesh) const override;


    virtual void setReferenceMesh(
            const Mesh& mesh) override;

    virtual Metric metricAt(
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
};

#endif // GPUMESH_TEXTURESAMPLER
