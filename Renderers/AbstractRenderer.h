#ifndef GPUMESH_ABSTRACTRENDERER
#define GPUMESH_ABSTRACTRENDERER

#include <functional>

#include <GL3/gl3w.h>

#include <CellarWorkbench/GL/GlProgram.h>
#include <CellarWorkbench/Camera/Camera.h>
#include <CellarWorkbench/DesignPattern/SpecificObserver.h>

#include <Scaena/Play/Play.h>

#include "DataStructures/Mesh.h"
#include "DataStructures/OptionMap.h"

class AbstractEvaluator;


class AbstractRenderer:
        public cellar::SpecificObserver<cellar::CameraMsg>
{
public:
    AbstractRenderer();
    virtual ~AbstractRenderer();

    virtual void setup();
    virtual void tearDown();
    virtual void notifyMeshUpdate();
    virtual void display(const Mesh& mesh, const AbstractEvaluator& evaluator);

    virtual OptionMapDetails availableShadings() const;
    virtual void useShading(const std::string& shadingName);
    virtual void useCutType(const ECutType& cutType);
    virtual void setElementVisibility(bool tet, bool pri, bool hex);
    virtual void setQualityCullingBounds(double min, double max);


    virtual void updateCamera(const glm::mat4& view,
                              const glm::vec3& pos) = 0;
    virtual void updateLight(const glm::mat4& view,
                             const glm::vec3& pos) = 0;
    virtual void updateCutPlane(const glm::dvec4& cutEq) = 0;

    virtual void handleKeyPress(const scaena::KeyboardEvent& event);
    virtual void handleInputs(const scaena::SynchronousKeyboard& keyboard,
                              const scaena::SynchronousMouse& mouse) = 0;


protected:
    virtual void updateBackdropScale(const glm::vec2& scale);
    virtual void drawBackdrop();

    virtual void fullScreenDraw();

    GLuint filterTex() const { return _filterTex;   }
    int filterWidth() const  { return _filterWidth; }
    int filterHeight() const { return _filterHeight;}

    virtual void updateGeometry(
            const Mesh& mesh,
            const AbstractEvaluator& evaluator) = 0;

    virtual void clearResources() = 0;
    virtual void resetResources() = 0;
    virtual void setupShaders() = 0;
    virtual void render() = 0;

    bool _buffNeedUpdate;

    // Cut plane
    ECutType _cutType;
    glm::dvec4 _cutPlaneEq;
    glm::dvec4 _physicalCutPlane;
    glm::dvec4 _virtualCutPlane;

    // Element visibility
    bool _tetVisibility;
    bool _priVisibility;
    bool _hexVisibility;

    // Quality culling
    double _qualityCullingMin;
    double _qualityCullingMax;

    // Shadings
    typedef std::function<void(void)> ShadingFunc;
    OptionMap<ShadingFunc> _shadingFuncs;

private:
    // Backdrop rendering
    cellar::GlProgram _gradientShader;
    GLuint _fullscreenVao;
    GLuint _fullscreenVbo;
    GLuint _filterTex;
    int _filterWidth;
    int _filterHeight;
};

#endif // GPUMESH_ABSTRACTRENDERER
