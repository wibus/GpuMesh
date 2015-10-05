#ifndef GPUMESH_BLINDRENDERER
#define GPUMESH_BLINDRENDERER

#include "AbstractRenderer.h"


class BlindRenderer : public AbstractRenderer
{
public:
    BlindRenderer();
    virtual ~BlindRenderer();

    virtual void updateCamera(const glm::mat4& view,
                              const glm::vec3& pos) override;
    virtual void updateLight(const glm::mat4& view,
                             const glm::vec3& pos) override;
    virtual void updateCutPlane(const glm::dvec4& cutEq) override;
    virtual void handleKeyPress(const scaena::KeyboardEvent& event) override;
    virtual void handleInputs(const scaena::SynchronousKeyboard& keyboard,
                              const scaena::SynchronousMouse& mouse) override;


protected:
    virtual void updateGeometry(const Mesh& mesh) override;
    virtual void notifyCameraUpdate(cellar::CameraMsg& msg) override;
    virtual void clearResources() override;
    virtual void resetResources() override;
    virtual void setupShaders() override;
    virtual void render() override;

    virtual void useNoShading();
};

#endif // GPUMESH_BLINDRENDERER
