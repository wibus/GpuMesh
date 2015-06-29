#ifndef GpuMesh_CHARACTER
#define GpuMesh_CHARACTER



#include <PropRoom2D/Prop/Hud/TextHud.h>

#include <Scaena/Play/Character.h>

class Mesh;
class AbstractMesher;
class AbstractEvaluator;
class AbstractSmoother;
class AbstractRenderer;


class GpuMeshCharacter : public scaena::Character
{
public:
    GpuMeshCharacter();

    virtual void enterStage() override;
    virtual void beginStep(const scaena::StageTime &time) override;
    virtual void draw(const std::shared_ptr<scaena::View> &view,
                      const scaena::StageTime &time) override;
    virtual void exitStage() override;

    virtual bool keyPressEvent(const scaena::KeyboardEvent &event) override;


protected:
    virtual void resetPipeline();
    virtual void processPipeline();
    virtual void scheduleSmoothing();
    virtual void printStep(int step, const std::string& stepName);

    virtual void moveCamera(float azimuth, float altitude, float distance);
    virtual void moveLight(float azimuth, float altitude, float distance);
    virtual void moveCutPlane(double azimuth, double altitude, double distance);

protected:
    float _camAzimuth;
    float _camAltitude;
    float _camDistance;

    float _lightAzimuth;
    float _lightAltitude;
    float _lightDistance;

    double _cutAzimuth;
    double _cutAltitude;
    double _cutDistance;

    int _stepId;
    bool _processFinished;

private:
    std::unique_ptr<Mesh> _mesh;
    std::unique_ptr<AbstractMesher> _mesher;
    std::unique_ptr<AbstractSmoother> _smoother;
    std::unique_ptr<AbstractEvaluator> _evaluator;
    std::unique_ptr<AbstractRenderer> _renderer;

    static const glm::vec3 nullVec;
    static const glm::vec3 upVec;

    std::shared_ptr<prop2::TextHud> _fps;
    std::shared_ptr<prop2::TextHud> _ups;
};

#endif //GpuMesh_CHARACTER
