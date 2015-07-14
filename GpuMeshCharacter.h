#ifndef GpuMesh_CHARACTER
#define GpuMesh_CHARACTER

#include <map>
#include <vector>

#include <PropRoom2D/Prop/Hud/TextHud.h>

#include <Scaena/Play/Character.h>

class Mesh;
class AbstractMesher;
class AbstractEvaluator;
class AbstractSmoother;
class AbstractRenderer;


enum class EMeshType
{
    Delaunay,
    Parametric
};

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


    virtual std::vector<std::string> availableMeshers() const;
    virtual std::vector<std::string> availableMeshModels(const std::string& mesherName) const;
    virtual std::vector<std::string> availableEvaluators() const;
    virtual std::vector<std::string> availableSmoothers() const;
    virtual std::vector<std::string> availableRenderers() const;
    virtual std::vector<std::string> availableShadings() const;

    virtual void generateMesh(
            const std::string& mesherName,
            const std::string& modelName,
            size_t vertexCount);

    virtual void smoothMesh(
            const std::string& smootherName,
            const std::string& evaluatorName,
            const std::string& implementationName,
            size_t minIterationCount,
            double moveFactor,
            double gainThreshold);

    virtual void useRenderer(const std::string& rendererName);
    virtual void useShading(const std::string& shadingName);
    virtual void displayQuality(const std::string& evaluatorName);
    virtual void useVirtualCutPlane(bool use);

protected:
    virtual void printStep(const std::string& stepDescription);

    virtual void setupInstalledRenderer();
    virtual void tearDownInstalledRenderer();
    virtual void installRenderer(const std::shared_ptr<AbstractRenderer>& renderer);
    virtual void moveCamera(float azimuth, float altitude, float distance);
    virtual void moveLight(float azimuth, float altitude, float distance);
    virtual void moveCutPlane(double azimuth, double altitude, double distance);

protected:
    bool _isEntered;

    float _camAzimuth;
    float _camAltitude;
    float _camDistance;

    float _lightAzimuth;
    float _lightAltitude;
    float _lightDistance;

    double _cutAzimuth;
    double _cutAltitude;
    double _cutDistance;

private:
    std::unique_ptr<Mesh> _mesh;
    std::shared_ptr<AbstractRenderer> _renderer;
    std::shared_ptr<AbstractEvaluator> _visualEvaluator;

    static const glm::vec3 nullVec;
    static const glm::vec3 upVec;

    std::shared_ptr<prop2::TextHud> _fps;
    std::shared_ptr<prop2::TextHud> _ups;

    std::map<std::string, std::shared_ptr<AbstractMesher>> _availableMeshers;
    std::map<std::string, std::shared_ptr<AbstractEvaluator>> _availableEvaluators;
    std::map<std::string, std::shared_ptr<AbstractSmoother>> _availableSmoothers;
    std::map<std::string, std::shared_ptr<AbstractRenderer>> _availableRenderers;

};

#endif //GpuMesh_CHARACTER
