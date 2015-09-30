#ifndef GpuMesh_CHARACTER
#define GpuMesh_CHARACTER

#include <vector>

#include <CellarWorkbench/Camera/CameraManFree.h>

#include <PropRoom2D/Prop/Hud/TextHud.h>
#include <PropRoom2D/Prop/Hud/ImageHud.h>

#include <Scaena/Play/Character.h>

#include "DataStructures/OptionMap.h"
#include "DataStructures/OptimizationPlot.h"

class Mesh;
class AbstractMesher;
class AbstractEvaluator;
class AbstractSmoother;
class AbstractRenderer;
class AbstractSerializer;
class AbstractDeserializer;
enum class ECutType;

enum class ECameraMan
{
    Sphere,
    Free
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


    virtual OptionMapDetails availableMeshers() const;
    virtual OptionMapDetails availableMeshModels(const std::string& mesherName) const;
    virtual OptionMapDetails availableEvaluators() const;
    virtual OptionMapDetails availableEvaluatorImplementations(const std::string& evaluatorName) const;
    virtual OptionMapDetails availableSmoothers() const;
    virtual OptionMapDetails availableSmootherImplementations(const std::string& smootherName) const;
    virtual OptionMapDetails availableRenderers() const;
    virtual OptionMapDetails availableShadings() const;
    virtual OptionMapDetails availableCameraMen() const;
    virtual OptionMapDetails availableCutTypes() const;

    // Mesh
    virtual void generateMesh(
            const std::string& mesherName,
            const std::string& modelName,
            size_t vertexCount);

    virtual void clearMesh();

    virtual void saveMesh(
            const std::string& fileName);

    virtual void loadMesh(
            const std::string& fileName);

    // Evaluate
    virtual void evaluateMesh(
            const std::string& evaluatorName,
            const std::string& implementationName);

    virtual void benchmarkEvaluator(
            const std::string& evaluatorName,
            const std::map<std::string, int>& cycleCounts);

    // Smooth
    virtual void smoothMesh(
            const std::string& smootherName,
            const std::string& evaluatorName,
            const std::string& implementationName,
            size_t minIterationCount,
            double moveFactor,
            double gainThreshold);

    virtual OptimizationPlot benchmarkSmoother(
            const std::string& smootherName,
            const std::string& evaluatorName,
            const std::map<std::string, bool>& activeImpls,
            size_t minIterationCount,
            double moveFactor,
            double gainThreshold);

    // Render
    virtual void useEvaluator(const std::string& evaluatorName);
    virtual void useRenderer(const std::string& rendererName);
    virtual void useShading(const std::string& shadingName);
    virtual void useCameraMan(const std::string& cameraManName);
    virtual void useCutType(const std::string& cutTypeName);
    virtual void setElementVisibility(bool tet, bool pri, bool hex);
    virtual void setQualityCullingBounds(double min, double max);

protected:
    virtual void printStep(const std::string& stepDescription);

    virtual void refreshCamera();
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
    ECameraMan _cameraMan;
    ECutType _cutType;

    float _lightAzimuth;
    float _lightAltitude;
    float _lightDistance;

    double _cutAzimuth;
    double _cutAltitude;
    double _cutDistance;

    bool _tetVisibility;
    bool _priVisibility;
    bool _hexVisibility;

    double _qualityCullingMin;
    double _qualityCullingMax;

private:
    std::unique_ptr<Mesh> _mesh;
    std::shared_ptr<AbstractRenderer> _renderer;
    std::shared_ptr<AbstractEvaluator> _evaluator;
    std::shared_ptr<cellar::CameraManFree> _cameraManFree;

    static const glm::vec3 nullVec;
    static const glm::vec3 upVec;

    std::shared_ptr<prop2::TextHud> _fps;
    std::shared_ptr<prop2::TextHud> _ups;
    std::shared_ptr<prop2::TextHud> _qualityTitle;
    std::shared_ptr<prop2::TextHud> _qualityNeg;
    std::shared_ptr<prop2::TextHud> _qualityMin;
    std::shared_ptr<prop2::TextHud> _qualityMax;
    std::shared_ptr<prop2::ImageHud> _qualityLut;

    OptionMap<std::shared_ptr<AbstractMesher>> _availableMeshers;
    OptionMap<std::shared_ptr<AbstractEvaluator>> _availableEvaluators;
    OptionMap<std::shared_ptr<AbstractSmoother>> _availableSmoothers;
    OptionMap<std::shared_ptr<AbstractRenderer>> _availableRenderers;
    OptionMap<std::shared_ptr<AbstractSerializer>> _availableSerializers;
    OptionMap<std::shared_ptr<AbstractDeserializer>> _availableDeserializers;
    OptionMap<ECameraMan> _availableCameraMen;
    OptionMap<ECutType> _availableCutTypes;
};

#endif //GpuMesh_CHARACTER
