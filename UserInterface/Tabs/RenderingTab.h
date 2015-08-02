#ifndef GPUMESH_RENDERINGTAB
#define GPUMESH_RENDERINGTAB

#include <memory>

#include <QWidget>

class GpuMeshCharacter;

namespace Ui
{
    class MainWindow;
}


class RenderingTab : public QObject
{
    Q_OBJECT

public:
    RenderingTab(Ui::MainWindow* ui, const std::shared_ptr<GpuMeshCharacter>& character);
    virtual ~RenderingTab();

protected slots:
    virtual void renderTypeChanged(const QString& text);
    virtual void shadingChanged(const QString& text);
    virtual void shapeMeasureChanged(const QString& text);
    virtual void useCameraMan(const std::string& cameraName);
    virtual void useCutType(const std::string& cutName);

protected:
    virtual void deployRenderTypes();
    virtual void deployShadings();
    virtual void deployShapeMeasures();
    virtual void deployCameraMen();
    virtual void deployCutTypes();

private:
    Ui::MainWindow* _ui;
    std::shared_ptr<GpuMeshCharacter> _character;
};

#endif // GPUMESH_RENDERINGTAB
