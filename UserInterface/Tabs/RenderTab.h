#ifndef GPUMESH_RENDERTAB
#define GPUMESH_RENDERTAB

#include <memory>

#include <QWidget>

class GpuMeshCharacter;

namespace Ui
{
    class MainWindow;
}


class RenderTab : public QObject
{
    Q_OBJECT

public:
    RenderTab(Ui::MainWindow* ui, const std::shared_ptr<GpuMeshCharacter>& character);
    virtual ~RenderTab();

protected slots:
    virtual void renderTypeChanged(const QString& text);
    virtual void shadingChanged(const QString& text);
    virtual void useCameraMan(const std::string& cameraName);
    virtual void useCutType(const std::string& cutName);
    virtual void elementVisibilityChanged(bool unused);

protected:
    virtual void deployRenderTypes();
    virtual void deployShadings();
    virtual void deployCameraMen();
    virtual void deployCutTypes();

private:
    Ui::MainWindow* _ui;
    std::shared_ptr<GpuMeshCharacter> _character;
};

#endif // GPUMESH_RENDERTAB
