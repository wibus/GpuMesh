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
    virtual void useCameraMan(const QString& text);
    virtual void useVirtualCutPlane(bool checked);

protected:
    virtual void deployRenderTypes();
    virtual void deployShadings();
    virtual void deployShapeMeasures();
    virtual void deployCameraMen();

private:
    Ui::MainWindow* _ui;
    std::shared_ptr<GpuMeshCharacter> _character;
};

#endif // GPUMESH_RENDERINGTAB
