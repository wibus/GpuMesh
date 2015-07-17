#ifndef GPUMESH_SMOOTHINGTAB
#define GPUMESH_SMOOTHINGTAB

#include <memory>

#include <QWidget>

class GpuMeshCharacter;

namespace Ui
{
    class MainWindow;
}


class SmoothingTab : public QObject
{
    Q_OBJECT

public:
    SmoothingTab(Ui::MainWindow* ui, const std::shared_ptr<GpuMeshCharacter>& character);
    virtual ~SmoothingTab();


protected slots:
    virtual void techniqueChanged(const QString&);
    virtual void smoothMesh();


protected:
    virtual void deployTechniques();
    virtual void deployShapeMeasures();
    virtual void deployImplementations();


private:
    Ui::MainWindow* _ui;
    std::shared_ptr<GpuMeshCharacter> _character;
};

#endif // GPUMESH_SMOOTHINGTAB
