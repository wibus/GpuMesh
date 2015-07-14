#ifndef GPUMESH_OPTIMISATIONTAB
#define GPUMESH_OPTIMISATIONTAB

#include <memory>

#include <QWidget>

class GpuMeshCharacter;

namespace Ui
{
    class MainWindow;
}


class OptimisationTab : public QObject
{
    Q_OBJECT

public:
    OptimisationTab(Ui::MainWindow* ui, const std::shared_ptr<GpuMeshCharacter>& character);
    virtual ~OptimisationTab();


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

#endif // GPUMESH_OPTIMISATIONTAB
