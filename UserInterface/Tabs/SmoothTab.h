#ifndef GPUMESH_SMOOTHTAB
#define GPUMESH_SMOOTHTAB

#include <memory>

#include <QWidget>

class GpuMeshCharacter;

namespace Ui
{
    class MainWindow;
}


class SmoothTab : public QObject
{
    Q_OBJECT

public:
    SmoothTab(Ui::MainWindow* ui, const std::shared_ptr<GpuMeshCharacter>& character);
    virtual ~SmoothTab();


protected slots:
    virtual void techniqueChanged(const QString&);
    virtual void smoothMesh();

    virtual void benchmarkImplementations();

protected:
    virtual void deployTechniques();
    virtual void deployImplementations();


private:
    Ui::MainWindow* _ui;
    std::shared_ptr<GpuMeshCharacter> _character;
};

#endif // GPUMESH_SMOOTHTAB
