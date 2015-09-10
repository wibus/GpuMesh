#ifndef GPUMESH_SMOOTHTAB
#define GPUMESH_SMOOTHTAB

#include <memory>

#include <QWidget>
class QGraphicsView;
class QGraphicsScene;

class GpuMeshCharacter;
class OptimizationPlot;

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
    virtual void implementationChanged(const QString&);
    virtual void smoothMesh();

    virtual void benchmarkImplementations();

protected:
    virtual void deployTechniques();
    virtual void deployImplementations();
    virtual void displayOptimizationPlot(const OptimizationPlot& plot);

private:
    Ui::MainWindow* _ui;
    std::shared_ptr<GpuMeshCharacter> _character;
    std::map<std::string, bool> _activeImpls;
    std::string _lastImpl;

    QGraphicsView* _currentView;
    QGraphicsScene* _currentScene;
};

#endif // GPUMESH_SMOOTHTAB
