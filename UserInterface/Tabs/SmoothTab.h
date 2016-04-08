#ifndef GPUMESH_SMOOTHTAB
#define GPUMESH_SMOOTHTAB

#include <memory>

#include <QWidget>
#include <QTextEdit>
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

    virtual void modifyTopology(bool enable);
    virtual void topoFrequency(int frequency);

    virtual void benchmarkImplementations();

protected:
    virtual void deployTechniques();
    virtual void deployImplementations();

private:
    Ui::MainWindow* _ui;
    std::shared_ptr<GpuMeshCharacter> _character;
    std::map<std::string, bool> _activeImpls;
    std::string _lastImpl;

    QTextEdit* _reportWidget;
};

#endif // GPUMESH_SMOOTHTAB
