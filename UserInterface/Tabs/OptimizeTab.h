#ifndef GPUMESH_OPTIMIZETAB
#define GPUMESH_OPTIMIZETAB

#include <memory>

#include <QWidget>
#include <QTextEdit>
class QGraphicsView;
class QGraphicsScene;

#include "DataStructures/Schedule.h"

class GpuMeshCharacter;
class OptimizationPlot;

namespace Ui
{
    class MainWindow;
}


class OptimizeTab : public QObject
{
    Q_OBJECT

public:
    OptimizeTab(Ui::MainWindow* ui, const std::shared_ptr<GpuMeshCharacter>& character);
    virtual ~OptimizeTab();


protected slots:
    virtual void autoPilotToggled(bool checked);
    virtual void minQualThresholdChanged(double threshold);
    virtual void qualMeanThresholdChanged(double threshold);
    virtual void fixedIterationsToggled(bool checked);
    virtual void globalPassCount(int passCount);
    virtual void nodeRelocationPassCount(int passCount);

    virtual void enableTopology(bool checked);
    virtual void topologyPassCount(int count);
    virtual void restructureMesh();

    virtual void techniqueChanged(const QString&);
    virtual void implementationChanged(const QString&);
    virtual void smoothMesh();


    virtual void benchmarkImplementations();

protected:
    virtual void deployTechniques();
    virtual void deployImplementations();

private:
    Ui::MainWindow* _ui;

    Schedule _schedule;
    std::shared_ptr<GpuMeshCharacter> _character;
    std::map<std::string, bool> _activeTechs;
    std::map<std::string, bool> _activeImpls;
    std::string _lastImpl;

    QTextEdit* _reportWidget;
};

#endif // GPUMESH_OPTIMIZETAB
