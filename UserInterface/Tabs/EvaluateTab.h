#ifndef GPUMESH_EVALUATETAB
#define GPUMESH_EVALUATETAB

#include <memory>

#include <QWidget>

class GpuMeshCharacter;

namespace Ui
{
    class MainWindow;
}


class EvaluateTab : public QObject
{
    Q_OBJECT

public:
    EvaluateTab(Ui::MainWindow* ui, const std::shared_ptr<GpuMeshCharacter>& character);
    virtual ~EvaluateTab();

protected slots:
    virtual void shapeMeasureTypeChanged(const QString& type);
    virtual void ImplementationChanged(const QString&);

    virtual void evaluateMesh();
    virtual void benchmarkImplementations();

    virtual void enableAnisotropy(bool enabled);

    virtual void scalingChanged(double scaling);
    virtual void samplingTypeChanged(const QString& type);
    virtual void displayDicretizationToggled(bool display);

protected:
    virtual void deployShapeMeasures();
    virtual void deployImplementations();
    virtual void deploySamplings();

private:
    Ui::MainWindow* _ui;
    std::shared_ptr<GpuMeshCharacter> _character;
    std::map<std::string, int> _cycleCounts;
    std::string _lastImpl;
};

#endif // GPUMESH_EVALUATETAB
