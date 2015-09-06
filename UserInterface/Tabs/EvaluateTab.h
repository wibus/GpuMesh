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

protected:
    virtual void deployShapeMeasures();
    virtual void deployImplementations();

private:
    Ui::MainWindow* _ui;
    std::shared_ptr<GpuMeshCharacter> _character;
    std::map<std::string, int> _cycleCounts;
    std::string _lastImpl;
};

#endif // GPUMESH_EVALUATETAB