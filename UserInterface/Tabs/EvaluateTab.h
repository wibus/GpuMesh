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
    void shapeMeasureTypeChanged(const QString& type);
    void evaluateMesh();

    void benchmarkImplementations();

protected:

    void deployShapeMeasures();
    void deployImplementations();

private:
    Ui::MainWindow* _ui;
    std::shared_ptr<GpuMeshCharacter> _character;
    std::map<std::string, int> _cycleCounts;
};

#endif // GPUMESH_EVALUATETAB
