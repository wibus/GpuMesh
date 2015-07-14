#ifndef GPUMESH_GEOMETRYTAB
#define GPUMESH_GEOMETRYTAB

#include <memory>

#include <QWidget>

class GpuMeshCharacter;

namespace Ui
{
    class MainWindow;
}


class GeometryTab : public QObject
{
    Q_OBJECT

public:
    GeometryTab(Ui::MainWindow* ui, const std::shared_ptr<GpuMeshCharacter>& character);
    virtual ~GeometryTab();

protected slots:
    virtual void methodChanged(const QString& methodName);
    virtual void generateMesh(bool checked);

protected:
    virtual void deployMethods();
    virtual void deployModels();

private:
    Ui::MainWindow* _ui;
    std::shared_ptr<GpuMeshCharacter> _character;
};

#endif // GPUMESH_GEOMETRYTAB
