#ifndef GPUMESH_MESHTAB
#define GPUMESH_MESHTAB

#include <memory>

#include <QWidget>

class GpuMeshCharacter;

namespace Ui
{
    class MainWindow;
}


class MeshTab : public QObject
{
    Q_OBJECT

public:
    MeshTab(Ui::MainWindow* ui, const std::shared_ptr<GpuMeshCharacter>& character);
    virtual ~MeshTab();

protected slots:
    virtual void techniqueChanged(const QString&);
    virtual void generateMesh();
    virtual void clearMesh();
    virtual void saveMesh();
    virtual void loadMesh();
    virtual void screenshot();

protected:
    virtual void deployTechniques();
    virtual void deployModels();

private:
    Ui::MainWindow* _ui;
    std::shared_ptr<GpuMeshCharacter> _character;
};

#endif // GPUMESH_MESHTAB
