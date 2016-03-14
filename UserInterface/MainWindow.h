#ifndef GPUMESH_MAINWINDOW
#define GPUMESH_MAINWINDOW

#include <QMainWindow>

#include <Scaena/Play/Play.h>
#include <Scaena/Play/Character.h>
#include <Scaena/ScaenaApplication/QGlWidgetView.h>

namespace Ui
{
    class MainWindow;
}

class MeshTab;
class EvaluateTab;
class SmoothTab;
class RenderTab;
class GpuMeshCharacter;


class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(const std::shared_ptr<scaena::Play>& play,
               const std::shared_ptr<GpuMeshCharacter>& character);
    ~MainWindow();

public slots:
    void aboutToQuitSlot();

private:
    Ui::MainWindow *_ui;
    std::shared_ptr<scaena::Play> _play;
    std::shared_ptr<scaena::QGlWidgetView> _view;
    std::shared_ptr<MeshTab> _meshTab;
    std::shared_ptr<EvaluateTab> _evaluateTab;
    std::shared_ptr<SmoothTab> _smoothTab;
    std::shared_ptr<RenderTab> _renderTab;
};

#endif // GPUMESH_MAINWINDOW_H
