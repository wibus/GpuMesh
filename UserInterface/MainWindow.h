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

class GeometryTab;
class SmoothingTab;
class RenderingTab;
class GpuMeshCharacter;


class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(const std::shared_ptr<scaena::Play>& play,
               const std::shared_ptr<GpuMeshCharacter>& character);
    ~MainWindow();

private:
    Ui::MainWindow *_ui;
    std::shared_ptr<scaena::Play> _play;
    std::shared_ptr<scaena::QGlWidgetView> _view;
    std::shared_ptr<GeometryTab> _geometryTab;
    std::shared_ptr<SmoothingTab> _smoothingTab;
    std::shared_ptr<RenderingTab> _renderingTab;
};

#endif // GPUMESH_MAINWINDOW_H
