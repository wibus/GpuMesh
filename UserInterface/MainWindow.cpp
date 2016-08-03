#include "MainWindow.h"
#include "ui_MainWindow.h"

#include <QApplication>
#include <QDesktopWidget>

#include "Tabs/MeshTab.h"
#include "Tabs/EvaluateTab.h"
#include "Tabs/OptimizeTab.h"
#include "Tabs/RenderTab.h"

using namespace scaena;


MainWindow::MainWindow(const std::shared_ptr<scaena::Play>& play,
                       const std::shared_ptr<GpuMeshCharacter>& character) :
    QMainWindow(),
    _ui(new Ui::MainWindow),
    _play(play),
    _view(new QGlWidgetView("MainView", this), [](QGlWidgetView*){})
{
    _ui->setupUi(this);
    _ui->horizontalLayout->addWidget(_view.get());

    _play->addView(_view);

    _meshTab.reset(new MeshTab(_ui, character));
    _evaluateTab.reset(new EvaluateTab(_ui, character));
    _optimizeTab.reset(new OptimizeTab(_ui, character));
    _renderTab.reset(new RenderTab(_ui, character));

    resize(1920, 1080);
    move(0, 0);
}

MainWindow::~MainWindow()
{
    delete _ui;
}
