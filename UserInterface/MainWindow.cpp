#include "MainWindow.h"
#include "ui_MainWindow.h"

#include <QApplication>
#include <QDesktopWidget>

#include "Tabs/GeometryTab.h"
#include "Tabs/OptimisationTab.h"
#include "Tabs/RenderingTab.h"

using namespace scaena;


MainWindow::MainWindow(const std::shared_ptr<scaena::Play>& play,
                       const std::shared_ptr<GpuMeshCharacter>& character) :
    _ui(new Ui::MainWindow),
    _play(play),
    _view(new QGlWidgetView("MainView"))
{
    _ui->setupUi(this);
    _ui->horizontalLayout->addWidget(_view.get());

    _play->addView(_view);

    _geometryTab.reset(new GeometryTab(_ui, character));
    _optimisationTab.reset(new OptimisationTab(_ui, character));
    _renderingTab.reset(new RenderingTab(_ui, character));

    resize(1200, 720);
    QPoint center = QApplication::desktop()->availableGeometry(this).center();
    move(center.x()-width()*0.5, center.y()-height()*0.5);
}

MainWindow::~MainWindow()
{
    delete _ui;
}
