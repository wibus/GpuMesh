#include "GeometryTab.h"

#include "GpuMeshCharacter.h"
#include "ui_MainWindow.h"

using namespace std;


GeometryTab::GeometryTab(Ui::MainWindow* ui,
                         const std::shared_ptr<GpuMeshCharacter>& character) :
    _ui(ui),
    _character(character)
{
    deployMethods();
    connect(_ui->geometryMethodMenu, &QComboBox::currentTextChanged,
            this, &GeometryTab::methodChanged);

    deployModels();

    connect(_ui->generateMeshButton, static_cast<void(QPushButton::*)(bool)>(&QPushButton::clicked),
            this, &GeometryTab::generateMesh);
}

GeometryTab::~GeometryTab()
{

}

void GeometryTab::methodChanged(const QString& methodName)
{
    deployModels();
}

void GeometryTab::generateMesh(bool)
{
    _character->generateMesh(
        _ui->geometryMethodMenu->currentText().toStdString(),
        _ui->geometryModelMenu->currentText().toStdString(),
        _ui->vertexCountSpin->value());
}

void GeometryTab::deployMethods()
{
    vector<string> methodNames = _character->availableMeshers();

    _ui->geometryMethodMenu->clear();
    for(const auto& name : methodNames)
        _ui->geometryMethodMenu->addItem(QString(name.c_str()));
}

void GeometryTab::deployModels()
{
    string mesher = _ui->geometryMethodMenu->currentText().toStdString();
    vector<string> modelNames = _character->availableMeshModels(mesher);

    _ui->geometryModelMenu->clear();
    for(const auto& name : modelNames)
        _ui->geometryModelMenu->addItem(QString(name.c_str()));
}
