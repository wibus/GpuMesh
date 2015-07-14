#include "GeometryTab.h"

#include "GpuMeshCharacter.h"
#include "ui_MainWindow.h"

using namespace std;


GeometryTab::GeometryTab(Ui::MainWindow* ui,
                         const std::shared_ptr<GpuMeshCharacter>& character) :
    _ui(ui),
    _character(character)
{
    deployTechniques();
    connect(_ui->geometryTechniqueMenu, &QComboBox::currentTextChanged,
            this, &GeometryTab::techniqueChanged);

    deployModels();

    connect(_ui->generateMeshButton,
            static_cast<void(QPushButton::*)(bool)>(&QPushButton::clicked),
            this, &GeometryTab::generateMesh);
}

GeometryTab::~GeometryTab()
{

}

void GeometryTab::techniqueChanged(const QString&)
{
    deployModels();
}

void GeometryTab::generateMesh()
{
    _character->generateMesh(
        _ui->geometryTechniqueMenu->currentText().toStdString(),
        _ui->geometryModelMenu->currentText().toStdString(),
        _ui->vertexCountSpin->value());
}

void GeometryTab::deployTechniques()
{
    vector<string> techniqueNames = _character->availableMeshers();

    _ui->geometryTechniqueMenu->clear();
    for(const auto& name : techniqueNames)
        _ui->geometryTechniqueMenu->addItem(QString(name.c_str()));
}

void GeometryTab::deployModels()
{
    string mesher = _ui->geometryTechniqueMenu->currentText().toStdString();
    vector<string> modelNames = _character->availableMeshModels(mesher);

    _ui->geometryModelMenu->clear();
    for(const auto& name : modelNames)
        _ui->geometryModelMenu->addItem(QString(name.c_str()));
}
