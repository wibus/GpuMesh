#include "MeshTab.h"

#include "GpuMeshCharacter.h"
#include "ui_MainWindow.h"

using namespace std;


MeshTab::MeshTab(Ui::MainWindow* ui,
                 const std::shared_ptr<GpuMeshCharacter>& character) :
    _ui(ui),
    _character(character)
{
    deployTechniques();
    connect(_ui->geometryTechniqueMenu, &QComboBox::currentTextChanged,
            this, &MeshTab::techniqueChanged);

    deployModels();

    connect(_ui->generateMeshButton,
            static_cast<void(QPushButton::*)(bool)>(&QPushButton::clicked),
            this, &MeshTab::generateMesh);
}

MeshTab::~MeshTab()
{

}

void MeshTab::techniqueChanged(const QString&)
{
    deployModels();
}

void MeshTab::generateMesh()
{
    _character->generateMesh(
        _ui->geometryTechniqueMenu->currentText().toStdString(),
        _ui->geometryModelMenu->currentText().toStdString(),
        _ui->meshVertexCountSpin->value());
}

void MeshTab::deployTechniques()
{
    OptionMapDetails techniques = _character->availableMeshers();

    _ui->geometryTechniqueMenu->clear();
    for(const auto& name : techniques.options)
        _ui->geometryTechniqueMenu->addItem(QString(name.c_str()));
    _ui->geometryTechniqueMenu->setCurrentText(techniques.defaultOption.c_str());
}

void MeshTab::deployModels()
{
    OptionMapDetails models = _character->availableMeshModels(
        _ui->geometryTechniqueMenu->currentText().toStdString());

    _ui->geometryModelMenu->clear();
    for(const auto& name : models.options)
        _ui->geometryModelMenu->addItem(QString(name.c_str()));
    _ui->geometryModelMenu->setCurrentText(models.defaultOption.c_str());
}
