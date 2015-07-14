#include "RenderingTab.h"

#include "GpuMeshCharacter.h"
#include "ui_MainWindow.h"

using namespace std;


RenderingTab::RenderingTab(Ui::MainWindow* ui,
                           const std::shared_ptr<GpuMeshCharacter>& character) :
    _ui(ui),
    _character(character)
{
    // Defining renderers
    deployRenderTypes();
    connect(_ui->renderingTypeMenu,
            static_cast<void(QComboBox::*)(const QString&)>(&QComboBox::currentIndexChanged),
            this, &RenderingTab::renderTypeChanged);


    // Defining shadings
    deployShadings();
    connect(_ui->shadingMenu,
            static_cast<void(QComboBox::*)(const QString&)>(&QComboBox::currentIndexChanged),
            this, &RenderingTab::shadingChanged);
    _ui->shadingMenu->setCurrentText("Wireframe");


    // Defining visual quality measures
    deployShapeMeasures();
    connect(_ui->visualShapeMeasureMenu,
            static_cast<void(QComboBox::*)(const QString&)>(&QComboBox::currentIndexChanged),
            this, &RenderingTab::shapeMeasureChanged);


    // Define virtual cut plane usage
    _character->useVirtualCutPlane(_ui->virtualCutPlaneCheck->isChecked());
    connect(_ui->virtualCutPlaneCheck, &QCheckBox::stateChanged,
            this, &RenderingTab::useVirtualCutPlane);
}

RenderingTab::~RenderingTab()
{

}

void RenderingTab::deployRenderTypes()
{
    vector<string> rendererNames = _character->availableRenderers();

    _ui->renderingTypeMenu->clear();
    for(const auto& name : rendererNames)
        _ui->renderingTypeMenu->addItem(QString(name.c_str()));

    size_t defaultRenderer = 0;
    _ui->renderingTypeMenu->setCurrentIndex(defaultRenderer);
    // Renderer must be set here, or no renderer will be installed
    _character->useRenderer(rendererNames[defaultRenderer]);
}

void RenderingTab::deployShadings()
{
    vector<string> shadingNames = _character->availableShadings();

    _ui->shadingMenu->clear();
    for(const auto& name : shadingNames)
        _ui->shadingMenu->addItem(QString(name.c_str()));

    size_t defaultShading = 0;
    _ui->shadingMenu->setCurrentIndex(defaultShading);
    // Shading must be set here, or no shading will be installed
    _character->useShading(shadingNames[defaultShading]);
}

void RenderingTab::deployShapeMeasures()
{
    vector<string> evaluatorNames = _character->availableEvaluators();

    _ui->visualShapeMeasureMenu->clear();
    for(const auto& name : evaluatorNames)
        _ui->visualShapeMeasureMenu->addItem(QString(name.c_str()));

    size_t defaultMeasure = 0;
    _ui->visualShapeMeasureMenu->setCurrentIndex(defaultMeasure);
    // Shading must be set here, or no shading will be installed
    _character->displayQuality(evaluatorNames[defaultMeasure]);
}

void RenderingTab::renderTypeChanged(QString text)
{
    _character->useRenderer(text.toStdString());

    deployShadings();
    _character->useVirtualCutPlane(
        _ui->virtualCutPlaneCheck->isChecked());
}

void RenderingTab::shadingChanged(QString text)
{
    if(text.length() != 0)
    {
        _character->useShading(text.toStdString());
    }
}

void RenderingTab::shapeMeasureChanged(QString text)
{
    _character->displayQuality(text.toStdString());
}

void RenderingTab::useVirtualCutPlane(bool checked)
{
    _character->useVirtualCutPlane(checked);
}
