#include "RenderingTab.h"

#include <QRadioButton>

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


    // Defining camera man
    deployCameraMen();


    // Define cut type
    deployCutTypes();
}

RenderingTab::~RenderingTab()
{

}

void RenderingTab::deployRenderTypes()
{
    OptionMapDetails renderers = _character->availableRenderers();

    _ui->renderingTypeMenu->clear();
    for(const auto& name : renderers.options)
        _ui->renderingTypeMenu->addItem(QString(name.c_str()));
    _ui->renderingTypeMenu->setCurrentText(renderers.defaultOption.c_str());

    // Renderer must be set here, or no renderer will be installed
    _character->useRenderer(renderers.defaultOption);
}

void RenderingTab::deployShadings()
{
    OptionMapDetails shadings = _character->availableShadings();

    _ui->shadingMenu->clear();
    for(const auto& name : shadings.options)
        _ui->shadingMenu->addItem(QString(name.c_str()));
    _ui->shadingMenu->setCurrentText(shadings.defaultOption.c_str());

    // Shading must be set here, or no shading will be installed
    _character->useShading(shadings.defaultOption);
}

void RenderingTab::deployShapeMeasures()
{
    OptionMapDetails evaluators = _character->availableEvaluators();

    _ui->visualShapeMeasureMenu->clear();
    for(const auto& name : evaluators.options)
        _ui->visualShapeMeasureMenu->addItem(QString(name.c_str()));
    _ui->visualShapeMeasureMenu->setCurrentText(evaluators.defaultOption.c_str());

    // Quality must be set here, or no shape measure will be installed
    _character->displayQuality(evaluators.defaultOption);
}

void RenderingTab::deployCameraMen()
{
    OptionMapDetails cameraMen = _character->availableCameraMen();

    if(_ui->cameraGroup->layout())
        QWidget().setLayout(_ui->cameraGroup->layout());

    QVBoxLayout* layout = new QVBoxLayout();
    for(const string& cam : cameraMen.options)
    {
        QRadioButton* button = new QRadioButton(cam.c_str());
        connect(button, &QRadioButton::toggled,
                [=](bool checked){ if(checked) useCameraMan(cam);});
        layout->addWidget(button);

        if(cam == cameraMen.defaultOption)
            button->setChecked(true);
    }
    _ui->cameraGroup->setLayout(layout);
}

void RenderingTab::deployCutTypes()
{
    OptionMapDetails cutTypes = _character->availableCutTypes();

    if(_ui->cutGroup->layout())
        QWidget().setLayout(_ui->cutGroup->layout());

    QVBoxLayout* layout = new QVBoxLayout();
    for(const string& cut : cutTypes.options)
    {
        QRadioButton* button = new QRadioButton(cut.c_str());
        connect(button, &QRadioButton::toggled,
                [=](bool checked){ if(checked) useCutType(cut);});
        layout->addWidget(button);

        if(cut == cutTypes.defaultOption)
            button->setChecked(true);
    }
    _ui->cutGroup->setLayout(layout);
}

void RenderingTab::renderTypeChanged(const QString& text)
{
    _character->useRenderer(text.toStdString());

    deployShadings();
}

void RenderingTab::shadingChanged(const QString& text)
{
    if(text.length() != 0)
    {
        _character->useShading(text.toStdString());
    }
}

void RenderingTab::shapeMeasureChanged(const QString& text)
{
    _character->displayQuality(text.toStdString());
}

void RenderingTab::useCameraMan(const string& cameraName)
{
    _character->useCameraMan(cameraName);
}

void RenderingTab::useCutType(const string& cutName)
{
    _character->useCutType(cutName);
}
