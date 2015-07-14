#include "OptimisationTab.h"


OptimisationTab::OptimisationTab(Ui::MainWindow* ui,
                                 const std::shared_ptr<GpuMeshCharacter>& character) :
    _ui(ui),
    _character(character)
{

}

OptimisationTab::~OptimisationTab()
{

}
