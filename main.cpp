#include <exception>
#include <iostream>
using namespace std;

#include <CellarWorkbench/Misc/Log.h>
using namespace cellar;

#include <Scaena/ScaenaApplication/Application.h>
#include <Scaena/Play/Play.h>
#include <Scaena/Play/Act.h>
using namespace scaena;

#include "GpuMeshCharacter.h"
#include "Meshers/AbstractMesher.h"
#include "Evaluators/AbstractEvaluator.h"
#include "Renderers/AbstractRenderer.h"
#include "Smoothers/AbstractSmoother.h"
#include "UserInterface/MainWindow.h"


int main(int argc, char** argv) try
{
    getLog().setOuput(std::cout);
    getApplication().init(argc, argv);
    
    std::shared_ptr<GpuMeshCharacter> character(new GpuMeshCharacter());
    std::shared_ptr<Act> act(new Act("MainAct"));
    act->addCharacter(character);
    
    std::shared_ptr<Play> play(new Play("GpuMesh"));
    play->appendAct(act);

    std::shared_ptr<MainWindow> window(new MainWindow(play, character));
    window->show();

    getApplication().setPlay(play);
    return getApplication().execute();    
}
catch(exception& e)
{
    cerr << "Exception caught : " << e.what() << endl;
}
catch(...)
{
    cerr << "Exception passed through.." << endl;
    throw;
}
