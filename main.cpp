#include <exception>
#include <iostream>
using namespace std;

#include <CellarWorkbench/Misc/Log.h>
using namespace cellar;

#include <Scaena/ScaenaApplication/Application.h>
#include <Scaena/ScaenaApplication/QGlWidgetView.h>
#include <Scaena/Play/Play.h>
#include <Scaena/Play/Act.h>
using namespace scaena;

#include "GpuMeshCharacter.h"
#include "Meshers/AbstractMesher.h"
#include "Evaluators/AbstractEvaluator.h"
#include "Renderers/AbstractRenderer.h"
#include "Smoothers/AbstractSmoother.h"

#include "Evaluators/SolidAngleEvaluator.h"


int main(int argc, char** argv) try
{
    getLog().setOuput(std::cout);
    getApplication().init(argc, argv);

    /* Evaluators verification
    Mesh mesh;
    mesh.vert.push_back(glm::dvec3(0, 0, 0));
    mesh.vert.push_back(glm::dvec3(1, 0, 0));
    mesh.vert.push_back(glm::dvec3(0.5, sqrt(3)/2, 0));
    mesh.vert.push_back(glm::dvec3(0.5, sqrt(3)/6, sqrt(2.0/3)));

    mesh.vert.push_back(glm::dvec3(0, 0, 0));
    mesh.vert.push_back(glm::dvec3(1, 0, 0));
    mesh.vert.push_back(glm::dvec3(0, 1, 0));
    mesh.vert.push_back(glm::dvec3(1, 1, 0));
    mesh.vert.push_back(glm::dvec3(0, sqrt(3)/2, sqrt(3)/2));
    mesh.vert.push_back(glm::dvec3(1, sqrt(3)/2, sqrt(3)/2));

    mesh.vert.push_back(glm::dvec3(0, 0, 0));
    mesh.vert.push_back(glm::dvec3(1, 0, 0));
    mesh.vert.push_back(glm::dvec3(0, 1, 0));
    mesh.vert.push_back(glm::dvec3(1, 1, 0));
    mesh.vert.push_back(glm::dvec3(0, 0, 1));
    mesh.vert.push_back(glm::dvec3(1, 0, 1));
    mesh.vert.push_back(glm::dvec3(0, 1, 1));
    mesh.vert.push_back(glm::dvec3(1, 1, 1));

    SolidAngleEvaluator eval;
    double regularTet = eval.tetrahedronQuality(mesh, MeshTet(0, 1, 2, 3));
    cout << "Regular tetrahedron quality: " << regularTet << endl;
    double regularPri = eval.prismQuality(mesh, MeshPri(4, 5, 6, 7, 8, 9));
    cout << "Regular prism quality: " << regularPri << endl;
    double regularHex = eval.hexahedronQuality(mesh, MeshHex(10, 11, 12, 13, 14, 15, 16, 17));
    cout << "Regular hexahedron quality: " << regularHex << endl;
    //*/
    
    std::shared_ptr<Character> character(new GpuMeshCharacter());
    std::shared_ptr<Act> act(new Act("MainAct"));
    act->addCharacter(character);
    
    std::shared_ptr<QGlWidgetView> view(
        new QGlWidgetView("MainView"));
    view->setGlWindowSpace(1280, 720);
    view->centerOnScreen();
    view->show();
    
    std::shared_ptr<Play> play(new Play("GpuMesh"));
    play->appendAct(act);
    play->addView(view);

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
