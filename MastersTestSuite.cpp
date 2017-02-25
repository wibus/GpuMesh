#include "MastersTestSuite.h"

#include <QString>
#include <QCoreApplication>

#include "GpuMeshCharacter.h"

using namespace std;


MastersTestSuite::MastersTestSuite(
        GpuMeshCharacter& character) :
    _character(character),
    _availableMastersTests("Master's tests")
{
    int tId = 0;
    _availableMastersTests.setDefault("N/A");

    _availableMastersTests.setContent({
        {to_string(++tId) + ". Metric Cost (Sphere)",
         MastersTestFunc(bind(&MastersTestSuite::masterMetricCostSphere,    this))},

        {to_string(++tId) + ". Metric Cost (HexGrid)",
         MastersTestFunc(bind(&MastersTestSuite::masterMetricCostHexGrid,   this))},

        {to_string(++tId) + ". Metric Precision",
         MastersTestFunc(bind(&MastersTestSuite::masterMetricPrecision,     this))},
    });

    _translateSampling = {
        {"Analytic",    "Analytique"},
        {"Local",       "Maillage"},
        {"Texture",     "Texture"},
        {"Kd-Tree",     "kD-Tree"},
    };

    _translateImplementations = {
        {"Serial",      "Séquentiel"},
        {"Thread",      "Parallèle"},
        {"GLSL",        "GLSL"},
        {"CUDA",        "CUDA"},
    };
}

MastersTestSuite::~MastersTestSuite()
{

}

OptionMapDetails MastersTestSuite::availableTests() const
{
    return _availableMastersTests.details();
}

string MastersTestSuite::runTests(const vector<string>& tests)
{
    string doc;

    for(const string& test : tests)
    {
        doc += test + "\n\n";

        MastersTestFunc func;
        if(_availableMastersTests.select(test, func))
        {
            doc += func();
        }
        else
        {
            doc += "UNDEFINED TEST";
        }

        doc += "\n\n\n";
    }

    return doc;
}

string MastersTestSuite::masterMetricCost(
        const string& mesher,
        const string& model)
{
    size_t nodeCount = 1e6;
    size_t cycleCount = 5;

    vector<string> sampling = {
        "Analytic",
        "Local",
        "Texture",
        "Kd-Tree"
    };

    vector<string> implement = {
        "Serial",
        "Thread",
        "GLSL",
        "CUDA"
    };

    map<string, int> implCycle;
    for(const auto& i : implement)
        implCycle[i] = cycleCount;


    _character.generateMesh(mesher, model, nodeCount);

    //QCoreApplication::processEvents();

    _character.setMetricScaling(20);
    _character.setMetricAspectRatio(4);


    QString table;
    table += QString("\\begin{tabular}{|l|cccc|}\n ");
    table += QString("\\hline\n");
    table += QString("Métriques\t& \\multicolumn{4}{c|}{Temps (ms)} \\\\\n");
    table += QString("\t& %1 \t& %2\t\t& %3\t\t& %4\\\\\n")
            .arg(_translateImplementations["Serial"],
                 _translateImplementations["Thread"],
                 _translateImplementations["GLSL"],
                 _translateImplementations["CUDA"]);

    table += QString("\\hline\n");

    for(const auto& s : sampling)
    {
        _character.useSampler(s);

        // Process exceptions
        map<string, int> currCycle = implCycle;
        if(model == "HexGrid" && s == "Local")
        {
            currCycle["GLSL"] = 0;
        }

        map<string, double> avgTimes;
        _character.benchmarkEvaluator(avgTimes, "Metric Conformity", currCycle);

        table += QString("%1\t& %2\t\t& %3\t\t& %4\t\t& %5 \\\\ \n")
                .arg(_translateSampling[s],
                     QString::number(avgTimes[implement[0]]),
                     QString::number(avgTimes[implement[1]]),
                     QString::number(avgTimes[implement[2]]),
                     QString::number(avgTimes[implement[3]]));
    }

    table += QString("\\hline\n");


    return table.toStdString();
}

string MastersTestSuite::masterMetricCostSphere()
{
    return masterMetricCost("Delaunay", "Sphere");
}

string MastersTestSuite::masterMetricCostHexGrid()
{
    return masterMetricCost("Debug", "HexGrid");
}

string MastersTestSuite::masterMetricPrecision()
{
    return "Precision";
}
