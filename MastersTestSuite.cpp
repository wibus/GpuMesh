#include "MastersTestSuite.h"

#include <fstream>
#include <chrono>

#include <QString>
#include <QCoreApplication>

#include <CellarWorkbench/DateAndTime/Calendar.h>

#include "DataStructures/OptimizationPlot.h"
#include "DataStructures/Schedule.h"

#include "GpuMeshCharacter.h"

using namespace std;


const string RESULTS_PATH = "resources/reports/";
const string MESHES_PATH = "resources/reports/";

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

    cellar::Time duration;
    auto allStart = chrono::high_resolution_clock::now();

    for(const string& test : tests)
    {
        MastersTestFunc func;
        if(_availableMastersTests.select(test, func))
        {
            auto testStart = chrono::high_resolution_clock::now();

            string data = func();

            auto testEnd = chrono::high_resolution_clock::now();
            duration.fromSeconds((testEnd - testStart).count() / (1.0E9));
            QString lengthStr = QString("[duration: %1]\n").arg(duration.toString().c_str());

            string table = lengthStr.toStdString() + test + "\n\n" + data;

            doc += table;

            ofstream dataFile;
            dataFile.open(RESULTS_PATH + test + ".txt", ios_base::trunc);
            if(dataFile.is_open())
            {
                dataFile << table;
                dataFile.close();
            }

            _character.clearMesh();
            QCoreApplication::processEvents();

        }
        else
        {
            doc += "UNDEFINED TEST";
        }

        doc += "\n\n\n";
    }

    auto allEnd = chrono::high_resolution_clock::now();
    duration.fromSeconds((allEnd - allStart).count() / (1.0E9));
    doc += "[Total duration: " + duration.toString() + "]\n";

    return doc;
}

string MastersTestSuite::masterMetricCost(
        const string& mesher,
        const string& model)
{
    size_t nodeCount = 1e6;
    size_t cycleCount = 5;
    double metricK = 20.0;
    double metricA = 4.0;

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

    //_character.saveMesh(MESHES_PATH + "Metric Cost (" + model + ").json");

    //QCoreApplication::processEvents();

    _character.setMetricScaling(metricK);
    _character.setMetricAspectRatio(metricA);


    QString table;
    table += QString("\\begin{tabular}{|l|cccc|}\n");
    table += QString("\t\\hline\n");
    table += QString("\tMétriques\t& \\multicolumn{4}{c|}{Temps (ms)} \\\\\n");
    table += QString("\t\t& %1 \t& %2\t\t& %3\t\t& %4\\\\\n")
            .arg(_translateImplementations["Serial"],
                 _translateImplementations["Thread"],
                 _translateImplementations["GLSL"],
                 _translateImplementations["CUDA"]);

    table += QString("\t\\hline\n");

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

        table += QString("\t%1\t& %2\t\t& %3\t\t& %4\t\t& %5 \\\\ \n")
                .arg(_translateSampling[s],
                     QString::number(avgTimes[implement[0]]),
                     QString::number(avgTimes[implement[1]]),
                     QString::number(avgTimes[implement[2]]),
                     QString::number(avgTimes[implement[3]]));
    }

    table += "\t\\hline\n";
    table += "\\end{tabular}\n";


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
    size_t nodeCount = 1e4;
    size_t passCount = 10;
    double metricK = 10.0;
    string implement = "Thread";
    string smoother = "Gradient Descent";

    vector<string> sampling = {
        "Analytic",
        "Local",
        "Texture",
        "Kd-Tree"
    };

    vector<double> ratios = {
        1, 2, 4, 8, 16
    };

    Schedule schedule;
    schedule.autoPilotEnabled = false;
    schedule.topoOperationEnabled = false;
    schedule.globalPassCount = passCount;

    QString header = "\tMétrique";
    QString minInit = "\t[Initial]";
    QString meanInit = "\t[Initial]";
    vector<QString> minLines;
    vector<QString> meanLines;
    for(const string& s : sampling)
    {
        minLines.push_back(_translateSampling[s]);
        meanLines.push_back(_translateSampling[s]);
    }

    _character.generateMesh("Delaunay", "Sphere", nodeCount);

    QString meshName = (MESHES_PATH + "Metric Precision (A=%1).json").c_str();
    _character.saveMesh(meshName.arg(0).toStdString());

    _character.setMetricScaling(metricK);
    for(int i=0; i < ratios.size(); ++i)
    {
        if(i != 0)
        {
            _character.loadMesh(meshName.arg(ratios[i-1]).toStdString());
        }

        _character.useSampler("Analytic");
        _character.setMetricAspectRatio(ratios[i]);
        _character.restructureMesh(10);

        _character.saveMesh(meshName.arg(ratios[i]).toStdString());


        QCoreApplication::processEvents();

        OptimizationPlot plot;
        vector<Configuration> configs;
        for(const string& samp : sampling)
            configs.push_back(Configuration{
                samp, smoother, implement});

        _character.benchmarkSmoothers(
            plot, schedule, configs);

        header += QString("\t& A=%1").arg(ratios[i]);
        minInit += QString("\t& %1").arg(plot.initialHistogram().minimumQuality());
        meanInit += QString("\t& %1").arg(plot.initialHistogram().harmonicMean());
        for(int s=0; s < sampling.size(); ++s)
        {
            const QualityHistogram& hist =
                plot.implementations()[s].finalHistogram;
            minLines[s] += QString("\t& %1").arg(hist.minimumQuality());
            meanLines[s] += QString("\t& %1").arg(hist.harmonicMean());
        }
    }

    QString col = QString("c").repeated(ratios.size());
    header = QString("\\begin{tabular}{|l|%1|}\n").arg(col) +
             QString("\t\\hline\n") +
             header + "\\\\\n" +
             QString("\t\\hline\n");

    QString minTable = header;
    minTable += minInit + "\\\\\n";
    for(const QString& l : minLines)
        minTable +=  "\t" + l + "\\\\\n";
    minTable += "\t\\hline\n";
    minTable += "\\end{tabular}\n";

    QString meanTable = header;
    meanTable += meanInit + "\\\\\n";
    for(const QString& l : meanLines)
        meanTable += "\t" + l + "\\\\\n";
    meanTable += "\t\\hline\n";
    meanTable += "\\end{tabular}\n";

    return (minTable + "\n\n" + meanTable).toStdString();
}
