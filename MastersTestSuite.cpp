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
         MastersTestFunc(bind(&MastersTestSuite::metricCostSphere,    this))},

        {to_string(++tId) + ". Metric Cost (HexGrid)",
         MastersTestFunc(bind(&MastersTestSuite::metricCostHexGrid,   this))},

        {to_string(++tId) + ". Metric Precision",
         MastersTestFunc(bind(&MastersTestSuite::metricPrecision,     this))},

         {to_string(++tId) + ". Node Order",
          MastersTestFunc(bind(&MastersTestSuite::nodeOrder,          this))},
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

string MastersTestSuite::metricCost(
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

string MastersTestSuite::metricCostSphere()
{
    return metricCost("Delaunay", "Sphere");
}

string MastersTestSuite::metricCostHexGrid()
{
    return metricCost("Debug", "HexGrid");
}

string MastersTestSuite::metricPrecision()
{
    size_t nodeCount = 1e4;
    size_t passCount = 10;
    double metricK = 2.0;
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
    QString histogram = "Initial";
    for(const string& s : sampling)
    {
        minLines.push_back("\t" + _translateSampling[s]);
        meanLines.push_back("\t" + _translateSampling[s]);
        histogram += "\t" + QString(s.c_str()) + "";
    }
    histogram += "\n";

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

        const QualityHistogram& initHist = plot.initialHistogram();

        header += QString("\t& A=%1").arg(ratios[i]);
        minInit += QString("\t& %1").arg(initHist.minimumQuality());
        meanInit += QString("\t& %1").arg(initHist.harmonicMean());
        for(int s=0; s < sampling.size(); ++s)
        {
            const QualityHistogram& hist =
                plot.implementations()[s].finalHistogram;
            minLines[s] += QString("\t& %1").arg(hist.minimumQuality());
            meanLines[s] += QString("\t& %1").arg(hist.harmonicMean());
        }

        if(i == ratios.size()-1)
        {
            for(size_t b=0; b < initHist.bucketCount(); ++b)
            {
                histogram += QString::number(initHist.buckets()[b]);

                for(int s=0; s < sampling.size(); ++s)
                {
                    const QualityHistogram& hist =
                        plot.implementations()[s].finalHistogram;
                    histogram += "\t" + QString::number(hist.buckets()[b]);
                }
                histogram += "\n";
            }
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
        minTable +=  l + "\\\\\n";
    minTable += "\t\\hline\n";
    minTable += "\\end{tabular}\n";

    QString meanTable = header;
    meanTable += meanInit + "\\\\\n";
    for(const QString& l : meanLines)
        meanTable += l + "\\\\\n";
    meanTable += "\t\\hline\n";
    meanTable += "\\end{tabular}\n";

    ofstream histFile;
    histFile.open(RESULTS_PATH + "Metric Precision.csv");
    if(histFile.is_open())
    {
        histFile << histogram.toStdString();
        histFile.close();
    }

    return (minTable + "\n\n" + meanTable + "\n\n" + histogram).toStdString();
}

string MastersTestSuite::nodeOrder()
{
    const string mesher = "Delaunay";
    const string model = "Sphere";
    const int nodeCount = 1E4;

    const string samp = "Analytic";
    const double metricK = 10.0;
    const double metricA = 4.0;

    const string smoother = "Gradient Descent";
    const vector<string> impl = {"Serial", "Thread"};

    Schedule schedule;
    schedule.autoPilotEnabled = false;
    schedule.topoOperationEnabled = false;
    schedule.globalPassCount = 10;


    _character.generateMesh(mesher, model, nodeCount);

    _character.useSampler(samp);
    _character.setMetricScaling(metricK);
    _character.setMetricAspectRatio(metricA);

    _character.saveMesh(MESHES_PATH + "Node Order.json");


    QCoreApplication::processEvents();


    OptimizationPlot plot;
    vector<Configuration> configs;
    for(const auto& i : impl)
        configs.push_back(Configuration{
            samp, smoother, i});

    _character.benchmarkSmoothers(
        plot, schedule, configs);


    QString table;
    table += "\\begin{tabular}{|l|cc|cc|}\n";
    table += "\t\\hline\n";
    table += "\tItérations \t& \\multicolumn{2}{c|}{Minimums} \t& \\multicolumn{2}{c|}{Moyennes} \\\\\n";
    table += "\t\t\t\t& Séquentiel \t& Parallèle \t\t& Séquentiel \t& Parallèle \\\\\n";
    table += "\t\\hline\n";
    for(int p=0; p <= schedule.globalPassCount; ++p)
    {
        const QualityHistogram& histSerial = plot.implementations()[0].passes[p].histogram;
        const QualityHistogram& histParallel = plot.implementations()[1].passes[p].histogram;

        table += QString("\t%1 \t\t\t& %2 \t& %3 \t\t& %4 \t& %5\\\\\n").arg(
                    QString::number(p),
                    QString::number(histSerial.minimumQuality()),
                    QString::number(histParallel.minimumQuality()),
                    QString::number(histSerial.harmonicMean()),
                    QString::number(histParallel.harmonicMean()));
    }
    table += "\t\\hline\n";
    table += "\\end{tabular}\n";

    return table.toStdString();
}
