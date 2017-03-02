#include "MastersTestSuite.h"

#include <fstream>
#include <chrono>
#include <sstream>

#include <QDir>
#include <QString>
#include <QMargins>
#include <QTextTable>
#include <QTextCursor>
#include <QTextDocument>
#include <QCoreApplication>

#include <CellarWorkbench/DateAndTime/Calendar.h>
#include <CellarWorkbench/DataStructure/Grid2D.h>

#include "DataStructures/OptimizationPlot.h"
#include "DataStructures/Schedule.h"

#include "GpuMeshCharacter.h"

using namespace std;
using namespace cellar;


const string RESULTS_PATH = "resources/reports/";
const string RESULT_LATEX_PATH = RESULTS_PATH + "latex/";
const string RESULT_MESH_PATH = RESULTS_PATH + "mesh/";
const string RESULT_CSV_PATH = RESULTS_PATH + "csv/";


MastersTestSuite::MastersTestSuite(
        GpuMeshCharacter& character) :
    _character(character),
    _availableMastersTests("Master's tests")
{
    using namespace std::placeholders;

    int tId = 0;
    _availableMastersTests.setDefault("N/A");

    _availableMastersTests.setContent({                                          
        {to_string(++tId) + ". Evaluator Block Size",
        MastersTestFunc(bind(&MastersTestSuite::evaluatorBlockSize, this, _1))},

        {to_string(++tId) + ". Metric Cost (Sphere)",
        MastersTestFunc(bind(&MastersTestSuite::metricCostSphere,   this, _1))},

        {to_string(++tId) + ". Metric Cost (HexGrid)",
        MastersTestFunc(bind(&MastersTestSuite::metricCostHexGrid,  this, _1))},

        {to_string(++tId) + ". Metric Precision",
        MastersTestFunc(bind(&MastersTestSuite::metricPrecision,    this, _1))},

        {to_string(++tId) + ". Node Order",
        MastersTestFunc(bind(&MastersTestSuite::nodeOrder,          this, _1))},

        {to_string(++tId) + ". Smoother Block Size",
        MastersTestFunc(bind(&MastersTestSuite::smootherBlockSize,  this, _1))},
    });

    _translateSampling = {
        {"Analytic",    "Analytique"},
        {"Local",       "Maillage"},
        {"Texture",     "Texture"},
        {"Kd-Tree",     "kD-Tree"},
    };

    _translateImplementation = {
        {"Serial",      "Séquentiel"},
        {"Thread",      "Parallèle"},
        {"GLSL",        "GLSL"},
        {"CUDA",        "CUDA"},
    };

    _translateSmoothers = {
        {"Spring Laplace",      "Laplace à ressorts"},
        {"Quality Laplace",     "Laplace de qualité"},
        {"Spawn Search",        "Recherche par grappe"},
        {"Nelder-Mead",         "Nelder-Mead"},
        {"Gradient Descent",    "Descente du gradient"},
        {"Multi Elem GD",       "DG multi-éléments"},
        {"Multi Pos GD",        "DG multi-positions"},
        {"Patch GD",            "DG mlti-axes"},
        {"GETMe",               "GETMe"}
    };


    // Build directory structure
    if(!QDir(RESULTS_PATH.c_str()).exists())
        QDir().mkdir(RESULTS_PATH.c_str());

    if(!QDir(RESULT_LATEX_PATH.c_str()).exists())
        QDir().mkdir(RESULT_LATEX_PATH.c_str());

    if(!QDir(RESULT_MESH_PATH.c_str()).exists())
        QDir().mkdir(RESULT_MESH_PATH.c_str());

    if(!QDir(RESULT_CSV_PATH.c_str()).exists())
        QDir().mkdir(RESULT_CSV_PATH.c_str());
}

MastersTestSuite::~MastersTestSuite()
{

}

OptionMapDetails MastersTestSuite::availableTests() const
{
    return _availableMastersTests.details();
}

void MastersTestSuite::runTests(
        QTextDocument& reportDocument,
        const vector<string>& tests)
{
    reportDocument.clear();
    _reportDocument = &reportDocument;
    QTextCursor cursor(_reportDocument);

    QTextBlockFormat titleFormat;


    cellar::Time duration;
    auto allStart = chrono::high_resolution_clock::now();

    for(const string& test : tests)
    {
        MastersTestFunc func;
        if(_availableMastersTests.select(test, func))
        {
            cursor.movePosition(QTextCursor::End);
            cursor.insertBlock();
            cursor.insertHtml(("<h2>" + test + "</h2>").c_str());

            auto testStart = chrono::high_resolution_clock::now();

            func(test);

            auto testEnd = chrono::high_resolution_clock::now();
            duration.fromSeconds((testEnd - testStart).count() / (1.0E9));

            cursor.movePosition(QTextCursor::End);
            cursor.insertBlock();
            cursor.insertHtml(("[duration: " +
                duration.toString() + "]").c_str());

            _character.clearMesh();
            QCoreApplication::processEvents();

        }
        else
        {
            cursor.movePosition(QTextCursor::End);
            cursor.insertBlock();
            cursor.insertHtml("UNDEFINED TEST");
        }
    }

    auto allEnd = chrono::high_resolution_clock::now();
    duration.fromSeconds((allEnd - allStart).count() / (1.0E9));

    cursor.movePosition(QTextCursor::End);
    cursor.insertBlock();
    cursor.insertHtml(("[Total duration: " +
            duration.toString() + "]").c_str());
}

void MastersTestSuite::saveToFile(
        const std::string& results,
        const std::string& fileName) const
{
    ofstream file;
    file.open(fileName, std::ios_base::trunc);

    if(file.is_open())
    {
        file << results;
        file.close();
    }
}

void MastersTestSuite::output(
        const std::string& title,
        const std::vector<std::string>& header,
        const std::vector<QualityHistogram>& histograms)
{
    saveCsvTable(title, header, histograms);
    saveLatexTable(title, header, histograms);
    saveReportTable(title, header, histograms);
}

void MastersTestSuite::saveCsvTable(
        const std::string& title,
        const std::vector<std::string>& header,
        const std::vector<QualityHistogram>& histograms)
{
    stringstream ss;

    for(int h=0; h < header.size(); ++h)
    {
        if(h != 0)
        {
            ss << ", ";
        }

        ss << header[h];
    }
    ss << "\n";

    size_t bc = histograms.front().bucketCount();

    for(int b=0; b < bc; ++b)
    {
        for(int h=0; h < histograms.size(); ++h)
        {
            const QualityHistogram& hist =
                    histograms[h];

            if(h != 0)
            {
                ss << ", ";
            }

            ss << hist.buckets()[b];
        }

        ss << "\n";
    }
    ss << "\n";

    saveToFile(ss.str(), RESULT_CSV_PATH + title + ".csv");
}

void MastersTestSuite::saveLatexTable(
        const std::string& title,
        const std::vector<std::string>& header,
        const std::vector<QualityHistogram>& histograms)
{
    stringstream ss;

    ss << "\\hline\n";

    for(int h=0; h < header.size(); ++h)
    {
        if(h != 0)
        {
            ss << " \t& ";
        }

        ss << header[h];
    }
    ss << "\\\\\n";

    ss << "\\hline\n";

    size_t bc = histograms.front().bucketCount();

    for(int b=0; b < bc; ++b)
    {
        for(int h=0; h < histograms.size(); ++h)
        {
            const QualityHistogram& hist =
                    histograms[h];

            if(h != 0)
            {
                ss << " \t& ";
            }

            ss << hist.buckets()[b];
        }

        ss << "\\\\\n";
    }
    ss << "\\hline\n";
    ss << "\n";

    saveToFile(ss.str(), RESULT_LATEX_PATH + title + ".txt");
}

void MastersTestSuite::saveReportTable(
        const std::string& title,
        const std::vector<std::string>& header,
        const std::vector<QualityHistogram>& histograms)
{
    QTextCursor cursor(_reportDocument);
    QTextBlockFormat blockFormat;
    QTextTableCell cell;
    QTextTable* table;

    QTextCharFormat tableHeaderFormat;
    tableHeaderFormat.setFontWeight(QFont::Bold);

    QTextTableFormat propertyTableFormat;
    propertyTableFormat.setCellPadding(5.0);
    propertyTableFormat.setBorderStyle(
        QTextFrameFormat::BorderStyle_Solid);

    size_t rc = histograms.size();
    size_t bc = histograms.front().bucketCount();

    cursor.movePosition(QTextCursor::End);
    cursor.insertBlock(blockFormat);
    cursor.insertHtml(("<h3>" + title + "</h3>").c_str());
    table = cursor.insertTable(
        bc + 1, rc, propertyTableFormat);


    for(int h=0; h < header.size(); ++h)
    {
        cell = table->cellAt(0, h);
        QTextCursor cellCursor =
            cell.firstCursorPosition();

        cellCursor.insertText(header[h].c_str(), tableHeaderFormat);
    }

    for(int b=0; b < bc; ++b)
    {
        for(int h=0; h < histograms.size(); ++h)
        {
            const QualityHistogram& hist =
                    histograms[h];

            cell = table->cellAt(b+1, h);
            QTextCursor cellCursor =
                cell.firstCursorPosition();

            cellCursor.insertText(
                QString::number(hist.buckets()[b]));
        }
    }

    cursor.movePosition(QTextCursor::End);
    cursor.insertBlock(blockFormat);
}

void MastersTestSuite::output(
        const std::string& title,
        const std::vector<std::pair<std::string, int>> header,
        const std::vector<std::pair<std::string, int>> subHeader,
        const std::vector<std::string> lineNames,
        const cellar::Grid2D<double>& data)
{
    saveCsvTable(title, header, subHeader, lineNames, data);
    saveLatexTable(title, header, subHeader, lineNames, data);
    saveReportTable(title, header, subHeader, lineNames, data);
}

void MastersTestSuite::saveCsvTable(
        const std::string& title,
        const std::vector<std::pair<std::string, int>> header,
        const std::vector<std::pair<std::string, int>> subHeader,
        const std::vector<std::string> lineNames,
        const cellar::Grid2D<double>& data)
{
    stringstream ss;

    for(int h=0; h < header.size(); ++h)
    {
        int width = header[h].second;
        const string& name = header[h].first;

        if(h != 0)
        {
            ss << ", ";
        }

        ss << name;

        for(int w=1; w < width; ++w)
            ss << ", ";
    }
    ss << "\n";

    if(!subHeader.empty())
    {
        for(int h=0; h < subHeader.size(); ++h)
        {
            int width = subHeader[h].second;
            const string& name = subHeader[h].first;

            if(h != 0)
            {
                ss << ", ";
            }

            ss << name;

            for(int w=1; w < width; ++w)
                ss << ", ";
        }
        ss << "\n";
    }

    for(int l=0; l < lineNames.size(); ++l)
    {
        ss << lineNames[l];

        for(int c=0; c < data.getWidth(); ++c)
        {
            ss << ", " << data[l][c];
        }

        ss << "\n";
    }
    ss << "\n";

    saveToFile(ss.str(), RESULT_CSV_PATH + title + ".csv");
}

void MastersTestSuite::saveLatexTable(
        const std::string& title,
        const std::vector<std::pair<std::string, int>> header,
        const std::vector<std::pair<std::string, int>> subHeader,
        const std::vector<std::string> lineNames,
        const cellar::Grid2D<double>& data)
{
    stringstream ss;

    ss << "\\hline\n";

    for(int h=0; h < header.size(); ++h)
    {
        int width = header[h].second;
        const string& name = header[h].first;

        if(h != 0)
        {
            ss << " \t& ";
        }

        if(width > 1)
        {
            ss << "multicolumn{" << width << "}{c|}{" << name << "}";
        }
        else
        {
            ss << name;
        }
    }
    ss << "\\\\\n";

    if(!subHeader.empty())
    {
        for(int h=0; h < subHeader.size(); ++h)
        {
            int width = subHeader[h].second;
            const string& name = subHeader[h].first;

            if(h != 0)
            {
                ss << " \t& ";
            }

            if(width > 1)
            {
                ss << "multicolumn{" << width << "}{c|}{" << name << "}";
            }
            else
            {
                ss << name;
            }
        }
        ss << "\\\\\n";
    }
    ss << "\\hline\n";

    for(int l=0; l < lineNames.size(); ++l)
    {
        ss << lineNames[l];

        for(int c=0; c < data.getWidth(); ++c)
        {
            ss << " \t& " << data[l][c];
        }

        ss << "\\\\\n";
    }

    ss << "\\hline\n";
    ss << "\n";

    saveToFile(ss.str(), RESULT_LATEX_PATH + title + ".txt");
}

void MastersTestSuite::saveReportTable(
        const std::string& title,
        const std::vector<std::pair<std::string, int>> header,
        const std::vector<std::pair<std::string, int>> subHeader,
        const std::vector<std::string> lineNames,
        const cellar::Grid2D<double>& data)
{
    QTextCursor cursor(_reportDocument);
    QTextBlockFormat blockFormat;
    QTextTableCell cell;
    QTextTable* table;

    QTextTableFormat propertyTableFormat;
    propertyTableFormat.setCellPadding(5.0);
    propertyTableFormat.setBorderStyle(
        QTextFrameFormat::BorderStyle_Solid);

    QTextCharFormat tableHeaderFormat;
    tableHeaderFormat.setFontWeight(QFont::Bold);


    size_t lc = data.getHeight() + 1;
    if(!subHeader.empty()) ++lc;
    size_t cc = data.getWidth() + 1;

    cursor.movePosition(QTextCursor::End);
    cursor.insertBlock(blockFormat);
    cursor.insertHtml(("<h3>" + title + "</h3>").c_str());
    table = cursor.insertTable(
        lc, cc, propertyTableFormat);

    int headpos = 0;
    int linepos = 0;
    for(int h=0; h < header.size(); ++h)
    {
        int width = header[h].second;
        const string& name = header[h].first;

        table->mergeCells(0, headpos, 1, width);
        cell = table->cellAt(0, headpos);
        QTextCursor cellCursor =
            cell.firstCursorPosition();

        cellCursor.insertText(name.c_str(), tableHeaderFormat);

        headpos += width;
    }
    ++linepos;

    if(!subHeader.empty())
    {
        int headpos = 0;
        for(int h=0; h < subHeader.size(); ++h)
        {
            int width = subHeader[h].second;
            const string& name = subHeader[h].first;

            table->mergeCells(1, headpos, 1, width);
            cell = table->cellAt(1, headpos);
            QTextCursor cellCursor =
                cell.firstCursorPosition();

            cellCursor.insertText(name.c_str());

            headpos += width;
        }
        ++linepos;
    }

    for(int l=0; l < lineNames.size(); ++l)
    {
        cell = table->cellAt(linepos+l, 0);
        QTextCursor cellCursor =
            cell.firstCursorPosition();

        cellCursor.insertText(lineNames[l].c_str());

        for(int c=0; c < data.getWidth(); ++c)
        {
            cell = table->cellAt(linepos+l, c+1);
            cellCursor = cell.firstCursorPosition();

                cellCursor.insertText(QString::number(data[l][c]));
        }
    }

    cursor.movePosition(QTextCursor::End);
    cursor.insertBlock(blockFormat);
}

void MastersTestSuite::evaluatorBlockSize(
        const string& testName)
{
    const vector<string> mesher = {
        "Delaunay",
        "Debug",
        //"file"
    };

    const vector<string> model = {
        "Sphere",
        "HexGrid",
        //"FINAL_MESH.cgns"
    };

    const int nodeCount = 5e3;

    const double metricK = 10;
    const double metricA = 5;
    const string sampler = "Texture";

    const string evaluator = "Metric Conformity";

    vector<string> implement = {
        "GLSL", "CUDA"
    };

    map<string, int> cycleCounts;
    for(const string& impl : implement)
        cycleCounts[impl] = 5;

    vector<uint> threadCounts = {
        8, 16, 32, 64, 128,
        192, 256, //512, 1024
    };

    _character.useSampler(sampler);
    _character.setMetricScaling(metricK);
    _character.setMetricAspectRatio(metricA);

    _character.useEvaluator(evaluator);


    Grid2D<double> data(mesher.size() * 2, threadCounts.size());

    for(int m=0; m < mesher.size(); ++m)
    {
        const string& me = mesher[m];
        const string& mo = model[m];

        if(me == "file")
            _character.loadMesh("resources/data/" + mo);
        else
            _character.generateMesh(me, mo, nodeCount);


        //QCoreApplication::processEvents();


        for(int t=0; t < threadCounts.size(); ++t)
        {
            uint tc = threadCounts[t];
            _character.setGlslEvaluatorThreadCount(tc);
            _character.setCudaEvaluatorThreadCount(tc);

            map<string, double> avrgTimes;
            _character.benchmarkEvaluator(
                avrgTimes, evaluator, cycleCounts);

            data[t][m*2 + 0] = avrgTimes[implement[0]];
            data[t][m*2 + 1] = avrgTimes[implement[1]];
        }
    }


    vector<pair<string, int>> header = {{"Tailles", 1}};
    vector<pair<string, int>> subheader = {{"", 1}};
    for(const string& m : model)
    {
        header.push_back({m, 2});

        for(const string& i : implement)
            subheader.push_back({
                _translateImplementation[i], 1});
    }

    vector<string> lineNames;
    for(int tc : threadCounts)
        lineNames.push_back(to_string(tc));

    output(testName, header, subheader, lineNames, data);
}

void MastersTestSuite::metricCost(
        const string& testName,
        const string& mesher,
        const string& model)
{
    size_t nodeCount = 5e3;
    size_t cycleCount = 5;
    double metricK = 20.0;
    double metricA = 4.0;
    int evaluateGlslThreadCount = 64;
    int evaluateCudaThreadCount = 32;
    string evaluator = "Metric Conformity";

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

    //_character.saveMesh(RESULT_MESH_PATH + testName + ".json");

    //QCoreApplication::processEvents();

    _character.setMetricScaling(metricK);
    _character.setMetricAspectRatio(metricA);
    _character.setGlslEvaluatorThreadCount(evaluateGlslThreadCount);
    _character.setCudaEvaluatorThreadCount(evaluateCudaThreadCount);

    Grid2D<double> data(implement.size(), sampling.size());

    for(int s=0; s < sampling.size(); ++s)
    {
        const string& samp = sampling[s];
        _character.useSampler(samp);

        // Process exceptions
        map<string, int> currCycle = implCycle;
        if(model == "HexGrid" && samp == "Local")
        {
            currCycle["GLSL"] = 0;
        }

        map<string, double> avgTimes;
        _character.benchmarkEvaluator(avgTimes, evaluator, currCycle);

        for(int i=0; i < implement.size(); ++i)
            data[s][i] = avgTimes[implement[i]];
    }

    vector<pair<string, int>> header = {
        {"Métriques", 1}, {"Temps (ms)", implement.size()}};

    vector<pair<string, int>> subheader = {{"", 1}};
    for(const string& i : implement)
        subheader.push_back({
            _translateImplementation[i], 1});

    vector<string> lineNames;
    for(const string& s : sampling)
        lineNames.push_back(_translateSampling[s]);

    output(testName, header, subheader, lineNames, data);
}

void MastersTestSuite::metricCostSphere(
        const string& testName)
{
    return metricCost(testName, "Delaunay", "Sphere");
}

void MastersTestSuite::metricCostHexGrid(
        const string& testName)
{
    return metricCost(testName, "Debug", "HexGrid");
}

void MastersTestSuite::metricPrecision(
        const string& testName)
{
    size_t nodeCount = 5e3;
    size_t passCount = 10;
    double metricK = 2.0;
    string implement = "Thread";
    string smoother = "Gradient Descent";
    string evaluator = "Metric Conformity";
    string mesher = "Delaunay";
    string model = "Sphere";

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

    vector<QualityHistogram> histograms;
    Grid2D<double> minData(ratios.size(), sampling.size() + 1);
    Grid2D<double> meanData(ratios.size(), sampling.size() + 1);

    _character.generateMesh(mesher, model, nodeCount);

    QString meshName = (RESULT_MESH_PATH + testName + " (A=%1).json").c_str();
    _character.saveMesh(meshName.arg(0).toStdString());

    _character.useEvaluator(evaluator);
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

        minData[0][i] = initHist.minimumQuality();
        meanData[0][i] = initHist.harmonicMean();
        for(int s=0; s < sampling.size(); ++s)
        {
            const QualityHistogram& hist =
                plot.implementations()[s].finalHistogram;
            minData[s+1][i] = hist.minimumQuality();
            meanData[s+1][i] = hist.harmonicMean();
        }

        if(i == ratios.size()-1)
        {
            histograms.push_back(initHist);

            for(int s=0; s < sampling.size(); ++s)
                histograms.push_back(
                    plot.implementations()[s].finalHistogram);
        }
    }

    vector<pair<string, int>> header = {{"Métriques", 1}};
    for(double r : ratios)
        header.push_back({"A = " + to_string(r), 1});

    vector<pair<string, int>> subheader = {};

    vector<string> lineNames = {"Initial"};
    for(const string& s : sampling)
        lineNames.push_back(_translateSampling[s]);

    vector<string> histHeader = {"Initial"};
    for(const string& s : sampling)
        histHeader.push_back(_translateSampling[s]);

    output(testName + "(Minimums)",
           header, subheader, lineNames, minData);
    output(testName + "(Harmonic means)",
           header, subheader, lineNames, meanData);
    output(testName + "(Histograms)",
           histHeader, histograms);
}

void MastersTestSuite::nodeOrder(
        const string& testName)
{
    const string mesher = "Delaunay";
    const string model = "Sphere";
    const int nodeCount = 5e3;

    const string samp = "Analytic";
    const double metricK = 10.0;
    const double metricA = 4.0;

    const string evaluator = "Metric Conformity";

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

    _character.useEvaluator(evaluator);

    _character.saveMesh(RESULT_MESH_PATH + testName + ".json");


    QCoreApplication::processEvents();


    OptimizationPlot plot;
    vector<Configuration> configs;
    for(const auto& i : impl)
        configs.push_back(Configuration{
            samp, smoother, i});

    _character.benchmarkSmoothers(
        plot, schedule, configs);

    Grid2D<double> data(impl.size()*2, schedule.globalPassCount+1);

    for(int p=0; p <= schedule.globalPassCount; ++p)
    {
        for(int i=0; i < impl.size(); ++i)
        {
            const QualityHistogram& hist =
                plot.implementations()[i].passes[p].histogram;
            data[p][0 + i] = hist.minimumQuality();
            data[p][2 + i] = hist.harmonicMean();

        }
    }


    vector<pair<string, int>> header = {
        {"Itérations", 1}, {"Minimums", 2}, {"Moyennes", 2}};

    vector<pair<string, int>> subheader = {{"", 1}};
    for(int m=0; m < 2; ++m)
    {
        for(const string& i : impl)
            subheader.push_back(
                {_translateImplementation[i], 1});
    }

    vector<string> lineNames;
    for(int p=0; p <= schedule.globalPassCount; ++p)
        lineNames.push_back(to_string(p));

    output(testName, header, subheader, lineNames, data);
}

void MastersTestSuite::smootherBlockSize(
        const string& testName)
{
    string mesh = RESULT_MESH_PATH + testName + ".json";
    string mesher = "Debug";
    string model = "Cube";
    int nodeCount = 5e3;

    string sampler = "Texture";
    double metricK = 4;
    double metricA = 5;

    string evaluator = "Metric Conformity";
    int evaluateGlslThreadCount = 64;
    int evaluateCudaThreadCount = 32;

    vector<string> smoother = {
        "Gradient Descent",
        "Nelder-Mead"
    };

    Schedule schedule;
    schedule.autoPilotEnabled = false;
    schedule.topoOperationEnabled = false;
    schedule.globalPassCount = 10;
    int topoPassCount = 10;

    vector<string> implement = {
        "GLSL", "CUDA"
    };

    vector<uint> threadCounts = {
        1, 2, 4, 8, 16, 32,
        64, 128, 192, 256//, 512, 1024
    };

    vector<Configuration> configs;
    for(const auto& s : smoother)
        for(const auto& i : implement)
            configs.push_back(Configuration{
                sampler, s, i});

    _character.useEvaluator(evaluator);
    _character.setGlslEvaluatorThreadCount(evaluateGlslThreadCount);
    _character.setCudaEvaluatorThreadCount(evaluateCudaThreadCount);

    _character.generateMesh(mesher, model, nodeCount);

    _character.useSampler("Analytic");
    _character.setMetricAspectRatio(metricA);
    for(double k=1.0; k < metricK; k *= 2)
    {
        _character.setMetricScaling(k);
        _character.restructureMesh(topoPassCount);

        QCoreApplication::processEvents();
    }

    _character.setMetricScaling(metricK);
    _character.restructureMesh(topoPassCount);

    QCoreApplication::processEvents();

    _character.saveMesh(mesh);
    _character.useSampler(sampler);


    Grid2D<double> data(smoother.size()*implement.size(), threadCounts.size());


    for(int t=0; t < threadCounts.size(); ++t)
    {
        if(t != 0)
        {
            _character.loadMesh(mesh);
        }

        uint tc = threadCounts[t];

        _character.setGlslSmootherThreadCount(tc);
        _character.setCudaSmootherThreadCount(tc);

        OptimizationPlot plot;
        _character.benchmarkSmoothers(
            plot, schedule, configs);

        for(int s=0; s < smoother.size(); ++s)
        {
            for(int i=0; i < implement.size(); ++i)
            {
                int id = s*implement.size() + i;
                data[t][id] = plot.implementations()[id]
                        .passes.back().timeStamp;
            }
        }
    }


    vector<pair<string, int>> header = {{"Tailles", 1}};
    vector<pair<string, int>> subheader = {{"", 1}};
    for(const string& s : smoother)
    {
        header.push_back({_translateSmoothers[s], implement.size()});
        for(const string& i : implement)
        {
            subheader.push_back({_translateImplementation[i], 1});
        }
    }

    vector<string> lineNames;
    for(int tc : threadCounts)
        lineNames.push_back(to_string(tc));

    output(testName, header, subheader, lineNames, data);
}
