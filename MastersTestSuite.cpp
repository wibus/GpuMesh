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
const string DATA_MESH_PATH = "resources/data/";

const string MESH_TETCUBE_K12 = RESULT_MESH_PATH + "TetCube (K=12).json";
const string MESH_TETCUBE_K24 = RESULT_MESH_PATH + "TetCube (K=24).json";
const string MESH_HEXGRID_500K = RESULT_MESH_PATH + "HexGrid (500K).json";
const string MESH_TURBINE_500K = DATA_MESH_PATH + "FINAL_MESH.cgns";

const double ADAPTATION_METRIC_K12 = 12;
const double ADAPTATION_METRIC_K24 = 24;
const double ADAPTATION_METRIC_A = 8;
const double ADAPTATION_TOPO_PASS = 5;
const double ADAPTATION_RELOC_PASS = 10;

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

        {to_string(++tId) + ". Metric Cost (TetCube)",
        MastersTestFunc(bind(&MastersTestSuite::metricCostTetCube,  this, _1))},

        {to_string(++tId) + ". Metric Cost (HexGrid)",
        MastersTestFunc(bind(&MastersTestSuite::metricCostHexGrid,  this, _1))},

        {to_string(++tId) + ". Metric Precision",
        MastersTestFunc(bind(&MastersTestSuite::metricPrecision,    this, _1))},

        {to_string(++tId) + ". Node Order",
        MastersTestFunc(bind(&MastersTestSuite::nodeOrder,          this, _1))},

        {to_string(++tId) + ". Smoothers Efficacity",
        MastersTestFunc(bind(&MastersTestSuite::smootherEfficacity, this, _1))},

        {to_string(++tId) + ". Smoothers Block Size",
        MastersTestFunc(bind(&MastersTestSuite::smootherBlockSize,  this, _1))},
    });

    _translateSamplingTechniques = {
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

    _meshNames = {
        {MESH_TETCUBE_K12, "TetCube (K=12)"},
        {MESH_TETCUBE_K24, "TetCube (K=24)"},
        {MESH_HEXGRID_500K, "HexGrid (500K)"},
        {MESH_TURBINE_500K, "Turbine (500K)"}
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
    // Prepare cases
    if(!QFile(MESH_TETCUBE_K12.c_str()).exists())
    {
        setupAdaptedCube(
            12,
            ADAPTATION_METRIC_A);

        _character.saveMesh(MESH_TETCUBE_K12);
    }

    if(!QFile(MESH_TETCUBE_K24.c_str()).exists())
    {
        setupAdaptedCube(
            24,
            ADAPTATION_METRIC_A);

        _character.saveMesh(MESH_TETCUBE_K24);
    }

    if(!QFile(MESH_HEXGRID_500K.c_str()).exists())
    {
        _character.useSampler("Analytic");

        _character.generateMesh("Debug", "HexGrid", 500e3);
        QCoreApplication::processEvents();

        _character.saveMesh(MESH_HEXGRID_500K);
    }

    if(!QFile(MESH_TURBINE_500K.c_str()).exists())
    {
        string original = DATA_MESH_PATH + "FINAL_MESH.cgns";

        if(QFile(original.c_str()).exists())
        {
            if(!QFile(original.c_str()).copy(
                        MESH_TURBINE_500K.c_str()))
            {
                getLog().postMessage(new Message('E', true,
                    "Could not copy 500K nodes turbine mesh",
                    "MastersTestSuite"));
                return;
            }
        }
        else
        {
            getLog().postMessage(new Message('E', true,
                "Missing 500K nodes turbine mesh",
                "MastersTestSuite"));
            return;
        }
    }

    _character.clearMesh();
    QCoreApplication::processEvents();


    reportDocument.clear();
    _reportDocument = &reportDocument;
    QTextCursor cursor(_reportDocument);


    cellar::Time duration;
    auto allStart = chrono::high_resolution_clock::now();

    for(const string& test : tests)
    {
        getLog().postMessage(new Message('I', false,
            "Running master's test: " + test,
            "MastersTestSuite"));

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

void MastersTestSuite::setupAdaptedCube(
        double metricK,
        double metricA)
{
    _character.useSampler("Analytic");
    _character.setMetricAspectRatio(metricA);
    _character.useEvaluator("Metric Conformity");
    _character.generateMesh("Debug", "Cube", 10);

    QCoreApplication::processEvents();


    for(double k=1.0; k < metricK; k *= 2.0)
    {
        _character.setMetricScaling(k);
        _character.restructureMesh(ADAPTATION_TOPO_PASS);

        QCoreApplication::processEvents();
    }


    _character.setMetricScaling(metricK);
    _character.restructureMesh(ADAPTATION_TOPO_PASS);

    QCoreApplication::processEvents();
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
            ss << "\\multicolumn{" << width << "}{c|}{" << name << "}";
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
                ss << "\\multicolumn{" << width << "}{c|}{" << name << "}";
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
    // Test case description
    const vector<string> meshes = {
        MESH_TETCUBE_K24,
        MESH_HEXGRID_500K,
        MESH_TURBINE_500K
    };

    const string sampler = "Texture";

    const string evaluator = "Metric Conformity";

    vector<string> implementations = {
        "GLSL", "CUDA"
    };

    map<string, int> cycleCounts;
    for(const string& impl : implementations)
        cycleCounts[impl] = 5;

    vector<uint> threadCounts = {
        1, 2, 4, 8, 16, 32,
        64, 128, 192, 256, //512, 1024
    };


    // Setup test
    _character.useSampler(sampler);
    _character.setMetricScaling(ADAPTATION_METRIC_K24);
    _character.setMetricAspectRatio(ADAPTATION_METRIC_A);

    _character.useEvaluator(evaluator);


    // Run test
    Grid2D<double> data(meshes.size() * 2, threadCounts.size());

    for(int m=0; m < meshes.size(); ++m)
    {
        _character.loadMesh(meshes[m]);

        //QCoreApplication::processEvents();


        for(int t=0; t < threadCounts.size(); ++t)
        {
            uint tc = threadCounts[t];
            _character.setGlslEvaluatorThreadCount(tc);
            _character.setCudaEvaluatorThreadCount(tc);

            map<string, double> avrgTimes;
            _character.benchmarkEvaluator(
                avrgTimes, evaluator, cycleCounts);

            data[t][m*2 + 0] = avrgTimes[implementations[0]];
            data[t][m*2 + 1] = avrgTimes[implementations[1]];
        }
    }


    // Print results
    vector<pair<string, int>> header = {{"Tailles", 1}};
    vector<pair<string, int>> subheader = {{"", 1}};
    for(const string& m : meshes)
    {
        header.push_back({_meshNames[m], 2});

        for(const string& i : implementations)
            subheader.push_back({
                _translateImplementations[i], 1});
    }

    vector<string> lineNames;
    for(int tc : threadCounts)
        lineNames.push_back(to_string(tc));

    output(testName, header, subheader, lineNames, data);
}

void MastersTestSuite::metricCost(
        const string& testName,
        const string& mesh)
{
    // Test case description
    size_t evalCycleCount = 5;
    int evaluateGlslThreadCount = 64;
    int evaluateCudaThreadCount = 32;
    string evaluator = "Metric Conformity";

    vector<string> samplings = {
        "Analytic",
        "Local",
        "Texture",
        "Kd-Tree"
    };

    vector<string> implementations = {
        "Serial",
        "Thread",
        "GLSL",
        "CUDA"
    };

    map<string, int> implCycle;
    for(const auto& impl : implementations)
        implCycle[impl] = evalCycleCount;


    // Setup test
    _character.loadMesh(mesh);

    //QCoreApplication::processEvents();

    _character.setMetricScaling(ADAPTATION_METRIC_K24);
    _character.setMetricAspectRatio(ADAPTATION_METRIC_A);
    _character.setGlslEvaluatorThreadCount(evaluateGlslThreadCount);
    _character.setCudaEvaluatorThreadCount(evaluateCudaThreadCount);


    // Run test
    Grid2D<double> data(implementations.size(), samplings.size());

    for(int s=0; s < samplings.size(); ++s)
    {
        const string& samp = samplings[s];
        _character.useSampler(samp);

        // Process exceptions
        map<string, int> currCycle = implCycle;
        if(mesh == MESH_HEXGRID_500K && samp == "Local")
        {
            currCycle["GLSL"] = 0;
        }

        map<string, double> avgTimes;
        _character.benchmarkEvaluator(avgTimes, evaluator, currCycle);

        for(int i=0; i < implementations.size(); ++i)
            data[s][i] = avgTimes[implementations[i]];
    }


    // Print results
    vector<pair<string, int>> header = {
        {"Métriques", 1}, {"Temps (ms)", implementations.size()}};

    vector<pair<string, int>> subheader = {{"", 1}};
    for(const string& i : implementations)
        subheader.push_back({
            _translateImplementations[i], 1});

    vector<string> lineNames;
    for(const string& s : samplings)
        lineNames.push_back(_translateSamplingTechniques[s]);

    output(testName, header, subheader, lineNames, data);
}

void MastersTestSuite::metricCostTetCube(
        const string& testName)
{
    return metricCost(testName, MESH_TETCUBE_K24);
}

void MastersTestSuite::metricCostHexGrid(
        const string& testName)
{
    return metricCost(testName, MESH_HEXGRID_500K);
}

void MastersTestSuite::metricPrecision(
        const string& testName)
{
    // Test case description
    string implementation = "Thread";
    string smoother = "Gradient Descent";
    string evaluator = "Metric Conformity";

    vector<string> samplings = {
        "Analytic",
        "Local",
        "Texture",
        "Kd-Tree"
    };

    vector<double> metricAs = {
        1, 2, 4, 8, 16
    };

    Schedule schedule;
    schedule.autoPilotEnabled = false;
    schedule.topoOperationEnabled = false;
    schedule.relocationPassCount = ADAPTATION_RELOC_PASS;


    // Setup test
    setupAdaptedCube(ADAPTATION_METRIC_K12, 1.0);

    QString meshName = (RESULT_MESH_PATH + testName + " (A=%1).json").c_str();
    _character.saveMesh(meshName.arg(0).toStdString());

    _character.setMetricScaling(ADAPTATION_METRIC_K12);

    _character.useEvaluator(evaluator);


    // Run test
    vector<QualityHistogram> histograms;
    Grid2D<double> minData(metricAs.size(), samplings.size() + 1);
    Grid2D<double> meanData(metricAs.size(), samplings.size() + 1);

    for(int a=0; a < metricAs.size(); ++a)
    {
        if(a != 0)
        {
            _character.loadMesh(meshName.arg(metricAs[a-1]).toStdString());
        }

        _character.useSampler("Analytic");
        _character.setMetricAspectRatio(metricAs[a]);
        _character.restructureMesh(ADAPTATION_TOPO_PASS);

        _character.saveMesh(meshName.arg(metricAs[a]).toStdString());


        QCoreApplication::processEvents();

        OptimizationPlot plot;
        vector<Configuration> configs;
        for(const string& samp : samplings)
            configs.push_back(Configuration{
                samp, smoother, implementation});

        _character.benchmarkSmoothers(
            plot, schedule, configs);

        const QualityHistogram& initHist = plot.initialHistogram();

        minData[0][a] = initHist.minimumQuality();
        meanData[0][a] = initHist.harmonicMean();
        for(int s=0; s < samplings.size(); ++s)
        {
            const QualityHistogram& hist =
                plot.implementations()[s].finalHistogram;
            minData[s+1][a] = hist.minimumQuality();
            meanData[s+1][a] = hist.harmonicMean();
        }

        if(a == metricAs.size()-1)
        {
            histograms.push_back(initHist);

            for(int s=0; s < samplings.size(); ++s)
                histograms.push_back(
                    plot.implementations()[s].finalHistogram);
        }
    }


    // Print results
    vector<pair<string, int>> header = {{"Métriques", 1}};
    for(double r : metricAs)
        header.push_back({"A = " + to_string(r), 1});

    vector<pair<string, int>> subheader = {};

    vector<string> lineNames = {"Initial"};
    for(const string& s : samplings)
        lineNames.push_back(_translateSamplingTechniques[s]);

    vector<string> histHeader = {"Initial"};
    for(const string& s : samplings)
        histHeader.push_back(_translateSamplingTechniques[s]);

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
    // Test case description
    string mesh = MESH_TETCUBE_K12;

    const string sampler = "Analytic";

    const string evaluator = "Metric Conformity";

    const string smoother = "Gradient Descent";

    const vector<string> implementations = {
        "Serial",
        "Thread"
    };

    OptimizationPlot plot;
    vector<Configuration> configs;
    for(const auto& impl : implementations)
        configs.push_back(Configuration{
            sampler, smoother, impl});

    Schedule schedule;
    schedule.autoPilotEnabled = false;
    schedule.topoOperationEnabled = false;
    schedule.relocationPassCount = ADAPTATION_RELOC_PASS;


    // Setup test
    _character.loadMesh(mesh);

    _character.useSampler(sampler);
    _character.setMetricScaling(ADAPTATION_METRIC_K12);
    _character.setMetricAspectRatio(ADAPTATION_METRIC_A);

    _character.useEvaluator(evaluator);


    // Run test
    _character.benchmarkSmoothers(
        plot, schedule, configs);

    Grid2D<double> data(implementations.size()*2, schedule.relocationPassCount+1);

    for(int r=0; r <= schedule.relocationPassCount; ++r)
    {
        for(int i=0; i < implementations.size(); ++i)
        {
            const QualityHistogram& hist =
                plot.implementations()[i].passes[r].histogram;
            data[r][0 + i] = hist.minimumQuality();
            data[r][2 + i] = hist.harmonicMean();

        }
    }


    // Print results
    vector<pair<string, int>> header = {
        {"Itérations", 1}, {"Minimums", 2}, {"Moyennes", 2}};

    vector<pair<string, int>> subheader = {{"", 1}};
    for(int m=0; m < 2; ++m)
    {
        for(const string& i : implementations)
            subheader.push_back(
                {_translateImplementations[i], 1});
    }

    vector<string> lineNames;
    for(int p=0; p <= schedule.relocationPassCount; ++p)
        lineNames.push_back(to_string(p));

    output(testName, header, subheader, lineNames, data);
}

void MastersTestSuite::smootherEfficacity(
        const string& testName)
{
    // Test case description
    string mesh = MESH_TETCUBE_K12;

    string sampler = "Analytic";

    string evaluator = "Metric Conformity";

    vector<string> smoothers = {
        "Spring Laplace",
        "Quality Laplace",
        "Spawn Search",
        "Nelder-Mead",
        "Gradient Descent"
    };

    string implementation = "Thread";

    vector<Configuration> configs;
    for(const auto& s : smoothers)
            configs.push_back(Configuration{
                sampler, s, implementation});


    Schedule schedule;
    schedule.autoPilotEnabled = false;
    schedule.topoOperationEnabled = false;
    schedule.relocationPassCount = ADAPTATION_RELOC_PASS;


    // Setup test
    _character.loadMesh(mesh);

    _character.useSampler(sampler);
    _character.setMetricScaling(ADAPTATION_METRIC_K12);
    _character.setMetricAspectRatio(ADAPTATION_METRIC_A);

    _character.useEvaluator(evaluator);


    // Run test
    Grid2D<double> data(2, smoothers.size()+1);

    OptimizationPlot plot;
    _character.benchmarkSmoothers(
        plot, schedule, configs);

    const QualityHistogram& initHist = plot.initialHistogram();
    data[0][0] = initHist.minimumQuality();
    data[0][1] = initHist.harmonicMean();

    for(int s=0; s < smoothers.size(); ++s)
    {
        const QualityHistogram& hist =
            plot.implementations()[s].finalHistogram;

        data[s+1][0] = hist.minimumQuality();
        data[s+1][1] = hist.harmonicMean();
    }


    // Print results
    vector<pair<string, int>> header = {
        {"Métriques", 1}, {"Minimums", 1}, {"Moyennes", 1}};
    vector<pair<string, int>> subheader = {};

    vector<string> lineNames = {"Initial"};
    for(const string& s : smoothers)
        lineNames.push_back(_translateSmoothers[s]);

    output(testName, header, subheader, lineNames, data);
}

void MastersTestSuite::smootherBlockSize(
        const string& testName)
{
    // Test case description
    string mesh = MESH_TETCUBE_K12;

    string sampler = "Texture";

    string evaluator = "Metric Conformity";
    int evaluateGlslThreadCount = 64;
    int evaluateCudaThreadCount = 32;

    vector<string> smoothers = {
        "Gradient Descent",
        "Nelder-Mead"
    };

    vector<string> implementations = {
        "GLSL", "CUDA"
    };

    vector<uint> threadCounts = {
        1, 2, 4, 8, 16, 32,
        64, 128, 192, 256//, 512, 1024
    };

    vector<Configuration> configs;
    for(const auto& smooth : smoothers)
        for(const auto& impl : implementations)
            configs.push_back(Configuration{
                sampler, smooth, impl});

    Schedule schedule;
    schedule.autoPilotEnabled = false;
    schedule.topoOperationEnabled = false;
    schedule.relocationPassCount = ADAPTATION_RELOC_PASS;


    // Setup test
    _character.useSampler(sampler);
    _character.setMetricScaling(ADAPTATION_METRIC_K12);
    _character.setMetricAspectRatio(ADAPTATION_METRIC_A);

    _character.useEvaluator(evaluator);
    _character.setGlslEvaluatorThreadCount(evaluateGlslThreadCount);
    _character.setCudaEvaluatorThreadCount(evaluateCudaThreadCount);


    // Run test
    Grid2D<double> data(smoothers.size()*implementations.size(), threadCounts.size());

    for(int t=0; t < threadCounts.size(); ++t)
    {
        _character.loadMesh(mesh);

        uint tc = threadCounts[t];

        _character.setGlslSmootherThreadCount(tc);
        _character.setCudaSmootherThreadCount(tc);

        OptimizationPlot plot;
        _character.benchmarkSmoothers(
            plot, schedule, configs);

        for(int s=0; s < smoothers.size(); ++s)
        {
            for(int i=0; i < implementations.size(); ++i)
            {
                int id = s*implementations.size() + i;
                data[t][id] = plot.implementations()[id]
                        .passes.back().timeStamp;
            }
        }
    }


    // Print results
    vector<pair<string, int>> header = {{"Tailles", 1}};
    vector<pair<string, int>> subheader = {{"", 1}};
    for(const string& s : smoothers)
    {
        header.push_back({_translateSmoothers[s], implementations.size()});
        for(const string& i : implementations)
        {
            subheader.push_back({_translateImplementations[i], 1});
        }
    }

    vector<string> lineNames;
    for(int tc : threadCounts)
        lineNames.push_back(to_string(tc));

    output(testName, header, subheader, lineNames, data);
}
