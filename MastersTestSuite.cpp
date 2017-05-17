#include "MastersTestSuite.h"

#include <set>
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
const string RESULT_MESH_PATH = RESULTS_PATH + "mesh/";
const string RESULT_LATEX_PATH = RESULTS_PATH + "latex/";
const string RESULT_CSV_PATH = RESULTS_PATH + "csv/";
const string DATA_MESH_PATH = "resources/data/";

const string MESH_SPHERE_100K = RESULT_MESH_PATH + "Sphere (N=100K).json";
const string MESH_TETCUBE_K5 = RESULT_MESH_PATH + "TetCube (K=5).json";
const string MESH_TETCUBE_K12 = RESULT_MESH_PATH + "TetCube (K=12).json";
const string MESH_TETCUBE_K16 = RESULT_MESH_PATH + "TetCube (K=16).json";
const string MESH_TETCUBE_K24 = RESULT_MESH_PATH + "TetCube (K=24).json";
const string MESH_HEXCUBE_10K = RESULT_MESH_PATH + "HexCube (N=10K).json";
const string MESH_HEXCUBE_175K = RESULT_MESH_PATH + "HexCube (N=175K).json";
const string MESH_HEXCUBE_500K = RESULT_MESH_PATH + "HexCube (N=500K).json";
const string MESH_TURBINE_500K = RESULT_MESH_PATH + "Turbine (N=500K).cgns";

const string MESH_PRECISION_BASE = RESULT_MESH_PATH + "Precision (A=%1).json";
const string MESH_SCALING_BASE = RESULT_MESH_PATH + "Scaling (Scale=%1).json";

vector<double> PRECISION_METRIC_As = {1, 2, 4, 8, 16};

const int BS = 1e4;
const vector<int> SPHERE_TARGET_SIZES = {
    1*BS,  2*BS,  4*BS,  8*BS,
    16*BS, 32*BS, 64*BS, 128*BS};
double sizeToScale(int size) {return glm::pow(size, 1/2.9) / 4.08;}

const double ADAPTATION_METRIC_K5 = 5;
const double ADAPTATION_METRIC_K12 = 12;
const double ADAPTATION_METRIC_K16 = 16;
const double ADAPTATION_METRIC_K24 = 24;
const double ADAPTATION_METRIC_A = 8;
const double ADAPTATION_TOPO_PASS = 5;
const double ADAPTATION_RELOC_PASS = 10;

const int EVALUATION_THREAD_COUNT_GLSL = 16;
const int EVALUATION_THREAD_COUNT_CUDA = 32;

const int SMOOTHING_THREAD_COUNT_GLSL = 16;
const int SMOOTHING_THREAD_COUNT_CUDA = 32;

string testNumber(int n)
{
    return QString("%1").arg(n, 2, 10, QChar('0')).toStdString();
}

MastersTestSuite::MastersTestSuite(
        GpuMeshCharacter& character) :
    _character(character),
    _availableMastersTests("Master's tests")
{
    using namespace std::placeholders;

    int tId = 0;
    _availableMastersTests.setDefault("N/A");

    _availableMastersTests.setContent({

        {testNumber(++tId) + ". Metric Precision",
        MastersTestFunc(bind(&MastersTestSuite::metricPrecision,            this, _1))},

        {testNumber(++tId) + ". Texture Precision",
        MastersTestFunc(bind(&MastersTestSuite::texturePrecision,           this, _1))},

        {testNumber(++tId) + ". Evaluator Block Size",
        MastersTestFunc(bind(&MastersTestSuite::evaluatorBlockSize,         this, _1))},

        {testNumber(++tId) + ". Metric Cost (TetCube)",
        MastersTestFunc(bind(&MastersTestSuite::metricCostTetCube,          this, _1))},

        {testNumber(++tId) + ". Metric Cost (HexCube)",
        MastersTestFunc(bind(&MastersTestSuite::metricCostHexCube,          this, _1))},

        {testNumber(++tId) + ". Node Order",
        MastersTestFunc(bind(&MastersTestSuite::nodeOrder,                  this, _1))},

        {testNumber(++tId) + ". Smoothers Efficacity",
        MastersTestFunc(bind(&MastersTestSuite::smootherEfficacity,         this, _1))},

        {testNumber(++tId) + ". Smoothers Block Size (TetCube)",
        MastersTestFunc(bind(&MastersTestSuite::smootherBlockSizeTetCube,   this, _1))},

        {testNumber(++tId) + ". Smoothers Block Size (HexCube)",
        MastersTestFunc(bind(&MastersTestSuite::smootherBlockSizeHexCube,   this, _1))},

        {testNumber(++tId) + ". Smoothers Speed (TetCube)",
        MastersTestFunc(bind(&MastersTestSuite::smootherSpeedTetCube,       this, _1))},

        {testNumber(++tId) + ". Smoothers Speed (HexCube)",
        MastersTestFunc(bind(&MastersTestSuite::smootherSpeedHexCube,       this, _1))},

        {testNumber(++tId) + ". Relocation Scaling",
        MastersTestFunc(bind(&MastersTestSuite::relocationScaling,          this, _1))},
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
        {"Multi Elem NM",       "NM multi-éléments"},
        {"Gradient Descent",    "Descente du gradient"},
        {"Multi Elem GD",       "DG multi-éléments"},
        {"Multi Pos GD",        "DG multi-positions"},
        {"Patch GD",            "DG multi-axes"},
        {"GETMe",               "GETMe"}
    };

    _meshNames = {
        {MESH_SPHERE_100K, "Sphere (N=100K)"},
        {MESH_TETCUBE_K5, "TetCube (K=5)"},
        {MESH_TETCUBE_K12, "TetCube (K=12)"},
        {MESH_TETCUBE_K16, "TetCube (K=16)"},
        {MESH_TETCUBE_K24, "TetCube (K=24)"},
        {MESH_HEXCUBE_10K, "HexCube (N=10K)"},
        {MESH_HEXCUBE_175K, "HexCube (N=175K)"},
        {MESH_HEXCUBE_500K, "HexCube (N=500K)"},
        {MESH_TURBINE_500K, "Turbine (N=500K)"}
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
    if(!QFile(MESH_SPHERE_100K.c_str()).exists())
    {
        _character.generateMesh("Delaunay", "Sphere", 10e3);

        _character.saveMesh(MESH_SPHERE_100K);
    }

    if(!QFile(MESH_TETCUBE_K5.c_str()).exists())
    {
        setupAdaptedCube(
            ADAPTATION_METRIC_K5,
            ADAPTATION_METRIC_A);

        _character.saveMesh(MESH_TETCUBE_K5);
    }

    if(!QFile(MESH_TETCUBE_K12.c_str()).exists())
    {
        setupAdaptedCube(
            ADAPTATION_METRIC_K12,
            ADAPTATION_METRIC_A);

        _character.saveMesh(MESH_TETCUBE_K12);
    }

    if(!QFile(MESH_TETCUBE_K16.c_str()).exists())
    {
        setupAdaptedCube(
            ADAPTATION_METRIC_K16,
            ADAPTATION_METRIC_A);

        _character.saveMesh(MESH_TETCUBE_K16);
    }

    if(!QFile(MESH_TETCUBE_K24.c_str()).exists())
    {
        setupAdaptedCube(
            ADAPTATION_METRIC_K24,
            ADAPTATION_METRIC_A);

        _character.saveMesh(MESH_TETCUBE_K24);
    }

    if(!QFile(MESH_HEXCUBE_10K.c_str()).exists())
    {
        _character.useSampler("Analytic");

        _character.generateMesh("Debug", "HexGrid", 10e3);
        QCoreApplication::processEvents();

        _character.saveMesh(MESH_HEXCUBE_10K);
    }

    if(!QFile(MESH_HEXCUBE_175K.c_str()).exists())
    {
        _character.useSampler("Analytic");

        _character.generateMesh("Debug", "HexGrid", 175e3);
        QCoreApplication::processEvents();

        _character.saveMesh(MESH_HEXCUBE_175K);
    }

    if(!QFile(MESH_HEXCUBE_500K.c_str()).exists())
    {
        _character.useSampler("Analytic");

        _character.generateMesh("Debug", "HexGrid", 500e3);
        QCoreApplication::processEvents();

        _character.saveMesh(MESH_HEXCUBE_500K);
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

    if(!QFile(QString(MESH_PRECISION_BASE.c_str()).arg(PRECISION_METRIC_As.back())).exists())
    {
        setupAdaptedCube(ADAPTATION_METRIC_K12, PRECISION_METRIC_As.front());

        for(int a=0; a < PRECISION_METRIC_As.size(); ++a)
        {
            double metricA = PRECISION_METRIC_As[a];
            QString name = QString(MESH_PRECISION_BASE.c_str()).arg(metricA);

            _character.setMetricAspectRatio(metricA);
            _character.restructureMesh(ADAPTATION_TOPO_PASS);

            if(!QFile(name).exists())
            {
                _character.saveMesh(name.toStdString());
            }
        }
    }

    for(int s=0; s < SPHERE_TARGET_SIZES.size(); ++s)
    {
        size_t size = SPHERE_TARGET_SIZES[s];
        QString name = QString(MESH_SCALING_BASE.c_str()).arg(s);

        if(!QFile(name).exists())
        {
            _character.generateMesh("Delaunay", "Sphere", size);

            _character.setMetricScaling(sizeToScale(size));
            _character.setMetricAspectRatio(ADAPTATION_METRIC_A);

            _character.useSampler("Analytic");
            _character.restructureMesh(10);

            _character.saveMesh(name.toStdString());
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

    //QCoreApplication::quit();
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

    ss << "Qualités" << ", ";

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
        double bottom = b / double(bc);
        double top = (b+1) / double(bc);

        ss << bottom << " - " << top << ", ";

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

    ss << "Qualités" << "\t& ";

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
        double bottom = b / double(bc);
        double top = (b+1) / double(bc);

        ss << bottom << " - " << top << "\t& ";

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

    size_t hc = histograms.size();
    size_t bc = histograms.front().bucketCount();

    cursor.movePosition(QTextCursor::End);
    cursor.insertBlock(blockFormat);
    cursor.insertHtml(("<h3>" + title + "</h3>").c_str());
    table = cursor.insertTable(
        bc + 1, hc+1, propertyTableFormat);


    cell = table->cellAt(0, 0);
    QTextCursor cellCursor = cell.firstCursorPosition();
    cellCursor.insertText("Qualités", tableHeaderFormat);


    for(int h=0; h < header.size(); ++h)
    {
        cell = table->cellAt(0, h+1);
        QTextCursor cellCursor =
            cell.firstCursorPosition();

        cellCursor.insertText(header[h].c_str(), tableHeaderFormat);
    }

    for(int b=0; b < bc; ++b)
    {
        double bottom = b / double(bc);
        double top = (b+1) / double(bc);

        cell = table->cellAt(b+1, 0);
        QTextCursor cellCursor =
            cell.firstCursorPosition();

        cellCursor.insertText(QString("%1 - %2")
                .arg(bottom).arg(top));


        for(int h=0; h < histograms.size(); ++h)
        {
            const QualityHistogram& hist =
                    histograms[h];

            cell = table->cellAt(b+1, h+1);
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

    Schedule schedule;
    schedule.autoPilotEnabled = false;
    schedule.topoOperationEnabled = false;
    schedule.relocationPassCount = ADAPTATION_RELOC_PASS;


    // Setup test
    _character.setMetricScaling(ADAPTATION_METRIC_K12);

    _character.useEvaluator(evaluator);


    // Run test
    vector<QualityHistogram> histograms;
    Grid2D<double> minData(samplings.size() + 1, PRECISION_METRIC_As.size());
    Grid2D<double> meanData(samplings.size() + 1, PRECISION_METRIC_As.size());

    for(int a=0; a < PRECISION_METRIC_As.size(); ++a)
    {
        double metricA = PRECISION_METRIC_As[a];

        _character.setMetricAspectRatio(metricA);
        _character.loadMesh(QString(MESH_PRECISION_BASE.c_str())
            .arg(metricA).toStdString());

        OptimizationPlot plot;
        vector<Configuration> configs;
        for(const string& samp : samplings)
            configs.push_back(Configuration{
                samp, smoother, implementation});

        _character.benchmarkSmoothers(
            plot, schedule, configs);

        const QualityHistogram& initHist = plot.initialHistogram();

        minData[a][0] = initHist.minimumQuality();
        meanData[a][0] = initHist.harmonicMean();
        for(int s=0; s < samplings.size(); ++s)
        {
            const QualityHistogram& hist =
                plot.implementations()[s].finalHistogram;
            minData[a][s+1] = hist.minimumQuality();
            meanData[a][s+1] = hist.harmonicMean();
        }

        if(a == PRECISION_METRIC_As.size()-1)
        {
            histograms.push_back(initHist);

            for(int s=0; s < samplings.size(); ++s)
                histograms.push_back(
                    plot.implementations()[s].finalHistogram);
        }
    }


    // Print results
    vector<pair<string, int>> header = {{"A", 1}};
    for(const string& s : samplings)
        header.push_back({_translateSamplingTechniques[s], 1});

    vector<pair<string, int>> subheader = {};

    vector<string> lineNames = {"Initial"};
    for(double r : PRECISION_METRIC_As)
        lineNames.push_back(QString::number(r).toStdString());

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

void MastersTestSuite::texturePrecision(
        const string& testName)
{
    // Test case description
    double metricA = ADAPTATION_METRIC_A;
    double metricK = PRECISION_METRIC_As[3];
    string mesh = QString(MESH_PRECISION_BASE.c_str())
            .arg(metricK).toStdString();

    string implementation = "Thread";
    string smoother = "Gradient Descent";
    string evaluator = "Metric Conformity";
    string sampling = "Texture";

    vector<double> depths = {
        1, 2, 4, 8, 16, 32, 64
    };


    Schedule schedule;
    schedule.autoPilotEnabled = false;
    schedule.topoOperationEnabled = false;
    schedule.relocationPassCount = ADAPTATION_RELOC_PASS;


    // Setup test
    _character.useEvaluator(evaluator);

    _character.setMetricScaling(metricK);
    _character.setMetricAspectRatio(metricA);


    // Run test
    Grid2D<double> data(2, depths.size() + 1);

    for(int d=0; d < depths.size(); ++d)
    {
        _character.loadMesh(mesh);
        _character.setMetricDiscretizationDepth(depths[d]);

        OptimizationPlot plot;
        vector<Configuration> configs;
        configs.push_back(Configuration{
            sampling, smoother, implementation});

        _character.benchmarkSmoothers(
            plot, schedule, configs);


        if(d == 0)
        {
            const QualityHistogram& initHist =
                    plot.initialHistogram();

            data[0][0] = initHist.minimumQuality();
            data[0][1] = initHist.harmonicMean();
        }


        const QualityHistogram& hist =
            plot.implementations()[0].finalHistogram;

        data[d+1][0] = hist.minimumQuality();
        data[d+1][1] = hist.harmonicMean();
    }


    // Print results
    vector<pair<string, int>> header = {
        {"Résolutions", 1}, {"Minimums", 1}, {"Moyennes", 1}};
    vector<pair<string, int>> subheader = {};

    vector<string> lineNames = {"Initial"};
    for(int d : depths)
        lineNames.push_back(to_string(d));

    output(testName, header, subheader, lineNames, data);
}

void MastersTestSuite::evaluatorBlockSize(
        const string& testName)
{
    // Test case description
    const vector<string> meshes = {
        MESH_TETCUBE_K24,
        MESH_HEXCUBE_500K
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
    _character.setGlslEvaluatorThreadCount(EVALUATION_THREAD_COUNT_GLSL);
    _character.setCudaEvaluatorThreadCount(EVALUATION_THREAD_COUNT_CUDA);


    // Run test
    int implCount = implementations.size();
    Grid2D<double> data((implCount-1)*2+1, samplings.size());

    for(int s=0; s < samplings.size(); ++s)
    {
        const string& samp = samplings[s];
        _character.useSampler(samp);

        // Process exceptions
        map<string, int> currCycle = implCycle;
        if(mesh == MESH_HEXCUBE_500K && samp == "Local")
        {
            currCycle["GLSL"] = 0;
        }

        map<string, double> avgTimes;
        _character.benchmarkEvaluator(avgTimes, evaluator, currCycle);

        for(int i=0; i < implCount; ++i)
            data[s][i] = avgTimes[implementations[i]];

        for(int i=1; i < implCount; ++i)
            data[s][implCount+i-1] = avgTimes[implementations[0]]
                    / avgTimes[implementations[i]];
    }


    // Print results
    vector<pair<string, int>> header = {
        {"Métriques", 1},
        {"Temps (ms)", implementations.size()},
        {"Accélérations", implementations.size()-1}};

    vector<pair<string, int>> subheader = {{"", 1}};
    for(const string& i : implementations)
        subheader.push_back({
            _translateImplementations[i], 1});

    for(int i=1; i < implementations.size(); ++i)
        subheader.push_back({
            _translateImplementations[implementations[i]], 1});


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

void MastersTestSuite::metricCostHexCube(
        const string& testName)
{
    return metricCost(testName, MESH_HEXCUBE_500K);
}

void MastersTestSuite::nodeOrder(
        const string& testName)
{
    // Test case description
    int fullRelocationPassCount = 100;
    int displayRelocationPassCount = 10;
    string mesh = MESH_SPHERE_100K;

    const string sampler = "Analytic";

    const string evaluator = "Metric Conformity";

    const string smoother = "Gradient Descent";

    const vector<string> implementations = {
        "Serial",
        "Thread",
        "GLSL",
        "CUDA"
    };

    OptimizationPlot plot;
    vector<Configuration> configs;
    for(const auto& impl : implementations)
        configs.push_back(Configuration{
            sampler, smoother, impl});

    Schedule schedule;
    schedule.autoPilotEnabled = false;
    schedule.topoOperationEnabled = false;
    schedule.relocationPassCount = fullRelocationPassCount;


    // Setup test
    _character.loadMesh(mesh);

    _character.useSampler(sampler);
    _character.setMetricScaling(ADAPTATION_METRIC_K12);
    _character.setMetricAspectRatio(ADAPTATION_METRIC_A);

    _character.useEvaluator(evaluator);


    // Run test
    _character.benchmarkSmoothers(
        plot, schedule, configs);

    Grid2D<double> dataCPU(2 * 2, displayRelocationPassCount+2);
    Grid2D<double> dataPara((implementations.size()-1) * 2, displayRelocationPassCount+2);

    for(int r=0; r <= displayRelocationPassCount; ++r)
    {
        for(int i=0; i < 2; ++i)
        {
            const QualityHistogram& hist =
                plot.implementations()[i].passes[r].histogram;
            dataCPU[r][0 + i] = hist.minimumQuality();
            dataCPU[r][2 + i] = hist.harmonicMean();
        }

        for(int i=1; i < implementations.size(); ++i)
        {
            const QualityHistogram& hist =
                plot.implementations()[i].passes[r].histogram;
            dataPara[r][0 + i-1] = hist.minimumQuality();
            dataPara[r][3 + i-1] = hist.harmonicMean();
        }
    }

    for(int i=0; i < 2; ++i)
    {
        const QualityHistogram& hist =
            plot.implementations()[i].passes[fullRelocationPassCount].histogram;
        dataCPU[displayRelocationPassCount+1][0 + i] = hist.minimumQuality();
        dataCPU[displayRelocationPassCount+1][2 + i] = hist.harmonicMean();
    }

    for(int i=1; i < implementations.size(); ++i)
    {
        const QualityHistogram& hist =
            plot.implementations()[i].passes[fullRelocationPassCount].histogram;
        dataPara[displayRelocationPassCount+1][0 + i-1] = hist.minimumQuality();
        dataPara[displayRelocationPassCount+1][3 + i-1] = hist.harmonicMean();
    }

    vector<QualityHistogram> histograms;
    histograms.push_back(plot.initialHistogram());
    for(int i=0; i < 2; ++i)
        histograms.push_back(plot.implementations()[i].finalHistogram);


    // Print results
    vector<pair<string, int>> headerCPU = {
        {"Itérations", 1}, {"Minimums", 2}, {"Moyennes", 2}};
    vector<pair<string, int>> headerPara = {
        {"Itérations", 1}, {"Minimums", 3}, {"Moyennes", 3}};

    vector<pair<string, int>> subheaderCPU = {{"", 1}};
    for(int m=0; m < 2; ++m)
    {
        for(int i=0; i < 2; ++i)
        {
            subheaderCPU.push_back(
                {_translateImplementations[implementations[i]], 1});
        }
    }

    vector<pair<string, int>> subheaderPara = {{"", 1}};
    for(int m=0; m < 2; ++m)
    {
        for(int i=1; i < implementations.size(); ++i)
        {
            subheaderPara.push_back(
                {_translateImplementations[implementations[i]], 1});
        }
    }

    vector<string> lineNames;
    for(int p=0; p <= displayRelocationPassCount; ++p)
        lineNames.push_back(to_string(p));
    lineNames.push_back(to_string(fullRelocationPassCount));

    vector<string> histHeader = {"Initial"};
    for(int i=0; i < 2; ++i)
    {
        histHeader.push_back(_translateImplementations[implementations[i]]);
    }

    output(testName + "(CPU)",      headerCPU,  subheaderCPU,  lineNames, dataCPU);
    output(testName + "(Parallel)", headerPara, subheaderPara, lineNames, dataPara);
    output(testName + "(Histograms)", histHeader, histograms);
}

void MastersTestSuite::smootherEfficacity(
        const string& testName)
{
    // Test case description
    vector<string> meshes = {
        MESH_TETCUBE_K5,
        MESH_HEXCUBE_10K,
    };

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
    _character.useSampler(sampler);
    _character.setMetricScaling(ADAPTATION_METRIC_K5);
    _character.setMetricAspectRatio(ADAPTATION_METRIC_A);

    _character.useEvaluator(evaluator);


    // Run test
    Grid2D<double> data(2*meshes.size(), smoothers.size()+1);

    for(int m=0; m < meshes.size(); ++m)
    {
        _character.loadMesh(meshes[m]);

        OptimizationPlot plot;
        _character.benchmarkSmoothers(
            plot, schedule, configs);

        const QualityHistogram& initHist = plot.initialHistogram();
        data[0][0+m*2] = initHist.minimumQuality();
        data[0][1+m*2] = initHist.harmonicMean();

        for(int s=0; s < smoothers.size(); ++s)
        {
            const QualityHistogram& hist =
                plot.implementations()[s].finalHistogram;

            data[s+1][0+m*2] = hist.minimumQuality();
            data[s+1][1+m*2] = hist.harmonicMean();
        }
    }


    // Print results
    vector<pair<string, int>> header = {{"Métriques", 1}};
    vector<pair<string, int>> subheader = {{"", 1}};
    for(int m=0; m < meshes.size(); ++m)
    {
        header.push_back({_meshNames[meshes[m]], 2});
        subheader.push_back({"Minimums", 1});
        subheader.push_back({"Moyennes", 1});
    }

    vector<string> lineNames = {"Initial"};
    for(const string& s : smoothers)
        lineNames.push_back(_translateSmoothers[s]);

    output(testName, header, subheader, lineNames, data);
}

void MastersTestSuite::smootherBlockSize(
        const string& testName,
        const string& mesh)
{
    // Test case description
    string sampler = "Texture";

    string evaluator = "Metric Conformity";

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
    _character.setMetricScaling(ADAPTATION_METRIC_K16);
    _character.setMetricAspectRatio(ADAPTATION_METRIC_A);

    _character.useEvaluator(evaluator);
    _character.setGlslEvaluatorThreadCount(EVALUATION_THREAD_COUNT_GLSL);
    _character.setCudaEvaluatorThreadCount(EVALUATION_THREAD_COUNT_CUDA);


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
    for(const string& smooth : smoothers)
    {
        header.push_back({_translateSmoothers[smooth], implementations.size()});
        for(const string& impl : implementations)
        {
            subheader.push_back({_translateImplementations[impl], 1});
        }
    }

    vector<string> lineNames;
    for(int tc : threadCounts)
        lineNames.push_back(to_string(tc));

    output(testName, header, subheader, lineNames, data);
}

void MastersTestSuite::smootherBlockSizeTetCube(
        const string& testName)
{
    string mesh = MESH_TETCUBE_K16;

    smootherBlockSize(testName, mesh);
}

void MastersTestSuite::smootherBlockSizeHexCube(
        const string& testName)
{
    string mesh = MESH_HEXCUBE_175K;

    smootherBlockSize(testName, mesh);
}

void MastersTestSuite::smootherSpeed(
        const std::string& testName,
        const std::string& mesh)
{
    // Test case description
    string evaluator = "Metric Conformity";

    string cpuSampler = "Local";
    string gpuSampler = "Texture";

    vector<string> smoothers = {
        "Spring Laplace",
        "Quality Laplace",
        "Spawn Search",
        "Nelder-Mead",
        "Multi Elem NM",
        "Gradient Descent",
        "Multi Elem GD",
        "Multi Pos GD",
        "Patch GD",
    };

    int nelderMeadPos = 3;
    int gradDescPos = 5;

    vector<string> implementations = {
        "Serial",
        "Thread",
        "GLSL",
        "CUDA"
    };

    vector<int> refImpls = {
        0,
        1,
        2,
        3,
        3, // Multi Elem NM
        5,
        5, // Multi Elem GD
        5, // Multi Pos GD
        5  // Patch GD
    };

    std::vector<Configuration> configs;

    for(int s=0; s <= nelderMeadPos; ++s)
        configs.push_back({cpuSampler, smoothers[s], "Serial"});
    configs.push_back({cpuSampler, smoothers[gradDescPos], "Serial"});
    int serialConfigEnd = configs.size();

    for(int s=0; s <= nelderMeadPos; ++s)
        configs.push_back({cpuSampler, smoothers[s], "Thread"});
    configs.push_back({cpuSampler, smoothers[gradDescPos], "Thread"});
    int threadConfigEnd = configs.size();

    for(int s=0; s < smoothers.size(); ++s)
        configs.push_back({gpuSampler, smoothers[s], "GLSL"});
    int glslConfigEnd = configs.size();

    for(int s=0; s < smoothers.size(); ++s)
        configs.push_back({gpuSampler, smoothers[s], "CUDA"});
    int cudaConfigEnd = configs.size();


    Schedule schedule;
    schedule.autoPilotEnabled = false;
    schedule.topoOperationEnabled = false;
    schedule.relocationPassCount = ADAPTATION_RELOC_PASS;


    // Setup test
    _character.loadMesh(mesh);

    _character.useSampler("Analytic");
    _character.setMetricScaling(ADAPTATION_METRIC_K16);
    _character.setMetricAspectRatio(ADAPTATION_METRIC_A);

    _character.useEvaluator(evaluator);
    _character.setGlslEvaluatorThreadCount(EVALUATION_THREAD_COUNT_GLSL);
    _character.setCudaEvaluatorThreadCount(EVALUATION_THREAD_COUNT_CUDA);

    _character.setGlslSmootherThreadCount(SMOOTHING_THREAD_COUNT_GLSL);
    _character.setCudaSmootherThreadCount(SMOOTHING_THREAD_COUNT_CUDA);


    // Run test
    Grid2D<double> data(4 + 3, smoothers.size());

    OptimizationPlot plot;
    _character.benchmarkSmoothers(
        plot, schedule, configs);

    for(int s=0; s < serialConfigEnd; ++s)
    {
        int i = s - 0;
        data[i][0] = plot.implementations()[s].passes.back().timeStamp;
    }

    for(int s=serialConfigEnd; s < threadConfigEnd; ++s)
    {
        int i = s - serialConfigEnd;
        data[i][1] = plot.implementations()[s].passes.back().timeStamp;
        data[i][4] = data[refImpls[i]][0] / data[i][1];
    }

    for(int s=threadConfigEnd; s < glslConfigEnd; ++s)
    {
        int i = s - threadConfigEnd;
        data[i][2] = plot.implementations()[s].passes.back().timeStamp;
        data[i][5] = data[refImpls[i]][0] / data[i][2];
    }

    for(int s=glslConfigEnd; s < cudaConfigEnd; ++s)
    {
        int i = s - glslConfigEnd;
        data[i][3] = plot.implementations()[s].passes.back().timeStamp;
        data[i][6] = data[refImpls[i]][0] / data[i][3];
    }


    // Print results
    vector<pair<string, int>> header = {
        {"Algorithmes", 1},
        {"Temps (ms)", implementations.size()},
        {"Accélérations", implementations.size()-1}
    };

    vector<pair<string, int>> subheader = {{"", 1}};
    for(const string& i : implementations)
        subheader.push_back({
            _translateImplementations[i], 1});

    for(int i=1; i < implementations.size(); ++i)
        subheader.push_back({
            _translateImplementations[implementations[i]], 1});

    vector<string> lineNames;
    for(const string& smooth : smoothers)
        lineNames.push_back(_translateSmoothers[smooth]);

    output(testName, header, subheader, lineNames, data);
}

void MastersTestSuite::smootherSpeedTetCube(
        const string& testName)
{
    string mesh = MESH_TETCUBE_K16;

    smootherSpeed(testName, mesh);
}

void MastersTestSuite::smootherSpeedHexCube(
        const string& testName)
{
    string mesh = MESH_HEXCUBE_175K;

    smootherSpeed(testName, mesh);
}

void MastersTestSuite::relocationScaling(
        const string& testName)
{
    // Test case description
    string evaluator = "Metric Conformity";

    vector<Configuration> configs;
    configs.push_back(Configuration{"Local", "Gradient Descent", "Serial"});
    configs.push_back(Configuration{"Local", "Gradient Descent", "Thread"});
    configs.push_back(Configuration{"Texture", "Patch GD", "GLSL"});
    configs.push_back(Configuration{"Texture", "Patch GD", "CUDA"});

    Schedule schedule;
    schedule.autoPilotEnabled = false;
    schedule.topoOperationEnabled = false;
    schedule.relocationPassCount = ADAPTATION_RELOC_PASS;
    schedule.topoOperationPassCount = ADAPTATION_TOPO_PASS;


    // Setup test
    _character.setMetricAspectRatio(ADAPTATION_METRIC_A);

    _character.useEvaluator(evaluator);
    _character.setGlslEvaluatorThreadCount(EVALUATION_THREAD_COUNT_GLSL);
    _character.setCudaEvaluatorThreadCount(EVALUATION_THREAD_COUNT_CUDA);

    _character.setGlslSmootherThreadCount(SMOOTHING_THREAD_COUNT_GLSL);
    _character.setCudaSmootherThreadCount(SMOOTHING_THREAD_COUNT_CUDA);


    // Run test
    Grid2D<double> data(4, SPHERE_TARGET_SIZES.size());

    for(int s=0; s < SPHERE_TARGET_SIZES.size(); ++s)
    {
        int size = SPHERE_TARGET_SIZES[s];
        _character.setMetricScaling(sizeToScale(size));

        _character.loadMesh(QString(
            MESH_SCALING_BASE.c_str()).arg(s).toStdString());


        OptimizationPlot plot;
        _character.benchmarkSmoothers(
            plot, schedule, configs);

        for(int i=0; i < plot.implementations().size(); ++i)
        {
            data[s][i] = plot.implementations()[i]
                    .passes.back().timeStamp;
        }
    }


    // Print results
    vector<pair<string, int>> header = {
        {"Tailles", 1},
        {_translateImplementations["Serial"], 1},
        {_translateImplementations["Thread"], 1},
        {_translateImplementations["GLSL"], 1},
        {_translateImplementations["CUDA"], 1}};
    vector<pair<string, int>> subheader = {};

    vector<string> lineNames;
    for(int size : SPHERE_TARGET_SIZES)
        lineNames.push_back(to_string(size));

    output(testName, header, subheader, lineNames, data);
}
