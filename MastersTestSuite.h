#ifndef GPUMESH_MASTERSTESTSUITE
#define GPUMESH_MASTERSTESTSUITE

#include <map>
#include <memory>
#include <vector>
#include <functional>

class QTextDocument;

#include "DataStructures/OptionMap.h"

namespace cellar
{
    template<typename T>
    class Grid2D;
}

class GpuMeshCharacter;
class QualityHistogram;


class MastersTestSuite
{
public:
    explicit MastersTestSuite(
        GpuMeshCharacter& character);

    virtual ~MastersTestSuite();


    OptionMapDetails availableTests() const;

    void runTests(
            QTextDocument& reportDocument,
            const std::vector<std::string>& tests);


protected:    
    void setupAdaptedCube(
            double metricK,
            double metricA);

    void saveToFile(
            const std::string& results,
            const std::string& fileName) const;


    void output(
            const std::string& title,
            const std::vector<std::string>& header,
            const std::vector<QualityHistogram>& histograms);

    void saveCsvTable(
            const std::string& title,
            const std::vector<std::string>& header,
            const std::vector<QualityHistogram>& histograms);

    void saveLatexTable(
            const std::string& title,
            const std::vector<std::string>& header,
            const std::vector<QualityHistogram>& histograms);

    void saveReportTable(
            const std::string& title,
            const std::vector<std::string>& header,
            const std::vector<QualityHistogram>& histograms);


    void output(
            const std::string& title,
            const std::vector<std::pair<std::string, int>> header,
            const std::vector<std::pair<std::string, int>> subHeader,
            const std::vector<std::string> lineNames,
            const cellar::Grid2D<double>& data);

    void saveCsvTable(
            const std::string& title,
            const std::vector<std::pair<std::string, int>> header,
            const std::vector<std::pair<std::string, int>> subHeader,
            const std::vector<std::string> lineNames,
            const cellar::Grid2D<double>& data);

    void saveLatexTable(
            const std::string& title,
            const std::vector<std::pair<std::string, int>> header,
            const std::vector<std::pair<std::string, int>> subHeader,
            const std::vector<std::string> lineNames,
            const cellar::Grid2D<double>& data);

    void saveReportTable(
            const std::string& title,
            const std::vector<std::pair<std::string, int>> header,
            const std::vector<std::pair<std::string, int>> subHeader,
            const std::vector<std::string> lineNames,
            const cellar::Grid2D<double>& data);



    void evaluatorBlockSize(
            const std::string& testName);


    void metricCost(
            const std::string& testName,
            const std::string& mesh);

    void metricCostTetCube(
            const std::string& testName);

    void metricCostHexGrid(
            const std::string& testName);


    void metricPrecision(
            const std::string& testName);

    void texturePrecision(
            const std::string& testName);


    void nodeOrder(
            const std::string& testName);

    void smootherEfficacity(
            const std::string& testName);

    void smootherBlockSize(
            const std::string& testName);


    void smootherSpeedAnalytic(
            const std::string& testName);

    void smootherSpeedMeshTex(
            const std::string& testName);


    void scaling(
            const std::string& testName,
            bool enableRelocation);

    void scalingRelocOnly(
            const std::string& testName);

    void scalingTopoReloc(
            const std::string& testName);


private:
    GpuMeshCharacter& _character;

    typedef std::function<void(const std::string&)> MastersTestFunc;
    OptionMap<MastersTestFunc> _availableMastersTests;

    std::map<std::string, std::string> _translateSamplingTechniques;
    std::map<std::string, std::string> _translateImplementations;
    std::map<std::string, std::string> _translateSmoothers;

    std::map<std::string, std::string> _meshNames;

    QTextDocument* _reportDocument;
};

#endif // GPUMESH_MASTERSTESTSUITE