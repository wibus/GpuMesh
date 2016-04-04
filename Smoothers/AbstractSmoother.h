#ifndef GPUMESH_ABSTRACTSMOOTHER
#define GPUMESH_ABSTRACTSMOOTHER

#include <functional>

#include <CellarWorkbench/GL/GlProgram.h>

#include "DataStructures/Mesh.h"
#include "DataStructures/OptionMap.h"
#include "DataStructures/OptimizationPlot.h"

class MeshCrew;


struct IndependentDispatch
{
    IndependentDispatch(uint base, uint size, uint wgc) :
        base(base), size(size), workgroupCount(wgc) {}

    // Independent set range
    uint base;
    uint size;

    // Compute Dispatch Size
    uint workgroupCount;
};


class AbstractSmoother
{
protected:
    typedef void (*installCudaFct)(void);
    AbstractSmoother(const installCudaFct installCuda);

public:
    virtual ~AbstractSmoother();

    virtual OptionMapDetails availableImplementations() const;

    virtual void smoothMesh(
            Mesh& mesh,
            const MeshCrew& crew,
            const std::string& implementationName,
            int minIteration,
            double moveFactor,
            double gainThreshold);

    virtual void smoothMeshSerial(
            Mesh& mesh,
            const MeshCrew& crew) = 0;

    virtual void smoothMeshThread(
            Mesh& mesh,
            const MeshCrew& crew) = 0;

    virtual void smoothMeshGlsl(
            Mesh& mesh,
            const MeshCrew& crew) = 0;

    virtual void smoothMeshCuda(
            Mesh& mesh,
            const MeshCrew& crew) = 0;

    virtual void benchmark(
            Mesh& mesh,
            const MeshCrew& crew,
            const std::map<std::string, bool>& activeImpls,
            int minIteration,
            double moveFactor,
            double gainThreshold,
            OptimizationPlot& outPlot);


protected:
    virtual void initializeProgram(
            Mesh& mesh,
            const MeshCrew& crew) = 0;

    virtual void printSmoothingParameters(
            const Mesh& mesh,
            OptimizationPlot& plot) const = 0;

    virtual void organizeDispatches(
            const Mesh& mesh,
            size_t workgroupSize,
            std::vector<IndependentDispatch>& dispatches) const;

    virtual bool isSmoothable(
            const Mesh& mesh,
            size_t vId) const;

    bool evaluateMeshQualitySerial(Mesh& mesh, const MeshCrew& crew);
    bool evaluateMeshQualityThread(Mesh& mesh, const MeshCrew& crew);
    bool evaluateMeshQualityGlsl(Mesh& mesh, const MeshCrew& crew);
    bool evaluateMeshQualityCuda(Mesh& mesh, const MeshCrew& crew);
    bool evaluateMeshQuality(Mesh& mesh, const MeshCrew& crew, int impl);



    std::string smoothingUtilsShader() const;


    installCudaFct _installCudaSmoother;

    int _minIteration;
    double _moveCoeff;
    double _gainThreshold;

    int _smoothPassId;
    double _lastAvgQuality;
    double _lastMinQuality;

private:
    std::string _smoothingUtilsShader;

    std::chrono::high_resolution_clock::time_point _implBeginTimeStamp;
    OptimizationImpl _currentImplementation;

    typedef std::function<void(Mesh&, const MeshCrew&)> ImplementationFunc;
    OptionMap<ImplementationFunc> _implementationFuncs;
};

#endif // GPUMESH_ABSTRACTSMOOTHER
