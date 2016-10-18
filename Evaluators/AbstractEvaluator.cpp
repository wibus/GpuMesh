#include "AbstractEvaluator.h"

#include <future>
#include <chrono>
#include <sstream>
#include <fstream>
#include <iomanip>

#include <CellarWorkbench/Misc/Log.h>

#include "DataStructures/GpuMesh.h"
#include "DataStructures/MeshCrew.h"
#include "DataStructures/QualityHistogram.h"
#include "Samplers/AbstractSampler.h"
#include "Measurers/AbstractMeasurer.h"

using namespace cellar;
using namespace std;

const std::string AbstractEvaluator::SERIAL_IMPL_NAME = "Serial";
const std::string AbstractEvaluator::THREAD_IMPL_NAME = "Thread";
const std::string AbstractEvaluator::GLSL_IMPL_NAME = "GLSL";
const std::string AbstractEvaluator::CUDA_IMPL_NAME = "CUDA";

const size_t AbstractEvaluator::WORKGROUP_SIZE = 256;
const size_t AbstractEvaluator::POLYHEDRON_TYPE_COUNT = 3;
const size_t AbstractEvaluator::MAX_GROUP_PARTICIPANTS =
        AbstractEvaluator::WORKGROUP_SIZE *
        AbstractEvaluator::POLYHEDRON_TYPE_COUNT;

const double AbstractEvaluator::VALIDITY_EPSILON = 1e-6;
const double AbstractEvaluator::MAX_INTEGER_VALUE = 2147483647.0;
const double AbstractEvaluator::MIN_QUALITY_PRECISION_DENOM = 4096.0;

const double AbstractEvaluator::MAX_QUALITY_VALUE =
        AbstractEvaluator::MAX_INTEGER_VALUE /
        (double) AbstractEvaluator::MAX_GROUP_PARTICIPANTS;

const glm::dmat3 AbstractEvaluator::Fr_TET_INV = glm::dmat3(
    glm::dvec3(1, 0, 0),
    glm::dvec3(-0.5773502691896257645091, 1.154700538379251529018, 0),
    glm::dvec3(-0.4082482904638630163662, -0.4082482904638630163662, 1.224744871391589049099));
const glm::dmat3 AbstractEvaluator::Fr_PRI_INV = glm::dmat3(
    glm::dvec3(1.0, 0.0, 0.0),
    glm::dvec3(-0.5773502691896257645091, 1.154700538379251529018, 0.0),
    glm::dvec3(0.0, 0.0, 1.0));
const glm::dmat3 AbstractEvaluator::Fr_HEX_INV = glm::dmat3(1.0);


// CUDA Drivers Interface
extern bool verboseCuda;
void evaluateCudaMeshQuality(
        size_t workgroupSize,
        size_t workgroupCount,
        QualityHistogram& histogram);


AbstractEvaluator::AbstractEvaluator(const std::string& shapeMeasuresShader,
                                     const installCudaFct installCuda) :
    _qualSsbo(0),
    _histSsbo(0),
    _evaluationShader(shapeMeasuresShader),
    _installCuda(installCuda),
    _implementationFuncs("Shape Measure Implementations")
{
/*
    static_assert(AbstractEvaluator::MAX_QUALITY_VALUE >=
                  AbstractEvaluator::MIN_QUALITY_PRECISION_DENOM,
                  "Shape measure on GPU may not be suffciently precise \
                   given this workgroup size.");
*/
    using namespace std::placeholders;
    _implementationFuncs.setDefault(CUDA_IMPL_NAME);
    _implementationFuncs.setContent({
      {string(SERIAL_IMPL_NAME),  ImplementationFunc(bind(&AbstractEvaluator::evaluateMeshQualitySerial, this, _1, _2, _3, _4))},
      {string(THREAD_IMPL_NAME),  ImplementationFunc(bind(&AbstractEvaluator::evaluateMeshQualityThread, this, _1, _2, _3, _4))},
      {string(GLSL_IMPL_NAME),    ImplementationFunc(bind(&AbstractEvaluator::evaluateMeshQualityGlsl,   this, _1, _2, _3, _4))},
      {string(CUDA_IMPL_NAME),    ImplementationFunc(bind(&AbstractEvaluator::evaluateMeshQualityCuda,   this, _1, _2, _3, _4))},
    });
}

AbstractEvaluator::~AbstractEvaluator()
{
    glDeleteBuffers(1, &_qualSsbo);
    glDeleteBuffers(1, &_histSsbo);
}

string AbstractEvaluator::evaluationShader() const
{
    return _evaluationShader;
}

OptionMapDetails AbstractEvaluator::availableImplementations() const
{
    return _implementationFuncs.details();
}

void AbstractEvaluator::initialize(
        const Mesh& mesh,
        const MeshCrew& crew)
{
    if(_samplingShader == crew.sampler().samplingShader() &&
       _measureShader == crew.measurer().measureShader() &&
       _qualSsbo != 0 && _histSsbo != 0)
        return;


    getLog().postMessage(new Message('I', false,
        "Initializing evaluator compute shader", "AbstractEvaluator"));

    _samplingShader == crew.sampler().samplingShader();
    _measureShader == crew.measurer().measureShader();

    // Shader setup
    _evaluationProgram.reset();
    crew.installPlugins(mesh, _evaluationProgram);

    _evaluationProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/glsl/compute/Evaluating/Evaluate.glsl"});

    _evaluationProgram.link();
    crew.setPluginGlslUniforms(mesh, _evaluationProgram);

    /*
    const GlProgramBinary& binary = _evaluationProgram.getBinary();
    std::ofstream file("Evaluator_binary.txt", std::ios_base::trunc);
    if(file.is_open())
    {
        file << "Length: " << binary.length << endl;
        file << "Format: " << binary.format << endl;
        file << "Binary ->" << endl;
        file.write(binary.binary, binary.length);
        file.close();
    }
    */

    // Shader storage
    glDeleteBuffers(1, &_qualSsbo);
    _qualSsbo = 0;
    glGenBuffers(1, &_qualSsbo);

    glDeleteBuffers(1, &_histSsbo);
    _histSsbo = 0;
    glGenBuffers(1, &_histSsbo);
}

void AbstractEvaluator::installPlugin(
        const Mesh& mesh,
        cellar::GlProgram& program) const
{
    std::vector<std::string> qualityInterface = {
        mesh.meshGeometryShaderName(),
        ":/glsl/compute/Evaluating/Base.glsl"
    };

    std::vector<std::string> shapeMeasure = {
        mesh.meshGeometryShaderName(),
        _evaluationShader
    };

    program.addShader(GL_COMPUTE_SHADER, qualityInterface);
    program.addShader(GL_COMPUTE_SHADER, shapeMeasure);

    _installCuda();
}

void AbstractEvaluator::setPluginGlslUniforms(
        const Mesh& mesh,
        const GlProgram& program) const
{
    // Seems to be a driver bug with inactive subroutines (NVIDIA GTX 780Ti )
    // The query API return correct info (omitting inactive subroutines uniforms),
    // but GlSubroutineUniformsiv() generate an error when it receives the deisred
    // number of subroutine nuiform locations (GL_ACTIVE_SUBROUTINE_UNIFORM_LOCATIONS).

    //program.setSubroutine(GL_COMPUTE_SHADER, "patchQualityUni", "patchQualityImpl");
}

void AbstractEvaluator::setPluginCudaUniforms(
        const Mesh& mesh) const
{

}

double AbstractEvaluator::tetQuality(
        const Mesh& mesh,
        const AbstractSampler& sampler,
        const AbstractMeasurer& measurer,
        const MeshTet& tet) const
{
    glm::dvec3 vp[] = {
        mesh.verts[tet.v[0]].p,
        mesh.verts[tet.v[1]].p,
        mesh.verts[tet.v[2]].p,
        mesh.verts[tet.v[3]].p,
    };

    return tetQuality(sampler, measurer, vp, tet);
}

double AbstractEvaluator::priQuality(
        const Mesh& mesh,
        const AbstractSampler& sampler,
        const AbstractMeasurer& measurer,
        const MeshPri& pri) const
{
    glm::dvec3 vp[] = {
        mesh.verts[pri.v[0]].p,
        mesh.verts[pri.v[1]].p,
        mesh.verts[pri.v[2]].p,
        mesh.verts[pri.v[3]].p,
        mesh.verts[pri.v[4]].p,
        mesh.verts[pri.v[5]].p,
    };

    return priQuality(sampler, measurer, vp, pri);
}

double AbstractEvaluator::hexQuality(
        const Mesh& mesh,
        const AbstractSampler& sampler,
        const AbstractMeasurer& measurer,
        const MeshHex& hex) const
{
    glm::dvec3 vp[] = {
        mesh.verts[hex.v[0]].p,
        mesh.verts[hex.v[1]].p,
        mesh.verts[hex.v[2]].p,
        mesh.verts[hex.v[3]].p,
        mesh.verts[hex.v[4]].p,
        mesh.verts[hex.v[5]].p,
        mesh.verts[hex.v[6]].p,
        mesh.verts[hex.v[7]].p,
    };

    return hexQuality(sampler, measurer, vp, hex);
}

double AbstractEvaluator::patchQuality(
            const Mesh& mesh,
            const AbstractSampler& sampler,
            const AbstractMeasurer& measurer,
            size_t vId) const
{
    const std::vector<MeshTet>& tets = mesh.tets;
    const std::vector<MeshPri>& pris = mesh.pris;
    const std::vector<MeshHex>& hexs = mesh.hexs;

    const MeshTopo& topo = mesh.topos[vId];

    size_t neigElemCount = topo.neighborElems.size();

    double patchWeight = 0.0;
    double patchQuality = 1.0;
    for(size_t n=0; n < neigElemCount; ++n)
    {
        const MeshNeigElem& neigElem = topo.neighborElems[n];

        switch(neigElem.type)
        {
        case MeshTet::ELEMENT_TYPE:
            accumulatePatchQuality(
                patchQuality, patchWeight,
                tetQuality(mesh, sampler, measurer, tets[neigElem.id]));
            break;

        case MeshPri::ELEMENT_TYPE:
            accumulatePatchQuality(
                patchQuality, patchWeight,
                priQuality(mesh, sampler, measurer, pris[neigElem.id]));
            break;

        case MeshHex::ELEMENT_TYPE:
            accumulatePatchQuality(
                patchQuality, patchWeight,
                hexQuality(mesh, sampler, measurer, hexs[neigElem.id]));
            break;
        }
    }

    return finalizePatchQuality(patchQuality, patchWeight);
}

bool AbstractEvaluator::assessMeasureValidy(
        const AbstractSampler& sampler,
        const AbstractMeasurer& measurer)
{
    Mesh mesh;
    mesh.verts.push_back(glm::dvec3(0, 0, 0));
    mesh.verts.push_back(glm::dvec3(1, 0, 0));
    mesh.verts.push_back(glm::dvec3(0.5, sqrt(3.0)/2, 0));
    mesh.verts.push_back(glm::dvec3(0.5, sqrt(3.0)/6, sqrt(2.0/3)));

    mesh.verts.push_back(glm::dvec3(0, 0, 0));
    mesh.verts.push_back(glm::dvec3(0, 1, 0));
    mesh.verts.push_back(glm::dvec3(0, 0.5, sqrt(3.0)/2));
    mesh.verts.push_back(glm::dvec3(1, 0, 0));
    mesh.verts.push_back(glm::dvec3(1, 1, 0));
    mesh.verts.push_back(glm::dvec3(1, 0.5, sqrt(3.0)/2));

    mesh.verts.push_back(glm::dvec3(0, 0, 0));
    mesh.verts.push_back(glm::dvec3(1, 0, 0));
    mesh.verts.push_back(glm::dvec3(1, 1, 0));
    mesh.verts.push_back(glm::dvec3(0, 1, 0));
    mesh.verts.push_back(glm::dvec3(0, 0, 1));
    mesh.verts.push_back(glm::dvec3(1, 0, 1));
    mesh.verts.push_back(glm::dvec3(1, 1, 1));
    mesh.verts.push_back(glm::dvec3(0, 1, 1));

    const MeshTet tet = MeshTet(0, 1, 2, 3);
    const MeshPri pri = MeshPri(4, 5, 6, 7, 8, 9);
    const MeshHex hex = MeshHex(10, 11, 12, 13, 14, 15, 16, 17);

    double regularTet = tetQuality(mesh, sampler, measurer, tet);
    double regularPri = priQuality(mesh, sampler, measurer, pri);
    double regularHex = hexQuality(mesh, sampler, measurer, hex);

    if(glm::abs(regularTet - 1.0) < VALIDITY_EPSILON &&
       glm::abs(regularPri - 1.0) < VALIDITY_EPSILON &&
       glm::abs(regularHex - 1.0) < VALIDITY_EPSILON)
    {
        getLog().postMessage(new Message('I', false,
            "Quality evaluator's measure is valid.", "AbstractEvaluator"));
        return true;
    }
    else
    {
        stringstream log;
        log.precision(20);
        log << "Quality evaluator's measure is invalid." << endl;
        log << "Regular tetrahedron quality: " << regularTet << endl;
        log << "Regular prism quality: " << regularPri << endl;
        log << "Regular hexahedron quality: " << regularHex << endl;
        getLog().postMessage(new Message('E', true, log.str(), "AbstractEvaluator"));

        assert(glm::abs(regularTet - 1.0) < VALIDITY_EPSILON);
        assert(glm::abs(regularPri - 1.0) < VALIDITY_EPSILON);
        assert(glm::abs(regularHex - 1.0) < VALIDITY_EPSILON);
        return false;
    }
}

void AbstractEvaluator::evaluateMesh(
        const Mesh& mesh,
        const AbstractSampler& sampler,
        const AbstractMeasurer& measurer,
        QualityHistogram& histogram,
        const std::string& implementationName) const
{
    histogram.clear();
    ImplementationFunc implementationFunc;
    if(_implementationFuncs.select(implementationName, implementationFunc))
    {
        if(implementationName == GLSL_IMPL_NAME)
        {
            mesh.updateGlslTopology();
            mesh.updateGlslVertices();
            sampler.updateGlslData(mesh);
        }
        else if(implementationName == CUDA_IMPL_NAME)
        {
            mesh.updateCudaTopology();
            mesh.updateCudaVertices();
            sampler.updateCudaData(mesh);
        }

        implementationFunc(mesh, sampler, measurer, histogram);

        if(implementationName == GLSL_IMPL_NAME)
        {
            sampler.clearGlslMemory(mesh);
            mesh.clearGlslMemory();
        }
        else if(implementationName == CUDA_IMPL_NAME)
        {
            sampler.clearCudaMemory(mesh);
            mesh.clearCudaMemory();
        }
    }
    else
    {
        histogram.setMinimumQuality(nan(""));
        histogram.setInvQualitySum(nan(""));
    }
}

void AbstractEvaluator::evaluateMeshQualitySerial(
        const Mesh& mesh,
        const AbstractSampler& sampler,
        const AbstractMeasurer& measurer,
        QualityHistogram& histogram) const
{
    int tetCount = mesh.tets.size();
    int priCount = mesh.pris.size();
    int hexCount = mesh.hexs.size();

    for(int i=0; i < tetCount; ++i)
        histogram.add(tetQuality(mesh, sampler, measurer, mesh.tets[i]));

    for(int i=0; i < priCount; ++i)
        histogram.add(priQuality(mesh, sampler, measurer, mesh.pris[i]));

    for(int i=0; i < hexCount; ++i)
        histogram.add(hexQuality(mesh, sampler, measurer, mesh.hexs[i]));
}

void AbstractEvaluator::evaluateMeshQualityThread(
        const Mesh& mesh,
        const AbstractSampler& sampler,
        const AbstractMeasurer& measurer,
        QualityHistogram& histogram) const
{
    int tetCount = mesh.tets.size();
    int priCount = mesh.pris.size();
    int hexCount = mesh.hexs.size();

    if((tetCount + priCount + hexCount) == 0)
    {
        return;
    }

    vector<future<QualityHistogram>> futures;
    uint coreCountHint = thread::hardware_concurrency();
    for(uint t=0; t < coreCountHint; ++t)
    {
        futures.push_back(async(launch::async, [&, t](){
            QualityHistogram coreHist(histogram.bucketCount());

            int tetBeg = (tetCount * t) / coreCountHint;
            int tetEnd = (tetCount * (t+1)) / coreCountHint;
            int priBeg = (priCount * t) / coreCountHint;
            int priEnd = (priCount * (t+1)) / coreCountHint;
            int hexBeg = (hexCount * t) / coreCountHint;
            int hexEnd = (hexCount * (t+1)) / coreCountHint;

            for(int i=tetBeg; i < tetEnd; ++i)
                coreHist.add(tetQuality(mesh, sampler, measurer, mesh.tets[i]));

            for(int i=priBeg; i < priEnd; ++i)
                coreHist.add(priQuality(mesh, sampler, measurer, mesh.pris[i]));

            for(int i=hexBeg; i < hexEnd; ++i)
                coreHist.add(hexQuality(mesh, sampler, measurer, mesh.hexs[i]));

            return coreHist;
        }));
    }


    // Combine workers' results
    for(uint i=0; i < coreCountHint; ++i)
    {
        QualityHistogram coreHist = futures[i].get();
        histogram.merge(coreHist);
    }
}

void AbstractEvaluator::evaluateMeshQualityGlsl(
        const Mesh& mesh,
        const AbstractSampler& sampler,
        const AbstractMeasurer& measurer,
        QualityHistogram& histogram) const
{
    if(_qualSsbo == 0 || _histSsbo == 0)
    {
        getLog().postMessage(new Message('E', false,
            "Evalator needs to be initialized first"\
            " evaluateMeshQualityGlsl().", "AbstractEvaluator"));
        return;
    }

    size_t tetCount = mesh.tets.size();
    size_t priCount = mesh.pris.size();
    size_t hexCount = mesh.hexs.size();
    size_t elemCount = tetCount + priCount + hexCount;
    size_t maxSize = glm::max(glm::max(tetCount, priCount), hexCount);
    size_t workgroupCount = ceil(maxSize / (double)WORKGROUP_SIZE);

    if(elemCount == 0)
    {
        return;
    }

    struct QualBuff
    {
        QualBuff() :
            qualMin(MAX_INTEGER_VALUE),
            invSum(0.0)
        {}

        GLint qualMin;
        GLfloat invSum;
    };

    // Workgroup integer accum VS. Mesh float accum for mean quality computation
    //
    //    Using atomic integer operations on an array (one int per workgroup)
    // to compute mesh mean quality is faster than using a single floating point
    // variable updated by all the invocations.
    //
    //    Not to metion that using a single float accumulator gives inacurate
    // results while the workgroup specific integer accumulators gives the
    // exact same result as the double floating point CPU computations.
    QualBuff qualBuff;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _qualSsbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(qualBuff), &qualBuff, GL_DYNAMIC_READ);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);


    std::vector<GLint> histBuff(histogram.bucketCount(), 0);
    size_t histSize = sizeof(decltype(histBuff.front())) * histBuff.size();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _histSsbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, histSize, histBuff.data(), GL_DYNAMIC_READ);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);


    _evaluationProgram.pushProgram();

    mesh.bindGlShaderStorageBuffers();
    GLuint qualsBinding = mesh.glBufferBinding(
        EBufferBinding::EVALUATE_QUAL_BUFFER_BINDING);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, qualsBinding, _qualSsbo);
    GLuint histBinding = mesh.glBufferBinding(
        EBufferBinding::EVALUATE_HIST_BUFFER_BINDING);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, histBinding, _histSsbo);

    setPluginGlslUniforms(mesh, _evaluationProgram);
    sampler.setPluginGlslUniforms(mesh, _evaluationProgram);
    measurer.setPluginGlslUniforms(mesh, _evaluationProgram);

    // When Nvidia compiler bug about linker crash on hists.length() is fixed
    // Remove this uniform and get hists length directly from the SSBO.
    GLint hists_lenght_loc = glGetUniformLocation(
                _evaluationProgram.id(), "hists_length");
    glUniform1f(hists_lenght_loc, histogram.bucketCount());

    // Simulatenous and separate elem evaluation deliver the same performance
    // Separate program series gives a tiny, not stable speed boost.
    // (tested on a parametric pri/hex mesh)
    glDispatchCompute(workgroupCount, 1, 1);
    _evaluationProgram.popProgram();


    // Fetch workgroup's statistics from GPU
    // We are using glMapBuffer since glGetBufferData seems to update the output
    // concurently while were are computing mesh quality mean. glMemoryBarrier
    // looks like having no effect on this (tried with GL_ALL_BARRIER_BITS
    // before and after the call to glGetBufferSubData).
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _qualSsbo);
    QualBuff* quals = (QualBuff*) glMapBufferRange(
        GL_SHADER_STORAGE_BUFFER, 0, sizeof(qualBuff), GL_MAP_READ_BIT);

    // Get minimum quality
    histogram.setMinimumQuality(quals->qualMin / MAX_INTEGER_VALUE);

    // Get inverse quality log sum
    histogram.setInvQualitySum(quals->invSum);

    histogram.setSampleCount(elemCount);

    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);


    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _histSsbo);
    GLint* hist = (GLint*) glMapBufferRange(
        GL_SHADER_STORAGE_BUFFER, 0, histSize, GL_MAP_READ_BIT);
    for(size_t i=0; i < histogram.bucketCount(); ++i)
        histogram.setBucket(i, hist[i]);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void AbstractEvaluator::evaluateMeshQualityCuda(
        const Mesh& mesh,
        const AbstractSampler& sampler,
        const AbstractMeasurer& measurer,
        QualityHistogram& histogram) const
{
    if(_qualSsbo == 0 || _histSsbo == 0)
    {
        getLog().postMessage(new Message('E', false,
            "Evalator needs to be initialized first"\
            " evaluateMeshQualityGlsl().", "AbstractEvaluator"));
        return;
    }

    size_t tetCount = mesh.tets.size();
    size_t priCount = mesh.pris.size();
    size_t hexCount = mesh.hexs.size();
    size_t elemCount = tetCount + priCount + hexCount;
    size_t maxSize = glm::max(glm::max(tetCount, priCount), hexCount);
    size_t workgroupCount = ceil(maxSize / (double)WORKGROUP_SIZE);

    if(elemCount == 0)
    {
        return;
    }

    setPluginCudaUniforms(mesh);
    sampler.setPluginCudaUniforms(mesh);
    measurer.setPluginCudaUniforms(mesh);

    evaluateCudaMeshQuality(
        WORKGROUP_SIZE,
        workgroupCount,
        histogram);

    histogram.setSampleCount(elemCount);
}

struct EvalBenchmarkStats
{
    string impl;
    int cycleCount;
    chrono::high_resolution_clock::rep totalTime;
    chrono::high_resolution_clock::rep averageTime;
};

void AbstractEvaluator::benchmark(
        const Mesh& mesh,
        const AbstractSampler& sampler,
        const AbstractMeasurer& measurer,
        const map<string, int>& cycleCounts)
{
    int markCount = 100 / cycleCounts.size();
    using std::chrono::high_resolution_clock;
    high_resolution_clock::time_point tStart;
    high_resolution_clock::time_point tEnd;

    QualityHistogram histogram;

    int measuredImplementations = -1;
    std::vector<EvalBenchmarkStats> statsVec;
    for(auto& impl : _implementationFuncs.details().options)
    {
        ++measuredImplementations;

        int cycleCount;
        auto cycleIt = cycleCounts.find(impl);
        if(cycleIt != cycleCounts.end())
        {
            cycleCount = cycleIt->second;

            if(cycleCount == 0)
                continue;
        }
        else
        {
            getLog().postMessage(new Message('W', false,
               "No cycle count defined for " + impl +
               ". Skipping this implementation...",
               "AbstractEvaluator"));
            continue;
        }

        getLog().postMessage(new Message('I', false,
           "Benchmarking "+ impl +" implementation (running "
            + to_string(cycleCount) + " cycles)",
           "AbstractEvaluator"));


        ImplementationFunc implementationFunc;
        if(_implementationFuncs.select(impl, implementationFunc))
        {
            if(impl == GLSL_IMPL_NAME)
            {
                mesh.updateGlslTopology();
                mesh.updateGlslVertices();
                sampler.updateGlslData(mesh);
            }
            else if(impl == CUDA_IMPL_NAME)
            {
                verboseCuda = false;
                mesh.updateCudaTopology();
                mesh.updateCudaVertices();
                sampler.updateCudaData(mesh);
            }

            high_resolution_clock::duration totalTime(0);
            size_t markSize = cycleCount / glm::min(markCount, cycleCount);
            for(size_t i=0, m=0; i < cycleCount; ++i)
            {
                histogram.clear();

                tStart = high_resolution_clock::now();
                implementationFunc(mesh, sampler, measurer, histogram);
                tEnd = high_resolution_clock::now();

                totalTime += (tEnd - tStart);

                if(i == m)
                {
                    float progressSize = 100.0f / cycleCounts.size();
                    float progressBase = measuredImplementations * 100.0f /
                        cycleCounts.size();
                    int progress = progressBase +
                        progressSize * (i / (float) cycleCount);

                    getLog().postMessage(new Message('I', false,
                       "Benchmark progress : " + to_string(progress) + "%\t" +
                       "(min=" + to_string(histogram.minimumQuality()) +
                       ", mean=" + to_string(histogram.harmonicMean()) + ")",
                       "AbstractEvaluator"));
                    m += markSize;
                }
            }

            if(impl == GLSL_IMPL_NAME)
            {
                sampler.clearGlslMemory(mesh);
                mesh.clearGlslMemory();
            }
            else if(impl == CUDA_IMPL_NAME)
            {
                sampler.clearCudaMemory(mesh);
                mesh.clearCudaMemory();
                verboseCuda = true;
            }

            EvalBenchmarkStats stats;
            stats.impl = impl;
            stats.cycleCount = cycleCount;
            stats.totalTime = totalTime.count();
            stats.averageTime = totalTime.count() / cycleCount;
            statsVec.push_back(stats);
        }
    }


    // Get minimums for ratio computations
    double minTime = statsVec[0].averageTime;
    for(size_t i = 1; i < statsVec.size(); ++i)
    {
        minTime = glm::min(minTime, double(statsVec[i].averageTime));
    }


    // Build ratio strings
    stringstream nameStream;
    stringstream timeStream;
    stringstream normTimeStream;
    for(size_t i = 0; i < statsVec.size(); ++i)
    {
        nameStream << statsVec[i].impl << ":";
        timeStream << int(statsVec[i].averageTime / 1000.0) << ":";
        normTimeStream << fixed << setprecision(3) <<
                          statsVec[i].averageTime / minTime << ":";
    }
    string nameString = nameStream.str(); nameString.back() = ' ';
    string timeString = timeStream.str(); timeString.back() = ' ';
    string normTimeString = normTimeStream.str(); normTimeString.back() = ' ';


    getLog().postMessage(new Message('I', false,
        "Shape measure time ratio (us) :\t "
         + nameString + "\t = "
         + timeString + "\t = "
         + normTimeString,
        "AbstractEvaluator"));
}

void AbstractEvaluator::accumulatePatchQuality(
        double& patchQuality,
        double& patchWeight,
        double elemQuality) const
{
    if(patchQuality > 0.0 &&  elemQuality > 0.0)
    {
        patchWeight += 1.0;
        patchQuality += 1/elemQuality;
    }
    else
    {
        patchWeight = 0.0;
        patchQuality = min(patchQuality, elemQuality);
    }
}

double AbstractEvaluator::finalizePatchQuality(
        double patchQuality,
        double patchWeight) const
{
    if(patchWeight != 0.0)
        return patchWeight/patchQuality;
    else
        return patchQuality;
}
