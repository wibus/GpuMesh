#include "AbstractEvaluator.h"

#include <future>
#include <chrono>
#include <sstream>
#include <fstream>
#include <iomanip>

#include <CellarWorkbench/Misc/Log.h>

#include "DataStructures/GpuMesh.h"
#include "DataStructures/MeshCrew.h"
#include "Discretizers/AbstractDiscretizer.h"
#include "Measurers/AbstractMeasurer.h"

using namespace cellar;
using namespace std;


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


AbstractEvaluator::AbstractEvaluator(const std::string& shapeMeasuresShader) :
    _qualSsbo(0),
    _evaluationShader(shapeMeasuresShader),
    _implementationFuncs("Shape Measure Implementations")
{
/*
    static_assert(AbstractEvaluator::MAX_QUALITY_VALUE >=
                  AbstractEvaluator::MIN_QUALITY_PRECISION_DENOM,
                  "Shape measure on GPU may not be suffciently precise \
                   given this workgroup size.");
*/
    using namespace std::placeholders;
    _implementationFuncs.setDefault("Thread");
    _implementationFuncs.setContent({
      {string("Serial"),  ImplementationFunc(bind(&AbstractEvaluator::evaluateMeshQualitySerial, this, _1, _2, _3, _4, _5))},
      {string("Thread"),  ImplementationFunc(bind(&AbstractEvaluator::evaluateMeshQualityThread, this, _1, _2, _3, _4, _5))},
      {string("GLSL"),    ImplementationFunc(bind(&AbstractEvaluator::evaluateMeshQualityGlsl,   this, _1, _2, _3, _4, _5))},
    });
}

AbstractEvaluator::~AbstractEvaluator()
{
    glDeleteBuffers(1, &_qualSsbo);
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
    if(_discretizationShader == crew.discretizer().discretizationShader() &&
       _measureShader == crew.measurer().measureShader() &&
       _qualSsbo != 0)
        return;


    getLog().postMessage(new Message('I', false,
        "Initializing evaluator compute shader", "AbstractEvaluator"));

    _discretizationShader == crew.discretizer().discretizationShader();
    _measureShader == crew.measurer().measureShader();


    // Shader setup
    _evaluationProgram.clearShaders();
    crew.installPlugins(mesh, _evaluationProgram);

    _evaluationProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/glsl/compute/Evaluating/Evaluate.glsl"});

    _evaluationProgram.link();
    crew.setPluginUniforms(mesh, _evaluationProgram);

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

    // Shader storage quality blocks
    glDeleteBuffers(1, &_qualSsbo);
    _qualSsbo = 0;
    glGenBuffers(1, &_qualSsbo);
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
}

void AbstractEvaluator::setPluginUniforms(
        const Mesh& mesh,
        cellar::GlProgram& program) const
{

}

void AbstractEvaluator::setupPluginExecution(
        const Mesh& mesh,
        const GlProgram& program) const
{
}

double AbstractEvaluator::tetQuality(
        const Mesh& mesh,
        const AbstractDiscretizer& discretizer,
        const AbstractMeasurer& measurer,
        const MeshTet& tet) const
{
    const glm::dvec3 vp[] = {
        mesh.verts[tet.v[0]],
        mesh.verts[tet.v[1]],
        mesh.verts[tet.v[2]],
        mesh.verts[tet.v[3]],
    };

    return tetQuality(discretizer, measurer, vp);
}

double AbstractEvaluator::priQuality(
        const Mesh& mesh,
        const AbstractDiscretizer& discretizer,
        const AbstractMeasurer& measurer,
        const MeshPri& pri) const
{
    const glm::dvec3 vp[] = {
        mesh.verts[pri.v[0]],
        mesh.verts[pri.v[1]],
        mesh.verts[pri.v[2]],
        mesh.verts[pri.v[3]],
        mesh.verts[pri.v[4]],
        mesh.verts[pri.v[5]]
    };

    return priQuality(discretizer, measurer, vp);
}

double AbstractEvaluator::hexQuality(
        const Mesh& mesh,
        const AbstractDiscretizer& discretizer,
        const AbstractMeasurer& measurer,
        const MeshHex& hex) const
{
    const glm::dvec3 vp[] = {
        mesh.verts[hex.v[0]],
        mesh.verts[hex.v[1]],
        mesh.verts[hex.v[2]],
        mesh.verts[hex.v[3]],
        mesh.verts[hex.v[4]],
        mesh.verts[hex.v[5]],
        mesh.verts[hex.v[6]],
        mesh.verts[hex.v[7]]
    };

    return hexQuality(discretizer, measurer, vp);
}

double AbstractEvaluator::patchQuality(
            const Mesh& mesh,
            const AbstractDiscretizer& discretizer,
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
                tetQuality(mesh, discretizer, measurer, tets[neigElem.id]));
            break;

        case MeshPri::ELEMENT_TYPE:
            accumulatePatchQuality(
                patchQuality, patchWeight,
                priQuality(mesh, discretizer, measurer, pris[neigElem.id]));
            break;

        case MeshHex::ELEMENT_TYPE:
            accumulatePatchQuality(
                patchQuality, patchWeight,
                hexQuality(mesh, discretizer, measurer, hexs[neigElem.id]));
            break;
        }
    }

    return finalizePatchQuality(patchQuality, patchWeight);
}

bool AbstractEvaluator::assessMeasureValidy(
        const AbstractDiscretizer& discretizer,
        const AbstractMeasurer& measurer)
{
    Mesh mesh;
    mesh.verts.push_back(glm::dvec3(0, 0, 0));
    mesh.verts.push_back(glm::dvec3(1, 0, 0));
    mesh.verts.push_back(glm::dvec3(0.5, sqrt(3.0)/6, sqrt(2.0/3)));
    mesh.verts.push_back(glm::dvec3(0.5, sqrt(3.0)/2, 0));

    mesh.verts.push_back(glm::dvec3(0, 0, 0));
    mesh.verts.push_back(glm::dvec3(1, 0, 0));
    mesh.verts.push_back(glm::dvec3(0, 1, 0));
    mesh.verts.push_back(glm::dvec3(1, 1, 0));
    mesh.verts.push_back(glm::dvec3(0, 0.5, sqrt(3.0)/2));
    mesh.verts.push_back(glm::dvec3(1, 0.5, sqrt(3.0)/2));

    mesh.verts.push_back(glm::dvec3(0, 0, 0));
    mesh.verts.push_back(glm::dvec3(1, 0, 0));
    mesh.verts.push_back(glm::dvec3(0, 1, 0));
    mesh.verts.push_back(glm::dvec3(1, 1, 0));
    mesh.verts.push_back(glm::dvec3(0, 0, 1));
    mesh.verts.push_back(glm::dvec3(1, 0, 1));
    mesh.verts.push_back(glm::dvec3(0, 1, 1));
    mesh.verts.push_back(glm::dvec3(1, 1, 1));

    const MeshTet tet = MeshTet(0, 1, 2, 3);
    const MeshPri pri = MeshPri(4, 5, 6, 7, 8, 9);
    const MeshHex hex = MeshHex(10, 11, 12, 13, 14, 15, 16, 17);

    double regularTet = tetQuality(mesh, discretizer, measurer, tet);
    double regularPri = priQuality(mesh, discretizer, measurer, pri);
    double regularHex = hexQuality(mesh, discretizer, measurer, hex);

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
        const AbstractDiscretizer& discretizer,
        const AbstractMeasurer& measurer,
        double& minQuality,
        double& qualityMean,
        const std::string& implementationName) const
{
    ImplementationFunc implementationFunc;
    if(_implementationFuncs.select(implementationName, implementationFunc))
    {
        implementationFunc(mesh, discretizer, measurer, minQuality, qualityMean);
    }
    else
    {
        minQuality = nan("");
        qualityMean = nan("");
    }
}

void AbstractEvaluator::evaluateMeshQualitySerial(
        const Mesh& mesh,
        const AbstractDiscretizer& discretizer,
        const AbstractMeasurer& measurer,
        double& minQuality,
        double& qualityMean) const
{
    int tetCount = mesh.tets.size();
    int priCount = mesh.pris.size();
    int hexCount = mesh.hexs.size();
    int elemCount = tetCount + priCount + hexCount;

    std::vector<double> qualities(elemCount);
    int idx = 0;

    for(int i=0; i < tetCount; ++i, ++idx)
        qualities[idx] = tetQuality(mesh, discretizer, measurer, mesh.tets[i]);

    for(int i=0; i < priCount; ++i, ++idx)
        qualities[idx] = priQuality(mesh, discretizer, measurer, mesh.pris[i]);

    for(int i=0; i < hexCount; ++i, ++idx)
        qualities[idx] = hexQuality(mesh, discretizer, measurer, mesh.hexs[i]);


    minQuality = 1.0;
    qualityMean = 0.0;
    for(int i=0; i < elemCount; ++i)
    {
        double qual = qualities[i];
        minQuality = glm::min(qual, minQuality);
        qualityMean += qual;
    }
    qualityMean /= elemCount;
}

void AbstractEvaluator::evaluateMeshQualityThread(
        const Mesh& mesh,
        const AbstractDiscretizer& discretizer,
        const AbstractMeasurer& measurer,
        double& minQuality,
        double& qualityMean) const
{
    int tetCount = mesh.tets.size();
    int priCount = mesh.pris.size();
    int hexCount = mesh.hexs.size();
    int elemCount = tetCount + priCount + hexCount;

    vector<future<pair<double, double>>> futures;
    uint coreCountHint = thread::hardware_concurrency();
    for(uint t=0; t < coreCountHint; ++t)
    {
        futures.push_back(async(launch::async, [&, t](){

            int tetBeg = (tetCount * t) / coreCountHint;
            int tetEnd = (tetCount * (t+1)) / coreCountHint;
            int tetBatchSize = tetEnd - tetBeg;
            int priBeg = (priCount * t) / coreCountHint;
            int priEnd = (priCount * (t+1)) / coreCountHint;
            int priBatchSize = priEnd - priBeg;
            int hexBeg = (hexCount * t) / coreCountHint;
            int hexEnd = (hexCount * (t+1)) / coreCountHint;
            int hexBatchSize = hexEnd - hexBeg;
            int totalBatchSize = tetBatchSize + priBatchSize + hexBatchSize;

            std::vector<double> qualities(totalBatchSize);
            int idx = 0;

            for(int i=tetBeg; i < tetEnd; ++i, ++idx)
                qualities[idx] = tetQuality(mesh, discretizer, measurer, mesh.tets[i]);

            for(int i=priBeg; i < priEnd; ++i, ++idx)
                qualities[idx] = priQuality(mesh, discretizer, measurer, mesh.pris[i]);

            for(int i=hexBeg; i < hexEnd; ++i, ++idx)
                qualities[idx] = hexQuality(mesh, discretizer, measurer, mesh.hexs[i]);

            double futureMinQuality = 1.0;
            double futureQualityMean = 0.0;
            for(int i=0; i < totalBatchSize; ++i)
            {
                double qual = qualities[i];
                futureMinQuality = glm::min(qual, futureMinQuality);
                futureQualityMean += qual;
            }

            return make_pair(futureMinQuality, futureQualityMean);
        }));
    }


    // Combine workers' results
    minQuality = 1.0;
    qualityMean = 0.0;
    for(uint i=0; i < coreCountHint; ++i)
    {
        pair<double, double> stats = futures[i].get();
        minQuality = glm::min(stats.first, minQuality);
        qualityMean += stats.second;
    }
    qualityMean /= elemCount;
}

void AbstractEvaluator::evaluateMeshQualityGlsl(
        const Mesh& mesh,
        const AbstractDiscretizer& discretizer,
        const AbstractMeasurer& measurer,
        double& minQuality,
        double& qualityMean) const
{
    if(_qualSsbo == 0)
    {
        getLog().postMessage(new Message('E', false,
            "Evalator needs to be initialized before calling"\
            " evaluateMeshQualityGlsl().", "AbstractEvaluator"));
        return;
    }

    size_t tetCount = mesh.tets.size();
    size_t priCount = mesh.pris.size();
    size_t hexCount = mesh.hexs.size();
    size_t elemCount = tetCount + priCount + hexCount;
    size_t maxSize = glm::max(glm::max(tetCount, priCount), hexCount);
    size_t workgroupCount = ceil(maxSize / (double)WORKGROUP_SIZE);

    // Workgroup integer accum VS. Mesh float accum for mean quality computation
    //
    //    Using atomic integer operations on an array (one int per workgroup)
    // to compute mesh mean quality is faster than using a single floating point
    // variable updated by all the invocations.
    //
    //    Not to metion that using a single float accumulator gives inacurate
    // results while the workgroup specific integer accumulators gives the
    // exact same result a the double floating point CPU computations.
    std::vector<GLint> qualBuff(1 + workgroupCount, 0);
    qualBuff[0] = GLint(MAX_INTEGER_VALUE);
    size_t qualSize = sizeof(decltype(qualBuff.front())) * qualBuff.size();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _qualSsbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, qualSize, qualBuff.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);


    _evaluationProgram.pushProgram();

    mesh.bindShaderStorageBuffers();
    discretizer.setupPluginExecution(mesh, _evaluationProgram);
    measurer.setupPluginExecution(mesh, _evaluationProgram);
    GLuint qualsBinding = mesh.bufferBinding(
        EBufferBinding::EVALUATE_QUALS_BUFFER_BINDING);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, qualsBinding, _qualSsbo);


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
    GLint* data = (GLint*) glMapBufferRange(
        GL_SHADER_STORAGE_BUFFER, 0, qualSize, GL_MAP_READ_BIT);

    // Get minimum quality
    minQuality = data[0] / MAX_INTEGER_VALUE;

    // Combine workgroups' mean
    size_t i = 0;
    qualityMean = 0.0;
    while(i <= workgroupCount)
        qualityMean += data[++i];
    qualityMean /= MAX_QUALITY_VALUE * elemCount;

    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
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
        const AbstractDiscretizer& discretizer,
        const AbstractMeasurer& measurer,
        const map<string, int>& cycleCounts)
{
    int markCount = 100 / cycleCounts.size();
    using std::chrono::high_resolution_clock;
    high_resolution_clock::time_point tStart;
    high_resolution_clock::time_point tEnd;


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
            high_resolution_clock::duration totalTime(0);
            size_t markSize = cycleCount / glm::min(markCount, cycleCount);
            for(size_t i=0, m=0; i < cycleCount; ++i)
            {
                double minQual, qualMean;

                tStart = high_resolution_clock::now();
                implementationFunc(mesh, discretizer, measurer, minQual, qualMean);
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
                       "(min=" + to_string(minQual) + ", mean=" + to_string(qualMean) + ")",
                       "AbstractEvaluator"));
                    m += markSize;
                }
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
    patchQuality = glm::min(
        glm::min(patchQuality, elemQuality),  // If sign(patch) != sign(elem)
        glm::min(patchQuality * elemQuality,  // If sign(patch) & sign(elem) > 0
                 patchQuality + elemQuality));// If sign(patch) & sign(elem) < 0
}

double AbstractEvaluator::finalizePatchQuality(
        double patchQuality,
        double patchWeight) const
{
    return patchQuality;
}
