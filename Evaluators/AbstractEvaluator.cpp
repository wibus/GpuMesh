#include "AbstractEvaluator.h"

#include <future>
#include <chrono>
#include <sstream>
#include <iomanip>

#include <CellarWorkbench/Misc/Log.h>

#include "DataStructures/GpuMesh.h"

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

AbstractEvaluator::AbstractEvaluator(const std::string& shapeMeasuresShader) :
    _initialized(false),
    _computeSimultaneously(true),
    _shapeMeasuresShader(shapeMeasuresShader),
    _qualSsbo(0),
    _implementationFuncs("Shape Measure Implementations")
{
    static_assert(AbstractEvaluator::MAX_QUALITY_VALUE >=
                  AbstractEvaluator::MIN_QUALITY_PRECISION_DENOM,
                  "Shape measure on GPU may not be suffciently precise \
                   given this workgroup size.");

    using namespace std::placeholders;
    _implementationFuncs.setDefault("Thread");
    _implementationFuncs.setContent({
      {string("Serial"),  ImplementationFunc(bind(&AbstractEvaluator::evaluateMeshQualitySerial, this, _1, _2, _3))},
      {string("Thread"),  ImplementationFunc(bind(&AbstractEvaluator::evaluateMeshQualityThread, this, _1, _2, _3))},
      {string("GLSL"),    ImplementationFunc(bind(&AbstractEvaluator::evaluateMeshQualityGlsl, this, _1, _2, _3))},
    });
}

AbstractEvaluator::~AbstractEvaluator()
{
    glDeleteBuffers(1, &_qualSsbo);
}

OptionMapDetails AbstractEvaluator::availableImplementations() const
{
    return _implementationFuncs.details();
}

double AbstractEvaluator::tetVolume(const Mesh& mesh, const MeshTet& tet) const
{
    const glm::dvec3 vp[] = {
        mesh.verts[tet.v[0]],
        mesh.verts[tet.v[1]],
        mesh.verts[tet.v[2]],
        mesh.verts[tet.v[3]],
    };

    return tetVolume(vp);
}

double AbstractEvaluator::tetVolume(const glm::dvec3 vp[]) const
{
    return glm::determinant(glm::dmat3(
        vp[0] - vp[3],
        vp[1] - vp[3],
        vp[2] - vp[3]));
}

double AbstractEvaluator::priVolume(const Mesh& mesh, const MeshPri& pri) const
{
    const glm::dvec3 vp[] = {
        mesh.verts[pri.v[0]],
        mesh.verts[pri.v[1]],
        mesh.verts[pri.v[2]],
        mesh.verts[pri.v[3]],
        mesh.verts[pri.v[4]],
        mesh.verts[pri.v[5]]
    };

    return priVolume(vp);
}

double AbstractEvaluator::priVolume(const glm::dvec3 vp[]) const
{
    double volume = 0.0;
    volume += glm::determinant(glm::dmat3(
        vp[4] - vp[2],
        vp[0] - vp[2],
        vp[1] - vp[2]));
    volume += glm::determinant(glm::dmat3(
        vp[5] - vp[2],
        vp[1] - vp[2],
        vp[3] - vp[2]));
    volume += glm::determinant(glm::dmat3(
        vp[4] - vp[2],
        vp[1] - vp[2],
        vp[5] - vp[2]));
    return volume;
}

double AbstractEvaluator::hexVolume(const Mesh& mesh, const MeshHex& hex) const
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

    return hexVolume(vp);
}

double AbstractEvaluator::hexVolume(const glm::dvec3 vp[]) const
{
    double volume = 0.0;
    volume += glm::determinant(glm::dmat3(
        vp[0] - vp[2],
        vp[1] - vp[2],
        vp[4] - vp[2]));
    volume += glm::determinant(glm::dmat3(
        vp[3] - vp[1],
        vp[2] - vp[1],
        vp[7] - vp[1]));
    volume += glm::determinant(glm::dmat3(
        vp[5] - vp[4],
        vp[1] - vp[4],
        vp[7] - vp[4]));
    volume += glm::determinant(glm::dmat3(
        vp[6] - vp[7],
        vp[2] - vp[7],
        vp[4] - vp[7]));
    volume += glm::determinant(glm::dmat3(
        vp[1] - vp[2],
        vp[7] - vp[2],
        vp[4] - vp[2]));
    return volume;
}


double AbstractEvaluator::tetQuality(const Mesh& mesh, const MeshTet& tet) const
{
    const glm::dvec3 vp[] = {
        mesh.verts[tet.v[0]],
        mesh.verts[tet.v[1]],
        mesh.verts[tet.v[2]],
        mesh.verts[tet.v[3]],
    };

    return tetQuality(vp);
}

double AbstractEvaluator::priQuality(const Mesh& mesh, const MeshPri& pri) const
{
    const glm::dvec3 vp[] = {
        mesh.verts[pri.v[0]],
        mesh.verts[pri.v[1]],
        mesh.verts[pri.v[2]],
        mesh.verts[pri.v[3]],
        mesh.verts[pri.v[4]],
        mesh.verts[pri.v[5]]
    };

    return priQuality(vp);
}

double AbstractEvaluator::hexQuality(const Mesh& mesh, const MeshHex& hex) const
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

    return hexQuality(vp);
}

bool AbstractEvaluator::assessMeasureValidy()
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
    double regularTet = tetQuality(mesh, tet);
    double regularPri = priQuality(mesh, pri);
    double regularHex = hexQuality(mesh, hex);

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
        double& minQuality,
        double& qualityMean,
        const std::string& implementationName)
{
    ImplementationFunc implementationFunc;
    if(_implementationFuncs.select(implementationName, implementationFunc))
    {
        implementationFunc(mesh, minQuality, qualityMean);
    }
    else
    {
        minQuality = nan("");
        qualityMean = nan("");
    }
}

void AbstractEvaluator::evaluateMeshQualitySerial(
        const Mesh& mesh,
        double& minQuality,
        double& qualityMean)
{
    int tetCount = mesh.tets.size();
    int priCount = mesh.pris.size();
    int hexCount = mesh.hexs.size();
    int elemCount = tetCount + priCount + hexCount;

    std::vector<double> qualities(elemCount);
    int idx = 0;

    for(int i=0; i < tetCount; ++i, ++idx)
        qualities[idx] = tetQuality(mesh, mesh.tets[i]);

    for(int i=0; i < priCount; ++i, ++idx)
        qualities[idx] = priQuality(mesh, mesh.pris[i]);

    for(int i=0; i < hexCount; ++i, ++idx)
        qualities[idx] = hexQuality(mesh, mesh.hexs[i]);


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
            double& minQuality,
            double& qualityMean)
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
                qualities[idx] = tetQuality(mesh, mesh.tets[i]);

            for(int i=priBeg; i < priEnd; ++i, ++idx)
                qualities[idx] = priQuality(mesh, mesh.pris[i]);

            for(int i=hexBeg; i < hexEnd; ++i, ++idx)
                qualities[idx] = hexQuality(mesh, mesh.hexs[i]);

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
        double& minQuality,
        double& qualityMean)
{
    initializeProgram(mesh);


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

    mesh.bindShaderStorageBuffers();
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,
                     mesh.firstFreeBufferBinding(), _qualSsbo);


    // Simulataneous and specialized series gives about the same performance
    // Specialized series gives a tiny, not stable speed boost.
    // (tested on a parametric pri/hex mesh)
    if(_computeSimultaneously)
    {
        _simultaneousProgram.pushProgram();
        glDispatchCompute(workgroupCount, 1, 1);
        _simultaneousProgram.popProgram();
    }
    else
    {
        if(tetCount > 0)
        {
            _tetProgram.pushProgram();
            glDispatchCompute(ceil(tetCount / (double)WORKGROUP_SIZE), 1, 1);
            _tetProgram.popProgram();
        }

        if(priCount > 0)
        {
            _priProgram.pushProgram();
            glDispatchCompute(ceil(priCount / (double)WORKGROUP_SIZE), 1, 1);
            _priProgram.popProgram();
        }

        if(hexCount > 0)
        {
            _hexProgram.pushProgram();
            glDispatchCompute(ceil(hexCount / (double)WORKGROUP_SIZE), 1, 1);
            _hexProgram.popProgram();
        }
    }

    // Fetch workgroup's statistics from GPU
    // We are using glMapBuffer since glGetBufferData seems to update the output
    // concurently while were are computing mesh quality mean. glMemoryBarrier
    // looks like having no effect on this (tried with GL_ALL_BARRIER_BITS
    // before and after the call to glGetBufferSubData).
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _qualSsbo);
    GLint* data = (GLint*) glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);

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

void AbstractEvaluator::initializeProgram(const Mesh& mesh)
{
    if(_initialized)
        return;


    getLog().postMessage(new Message('I', false,
        "Initializing evaluator compute shader", "AbstractEvaluator"));

    std::vector<std::string> qualityInterface = {
        mesh.meshGeometryShaderName(),
        ":/shaders/compute/Quality/QualityInterface.glsl"
    };

    std::vector<std::string> shapeMeasure = {
        mesh.meshGeometryShaderName(),
        _shapeMeasuresShader
    };


    // Simultenous evaluation shader
    _simultaneousProgram.addShader(GL_COMPUTE_SHADER, qualityInterface);
    _simultaneousProgram.addShader(GL_COMPUTE_SHADER, shapeMeasure);
    _simultaneousProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/shaders/compute/Measuring/SimultaneousEvaluation.glsl"});
    _simultaneousProgram.link();
    _simultaneousProgram.pushProgram();
    _simultaneousProgram.popProgram();
    mesh.uploadGeometry(_simultaneousProgram);


    // Specialized evaluation shader series
    _tetProgram.addShader(GL_COMPUTE_SHADER, qualityInterface);
    _tetProgram.addShader(GL_COMPUTE_SHADER, shapeMeasure);
    _tetProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/shaders/compute/Measuring/TetrahedraEvaluation.glsl"});
    _tetProgram.link();
    _tetProgram.pushProgram();
    _tetProgram.popProgram();
    mesh.uploadGeometry(_tetProgram);

    _priProgram.addShader(GL_COMPUTE_SHADER, qualityInterface);
    _priProgram.addShader(GL_COMPUTE_SHADER, shapeMeasure);
    _priProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/shaders/compute/Measuring/PrismsEvaluation.glsl"});
    _priProgram.link();
    _priProgram.pushProgram();
    _priProgram.popProgram();
    mesh.uploadGeometry(_priProgram);

    _hexProgram.addShader(GL_COMPUTE_SHADER, qualityInterface);
    _hexProgram.addShader(GL_COMPUTE_SHADER, shapeMeasure);
    _hexProgram.addShader(GL_COMPUTE_SHADER, {
        mesh.meshGeometryShaderName(),
        ":/shaders/compute/Measuring/HexahedraEvaluation.glsl"});
    _hexProgram.link();
    _hexProgram.pushProgram();
    _hexProgram.popProgram();
    mesh.uploadGeometry(_hexProgram);


    // Shader storage quality blocks
    glDeleteBuffers(1, &_qualSsbo);
    _qualSsbo = 0;
    glGenBuffers(1, &_qualSsbo);


    _initialized = true;
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
        const map<string, int>& cycleCounts)
{
    initializeProgram(mesh);

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
                implementationFunc(mesh, minQual, qualMean);
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

string AbstractEvaluator::shapeMeasureShader() const
{
    return _shapeMeasuresShader;
}
