#include "AbstractSmoother.h"

#include <iostream>

using namespace std;


AbstractSmoother::AbstractSmoother(
        Mesh &mesh,
        double moveFactor,
        double gainThreshold) :
    _mesh(mesh),
    _moveFactor(moveFactor),
    _gainThreshold(gainThreshold)
{
}

AbstractSmoother::~AbstractSmoother()
{

}

void AbstractSmoother::evaluateInitialMeshQuality()
{
    _smoothPassId = 0;
}

bool AbstractSmoother::evaluateIterationMeshQuality()
{
    double newQualityMean, newQualityVar, newMinQuality;
    _mesh.compileElementQuality(
                newQualityMean,
                newQualityVar,
                newMinQuality);

    cout << "Smooth pass number " << _smoothPassId << endl;
    cout << "Mesh quality mean: " << newQualityMean << endl;
    cout << "Mesh quality std dev: " << newQualityVar << endl;
    cout << "Mesh minimum quality: " << newMinQuality << endl;

    bool continueSmoothing = true;
    if(_smoothPassId > 0)
    {
        continueSmoothing = (newQualityMean - _lastQualityMean) > _gainThreshold;
    }

    _lastQualityMean = newQualityMean;
    _lastQualityVar = newQualityVar;
    _lastMinQuality = newMinQuality;
    ++_smoothPassId;

    return continueSmoothing;
}
