#include "AbstractSmoother.h"

#include <iostream>

#include "Evaluators/AbstractEvaluator.h"

using namespace std;


AbstractSmoother::AbstractSmoother(
        double moveFactor,
        double gainThreshold) :
    _moveFactor(moveFactor),
    _gainThreshold(gainThreshold)
{
}

AbstractSmoother::~AbstractSmoother()
{

}

void AbstractSmoother::evaluateInitialMeshQuality(Mesh& mesh, AbstractEvaluator& evaluator)
{
    _smoothPassId = 0;
}

bool AbstractSmoother::evaluateIterationMeshQuality(Mesh& mesh, AbstractEvaluator& evaluator)
{
    bool continueSmoothing = true;
    if(_smoothPassId >= 100)
    {
        continueSmoothing = false;

        double newQualityMean, newQualityVar, newMinQuality;
        evaluator.evaluateMeshQuality(
            mesh,
            newQualityMean,
            newQualityVar,
            newMinQuality);

        cout << "Smooth pass number " << _smoothPassId << endl;
        cout << "Mesh quality mean: " << newQualityMean << endl;
        cout << "Mesh quality std dev: " << newQualityVar << endl;
        cout << "Mesh minimum quality: " << newMinQuality << endl;

        /*
        if(_smoothPassId > 0)
        {
            continueSmoothing = (newQualityMean - _lastQualityMean) > _gainThreshold;
        }

        _lastQualityMean = newQualityMean;
        _lastQualityVar = newQualityVar;
        _lastMinQuality = newMinQuality;
        */
    }
    else
    {
        continueSmoothing = true;
    }

    ++_smoothPassId;
    return continueSmoothing;
}
