#include "Schedule.h"


Schedule::Schedule() :
    autoPilotEnabled(false),
    minQualThreshold(0.001),
    qualMeanThreshold(0.000),
    topoOperationEnabled(true),
    topoOperationPassCount(5),
    globalPassCount(5),
    nodeRelocationsPassCount(10)
{

}
