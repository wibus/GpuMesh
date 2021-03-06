#ifndef GPUMESH_SCHEDULE
#define GPUMESH_SCHEDULE

struct Schedule
{
public:
    Schedule();

    bool autoPilotEnabled;
    double minQualThreshold;
    double qualMeanThreshold;

    bool topoOperationEnabled;
    int topoOperationPassCount;
    int refinementSweepCount;

    int globalPassCount;
    int relocationPassCount;
};

#endif // GPUMESH_SCHEDULE
