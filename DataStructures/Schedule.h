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

    int globalPassCount;
    int nodeRelocationsPassCount;
};

#endif // GPUMESH_SCHEDULE
