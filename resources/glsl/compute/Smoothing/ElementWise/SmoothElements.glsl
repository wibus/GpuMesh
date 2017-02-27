// Algorithm entry points
void smoothTet(uint eId);
void smoothPri(uint eId);
void smoothHex(uint eId);


// Smoothing helper
uint getInvocationTetId();
uint getInvocationPriId();
uint getInvocationHexId();
bool isSmoothableTet(uint eId);
bool isSmoothablePri(uint eId);
bool isSmoothableHex(uint eId);


void main()
{
    uint tetId = getInvocationTetId();
    if(isSmoothableTet(tetId))
    {
        smoothTet(tetId);
    }

    uint priId = getInvocationPriId();
    if(isSmoothablePri(priId))
    {
        smoothPri(priId);
    }

    uint hexId = getInvocationHexId();
    if(isSmoothableHex(hexId))
    {
        smoothHex(hexId);
    }
}
