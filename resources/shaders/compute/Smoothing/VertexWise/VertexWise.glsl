layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


// Worgroup invocation disptach mode
uniform int DispatchMode = 0;
const int DISPATCH_MODE_CLUSTER = 0;
const int DISPATCH_MODE_SCATTER = 1;


// Algorithm entry point
void smoothVertex(uint vId);

// Smoothing Helper
bool isSmoothable(uint vId);


void main()
{
    uint vId;

    switch(DispatchMode)
    {
    case DISPATCH_MODE_SCATTER :
        vId = gl_LocalInvocationID.x * gl_NumWorkGroups.x + gl_WorkGroupID.x;
        break;

    case DISPATCH_MODE_CLUSTER :
        // FALLTHROUGH default case
    default :
        vId = gl_GlobalInvocationID.x;
        break;
    }

    if(!isSmoothable(vId))
        return;

    smoothVertex(vId);
}
