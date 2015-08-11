layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


uniform float MoveCoeff;


// Boundaries
vec3 snapToBoundary(int boundaryID, vec3 pos);

// Optimization helper functions
bool isSmoothable(uint vId);
vec3 computePatchCenter(in uint vId);


void main()
{
    //* Workgroup clusters dispatching scheme
    uint vId = gl_GlobalInvocationID.x;
    /*/// Scattered workgroup dispatching sceme
    uint vId = gl_LocalInvocationID.x * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    //*/

    if(!isSmoothable(vId))
        return;


    vec3 patchCenter = computePatchCenter(vId);

    vec3 pos = vec3(verts[vId].p);
    pos = mix(pos, patchCenter, MoveCoeff);


    Topo topo = topos[vId];
    if(topo.type > 0)
    {
        pos = snapToBoundary(topo.type, pos);
    }


    // Write
    verts[vId].p = vec4(pos, 0.0);
}
