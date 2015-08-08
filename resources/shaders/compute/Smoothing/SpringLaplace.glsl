layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


uniform float MoveCoeff;


vec3 snapToBoundary(int boundaryID, vec3 pos);


void main()
{
    uint uid = gl_GlobalInvocationID.x;

    if(uid >= verts.length())
        return;

    Topo topo = topos[uid];
    if(topo.type == TOPO_FIXED)
        return;

    uint neigElemCount = topo.neigElemCount;
    if(neigElemCount == 0)
        return;


    // Read
    vec3 pos = vec3(verts[uid].p);


    float weightSum = 0.0;
    vec3 patchCenter = vec3(0.0);

    uint n = topo.neigVertBase;
    uint neigVertCount = topo.neigVertCount;
    for(uint i=0; i<neigVertCount; ++i, ++n)
    {
        vec3 npos = vec3(verts[neigVerts[n].v].p);

        vec3 dist = npos - pos;
        float weight = dot(dist, dist) + 0.0001;

        patchCenter += npos * weight;
        weightSum += weight;
    }

    patchCenter /= weightSum;
    pos += MoveCoeff * (patchCenter - pos);

    if(topo.type > 0)
    {
        pos = snapToBoundary(topo.type, pos);
    }


    // Write
    verts[uid].p = vec4(pos, 0.0);
}
