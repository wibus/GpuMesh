uniform float GainThreshold;
uniform int SecurityCycleCount;
uniform float LocalSizeToNodeShift;
uniform float Alpha;
uniform float Beta;
uniform float Gamma;
uniform float Delta;

// Boundaries
vec3 snapToBoundary(int boundaryID, vec3 pos);

// Smoothing Helper
float computeLocalElementSize(in uint vId);
float patchQuality(in uint vId);
void swap(inout vec4 v1, inout vec4 v2)
{
    vec4 tmp = v1;
    v1 = v2;
    v2 = tmp;
}

// ENTRY POINT //
void smoothVert(uint vId)
{
    // Compute local element size
    float localSize = computeLocalElementSize(vId);

    // Initialize node shift distance
    float nodeShift = localSize * LocalSizeToNodeShift;


    Topo topo = topos[vId];
    vec3 pos = verts[vId].p;
    vec4 vo = vec4(pos, patchQuality(vId));

    vec4 simplex[TET_VERTEX_COUNT] = {
        vec4(pos + vec3(nodeShift, 0, 0), 0),
        vec4(pos + vec3(0, nodeShift, 0), 0),
        vec4(pos + vec3(0, 0, nodeShift), 0),
        vo
    };

    int cycle = 0;
    bool reset = false;
    bool terminated = false;
    while(!terminated)
    {
        for(uint p=0; p < TET_VERTEX_COUNT-1; ++p)
        {
            // Since 'pos' is a reference on vertex's position
            // modifing its value here should be seen by the evaluator
            if(topo.type > 0)
                verts[vId].p = snapToBoundary(topo.type, vec3(simplex[p]));
            else
                verts[vId].p = vec3(simplex[p]);

            // Compute patch quality
            simplex[p] = vec4(verts[vId].p, patchQuality(vId));
        }

        // Mini bubble sort
        if(simplex[0].w > simplex[1].w)
            swap(simplex[0], simplex[1]);
        if(simplex[1].w > simplex[2].w)
            swap(simplex[1], simplex[2]);
        if(simplex[2].w > simplex[3].w)
            swap(simplex[2], simplex[3]);
        if(simplex[0].w > simplex[1].w)
            swap(simplex[0], simplex[1]);
        if(simplex[1].w > simplex[2].w)
            swap(simplex[1], simplex[2]);
        if(simplex[0].w > simplex[1].w)
            swap(simplex[0], simplex[1]);


        for(; cycle < SecurityCycleCount; ++cycle)
        {
            // Centroid
            vec3 c = 1/3.0f * (
                vec3(simplex[1]) +
                vec3(simplex[2]) +
                vec3(simplex[3]));

            float f = 0.0;

            // Reflect
            verts[vId].p = c + Alpha*(c - vec3(simplex[0]));
            if(topo.type > 0) verts[vId].p = snapToBoundary(topo.type, verts[vId].p);
            float fr = f = patchQuality(vId);

            vec3 xr = verts[vId].p;

            // Expand
            if(simplex[3].w < fr)
            {
                verts[vId].p = c + Gamma*(verts[vId].p - c);
                if(topo.type > 0) verts[vId].p = snapToBoundary(topo.type, verts[vId].p);
                float fe = f = patchQuality(vId);

                if(fe <= fr)
                {
                    verts[vId].p = xr;
                    f = fr;
                }
            }
            // Contract
            else if(simplex[1].w >= fr)
            {
                // Outside
                if(fr > simplex[0].w)
                {
                    verts[vId].p = c + Beta*(vec3(xr) - c);
                    if(topo.type > 0) verts[vId].p = snapToBoundary(topo.type, verts[vId].p);
                    f = patchQuality(vId);
                }
                // Inside
                else
                {
                    verts[vId].p = c + Beta*(vec3(simplex[0]) - c);
                    if(topo.type > 0) verts[vId].p = snapToBoundary(topo.type, verts[vId].p);
                    f = patchQuality(vId);
                }
            }

            // Insert new vertex in the working simplex
            vec4 vertex = vec4(verts[vId].p, f);
            if(vertex.w > simplex[3].w)
                swap(simplex[3], vertex);
            if(vertex.w > simplex[2].w)
                swap(simplex[2], vertex);
            if(vertex.w > simplex[1].w)
                swap(simplex[1], vertex);
            if(vertex.w > simplex[0].w)
                swap(simplex[0], vertex);


            if( (simplex[3].w - simplex[1].w) < GainThreshold )
            {
                terminated = true;
                break;
            }
        }

        if( terminated || (cycle >= SecurityCycleCount && reset) )
        {
            break;
        }
        else
        {
            simplex[0] = vo - vec4(nodeShift, 0, 0, 0);
            simplex[1] = vo - vec4(0, nodeShift, 0, 0);
            simplex[2] = vo - vec4(0, 0, nodeShift, 0);
            simplex[3] = vo;
            reset = true;
            cycle = 0;
        }
    }

    if(topo.type > 0)
        verts[vId].p = snapToBoundary(topo.type, vec3(simplex[3]));
    else
        verts[vId].p = vec3(simplex[3]);
}
