layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;


// Boundaries
vec3 snapToBoundary(int boundaryID, vec3 pos);

// Vertex Accum
bool assignAverage(in uint vId, inout vec3 pos);
void reinitAccum(in uint vId);

// Smoothing Helper
uint getInvocationVertexId();
bool isSmoothableVertex(uint vId);
float computePatchQuality(in uint vId);


void main()
{
    uint vId = getInvocationVertexId();

    if(isSmoothableVertex(vId))
    {
        vec3 pos = vec3(verts[vId].p);
        vec3 posPrim = pos;

        if(assignAverage(vId, posPrim))
        {
            Topo topo = topos[vId];
            if(topo.type > 0)
                posPrim = snapToBoundary(topo.type, posPrim);

            float patchQuality =
                computePatchQuality(vId);

            verts[vId].p = vec4(posPrim, 0.0);

            float patchQualityPrime =
                computePatchQuality(vId);

            if(patchQualityPrime < patchQuality)
                verts[vId].p = vec4(pos, 0.0);
        }

        reinitAccum(vId);
    }
}
