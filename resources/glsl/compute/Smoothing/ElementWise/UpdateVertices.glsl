// Vertex Accum
bool assignAverage(in uint vId, inout vec3 pos);
void reinitAccum(in uint vId);

// Smoothing Helper
uint getInvocationVertexId();
bool isSmoothableVertex(uint vId);
float patchQuality(in uint vId);


void main()
{
    uint vId = getInvocationVertexId();

    if(isSmoothableVertex(vId))
    {
        vec3 pos = verts[vId].p;
        vec3 posPrim = pos;

        if(assignAverage(vId, posPrim))
        {
            float prePatchQuality =
                patchQuality(vId);

            verts[vId].p = posPrim;

            float patchQualityPrime =
                patchQuality(vId);

            if(patchQualityPrime < prePatchQuality)
                verts[vId].p = pos;
        }

        reinitAccum(vId);
    }
}
