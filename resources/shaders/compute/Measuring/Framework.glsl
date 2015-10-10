// Subroutine definitions
void accumulatePatchQuality(
        inout float patchQuality,
        inout float patchWeight,
        in float elemQuality)
{
    patchQuality = min(
        min(patchQuality, elemQuality),  // If sign(patch) != sign(elem)
        min(patchQuality * elemQuality,  // If sign(patch) & sign(elem) > 0
            patchQuality + elemQuality));// If sign(patch) & sign(elem) < 0
}

float finalizePatchQuality(in float patchQuality, in float patchWeight)
{
    return patchQuality;
}
