vec3 snapToBoundary(int boundaryID, vec3 pos)
{
    pos[boundaryID-1] /= abs(pos[boundaryID-1]);
    return pos;
}
