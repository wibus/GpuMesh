
float tetQuality(Tet tet)
{
    vec3 A = vec3(verts[tet.v[0]].p);
    vec3 B = vec3(verts[tet.v[1]].p);
    vec3 C = vec3(verts[tet.v[2]].p);
    vec3 D = vec3(verts[tet.v[3]].p);

    float u = distance(A, B);
    float v = distance(A, C);
    float w = distance(A, D);
    float U = distance(C, D);
    float V = distance(D, B);
    float W = distance(B, C);

    float Volume = 4.0*u*u*v*v*w*w;
    Volume -= u*u*pow(v*v+w*w-U*U, 2.0);
    Volume -= v*v*pow(w*w+u*u-V*V, 2.0);
    Volume -= w*w*pow(u*u+v*v-W*W, 2.0);
    Volume += (v*v+w*w-U*U)*(w*w+u*u-V*V)*(u*u+v*v-W*W);
    Volume = sqrt(max(Volume, 0.0));
    Volume /= 12.0;

    float s1 = (U + V + W) / 2.0;
    float s2 = (u + v + W) / 2.0;
    float s3 = (u + V + w) / 2.0;
    float s4 = (U + v + w) / 2.0;

    float L1 = sqrt(s1*(s1-U)*(s1-V)*(s1-W));
    float L2 = sqrt(s2*(s2-u)*(s2-v)*(s2-W));
    float L3 = sqrt(s3*(s3-u)*(s3-V)*(s3-w));
    float L4 = sqrt(s4*(s4-U)*(s4-v)*(s4-w));

    float R = (Volume*3)/(L1+L2+L3+L4);

    float maxLen = max(max(max(u, v), w),
                       max(max(U, V), W));

    return (4.89897948557) * R / maxLen;
}

float priQuality(Pri pri)
{
    // Prism quality ~= mean of 6 possible tetrahedrons from prism triangular faces
    Tet tetA = Tet(uint[] (pri.v[4], pri.v[1], pri.v[5], pri.v[3]));
    Tet tetB = Tet(uint[] (pri.v[5], pri.v[2], pri.v[4], pri.v[0]));
    Tet tetC = Tet(uint[] (pri.v[2], pri.v[1], pri.v[5], pri.v[3]));
    Tet tetD = Tet(uint[] (pri.v[3], pri.v[2], pri.v[4], pri.v[0]));
    Tet tetE = Tet(uint[] (pri.v[0], pri.v[1], pri.v[5], pri.v[3]));
    Tet tetF = Tet(uint[] (pri.v[1], pri.v[2], pri.v[4], pri.v[0]));

    float tetAq = tetQuality(tetA);
    float tetBq = tetQuality(tetB);
    float tetCq = tetQuality(tetC);
    float tetDq = tetQuality(tetD);
    float tetEq = tetQuality(tetE);
    float tetFq = tetQuality(tetF);
    return (tetAq + tetBq + tetCq + tetDq + tetEq + tetFq)
            / 3.9067138981002011988; // C/6
}

float hexQuality(Hex hex)
{
    Tet tetA = Tet(uint[] (hex.v[0], hex.v[3], hex.v[5], hex.v[6]));
    Tet tetB = Tet(uint[] (hex.v[1], hex.v[2], hex.v[7], hex.v[4]));

    float tetAQuality = tetQuality(tetA);
    float tetBQuality = tetQuality(tetB);
    return (tetAQuality + tetBQuality) * 0.5; // 1/2
}
