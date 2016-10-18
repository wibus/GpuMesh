vec3 riemannianSegment(in vec3 a, in vec3 b, inout uint cachedRefTet);


const mat3 Fr_TET_INV = mat3(
    vec3(1, 0, 0),
    vec3(-0.5773502691896257645091, 1.154700538379251529018, 0),
    vec3(-0.4082482904638630163662, -0.4082482904638630163662, 1.224744871391589049099)
);

const mat3 Fr_PRI_INV = mat3(
    vec3(1.0, 0.0, 0.0),
    vec3(-0.5773502691896257645091, 1.154700538379251529018, 0.0),
    vec3(0.0, 0.0, 1.0)
);

float cornerQuality(in mat3 Fk)
{
    float Fk_det = determinant(Fk);

    float Fk_frobenius2 =
        dot(Fk[0], Fk[0]) +
        dot(Fk[1], Fk[1]) +
        dot(Fk[2], Fk[2]);

    return sign(Fk_det) * 3.0 * pow(abs(Fk_det), 2.0/3.0) / Fk_frobenius2;
}

float tetQuality(in vec3 vp[TET_VERTEX_COUNT], inout Tet tet)
{
    vec3 e03 = riemannianSegment(vp[0], vp[3], tet.c[0]);
    vec3 e13 = riemannianSegment(vp[1], vp[3], tet.c[0]);
    vec3 e23 = riemannianSegment(vp[2], vp[3], tet.c[0]);

    mat3 Fk0 = mat3(e03, e13, e23) * Fr_TET_INV;

    float qual0 = cornerQuality(Fk0);

    // Shape measure is independent of chosen corner
    return qual0;
}

float priQuality(in vec3 vp[PRI_VERTEX_COUNT], inout Pri pri)
{
    vec3 e03 = riemannianSegment(vp[0], vp[3], pri.c[0]);
    vec3 e14 = riemannianSegment(vp[1], vp[4], pri.c[1]);
    vec3 e25 = riemannianSegment(vp[2], vp[5], pri.c[2]);
    vec3 e01 = riemannianSegment(vp[0], vp[1], pri.c[0]);
    vec3 e12 = riemannianSegment(vp[1], vp[2], pri.c[1]);
    vec3 e20 = riemannianSegment(vp[2], vp[0], pri.c[2]);
    vec3 e34 = riemannianSegment(vp[3], vp[4], pri.c[3]);
    vec3 e45 = riemannianSegment(vp[4], vp[5], pri.c[4]);
    vec3 e53 = riemannianSegment(vp[5], vp[3], pri.c[5]);


    // Prism corner quality is not invariant under edge swap
    // Third edge is the expected to be colinear with the first two cross product
    mat3 Fk0 = mat3(-e01, e20, e03) * Fr_PRI_INV;
    mat3 Fk1 = mat3(-e12, e01, e14) * Fr_PRI_INV;
    mat3 Fk2 = mat3(-e20, e12, e25) * Fr_PRI_INV;
    mat3 Fk3 = mat3(-e34, e53, e03) * Fr_PRI_INV;
    mat3 Fk4 = mat3(-e45, e34, e14) * Fr_PRI_INV;
    mat3 Fk5 = mat3(-e53, e45, e25) * Fr_PRI_INV;

    float qual0 = cornerQuality(Fk0);
    float qual1 = cornerQuality(Fk1);
    float qual2 = cornerQuality(Fk2);
    float qual3 = cornerQuality(Fk3);
    float qual4 = cornerQuality(Fk4);
    float qual5 = cornerQuality(Fk5);


    if(qual0 <= 0.0 || qual1 <= 0.0 || qual2 <= 0.0 ||
       qual3 <= 0.0 || qual4 <= 0.0 || qual5 <= 0.0)
    {
        float qualMin = qual0;
        if(qual1 < qualMin) qualMin = qual1;
        if(qual2 < qualMin) qualMin = qual2;
        if(qual3 < qualMin) qualMin = qual3;
        if(qual4 < qualMin) qualMin = qual4;
        if(qual5 < qualMin) qualMin = qual5;
        return qualMin;
    }
    else
    {
        double mean = 6.0 / (
            1/qual0 + 1/qual1 + 1/qual2 +
            1/qual3 + 1/qual4 + 1/qual5);

        return mean;
    }
}

float hexQuality(in vec3 vp[HEX_VERTEX_COUNT], inout Hex hex)
{
    // Since hex's corner matrix is the identity matrix,
    // there's no need to define Fr_HEX_INV.
    vec3 e01 = riemannianSegment(vp[0], vp[1], hex.c[0]);
    vec3 e03 = riemannianSegment(vp[0], vp[3], hex.c[0]);
    vec3 e04 = riemannianSegment(vp[0], vp[4], hex.c[0]);
    vec3 e12 = riemannianSegment(vp[1], vp[2], hex.c[1]);
    vec3 e15 = riemannianSegment(vp[1], vp[5], hex.c[1]);
    vec3 e23 = riemannianSegment(vp[2], vp[3], hex.c[3]);
    vec3 e26 = riemannianSegment(vp[2], vp[6], hex.c[2]);
    vec3 e37 = riemannianSegment(vp[3], vp[7], hex.c[3]);
    vec3 e45 = riemannianSegment(vp[4], vp[5], hex.c[4]);
    vec3 e47 = riemannianSegment(vp[4], vp[7], hex.c[4]);
    vec3 e56 = riemannianSegment(vp[5], vp[6], hex.c[5]);
    vec3 e67 = riemannianSegment(vp[6], vp[7], hex.c[6]);

    mat3 Fk0 = mat3(e01,  e04, -e03);
    mat3 Fk1 = mat3(e01,  e12,  e15);
    mat3 Fk2 = mat3(e12,  e26, -e23);
    mat3 Fk3 = mat3(e03,  e23,  e37);
    mat3 Fk4 = mat3(e04,  e45,  e47);
    mat3 Fk5 = mat3(e15, -e56,  e45);
    mat3 Fk6 = mat3(e26,  e56,  e67);
    mat3 Fk7 = mat3(e37,  e67, -e47);

    float qual0 = cornerQuality(Fk0);
    float qual1 = cornerQuality(Fk1);
    float qual2 = cornerQuality(Fk2);
    float qual3 = cornerQuality(Fk3);
    float qual4 = cornerQuality(Fk4);
    float qual5 = cornerQuality(Fk5);
    float qual6 = cornerQuality(Fk6);
    float qual7 = cornerQuality(Fk7);


    if(qual0 <= 0.0 || qual1 <= 0.0 || qual2 <= 0.0 || qual3 <= 0.0 ||
       qual4 <= 0.0 || qual5 <= 0.0 || qual6 <= 0.0 || qual7 <= 0.0)
    {
        float minQual = qual0;
        if(qual1 < minQual) minQual = qual1;
        if(qual2 < minQual) minQual = qual2;
        if(qual3 < minQual) minQual = qual3;
        if(qual4 < minQual) minQual = qual4;
        if(qual5 < minQual) minQual = qual5;
        if(qual6 < minQual) minQual = qual6;
        if(qual7 < minQual) minQual = qual7;
        return minQual;
    }
    else
    {
        double mean = 8.0 / (
            1/qual0 + 1/qual1 + 1/qual2 + 1/qual3 +
            1/qual4 + 1/qual5 + 1/qual6 + 1/qual7);

        return mean;
    }
}
