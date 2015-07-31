
float tetQuality(in vec3 vp[4])
{
    float u = distance(vp[0], vp[1]);
    float v = distance(vp[0], vp[2]);
    float w = distance(vp[0], vp[3]);
    float U = distance(vp[2], vp[3]);
    float V = distance(vp[3], vp[1]);
    float W = distance(vp[1], vp[2]);

    float Volume = 4.0*u*u*v*v*w*w;
    Volume -= u*u*pow(v*v+w*w-U*U, 2.0);
    Volume -= v*v*pow(w*w+u*u-V*V, 2.0);
    Volume -= w*w*pow(u*u+v*v-W*W, 2.0);
    Volume += (v*v+w*w-U*U)*(w*w+u*u-V*V)*(u*u+v*v-W*W);
    Volume = sign(Volume) * sqrt(abs(Volume));
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

float priQuality(in vec3 vp[6])
{
    vec3 tetA[] = vec3[](vp[4], vp[1], vp[5], vp[3]);
    vec3 tetB[] = vec3[](vp[5], vp[2], vp[4], vp[0]);
    vec3 tetC[] = vec3[](vp[2], vp[1], vp[5], vp[3]);
    vec3 tetD[] = vec3[](vp[3], vp[2], vp[4], vp[0]);
    vec3 tetE[] = vec3[](vp[0], vp[1], vp[5], vp[3]);
    vec3 tetF[] = vec3[](vp[1], vp[2], vp[4], vp[0]);

    float tetAq = tetQuality(tetA);
    float tetBq = tetQuality(tetB);
    float tetCq = tetQuality(tetC);
    float tetDq = tetQuality(tetD);
    float tetEq = tetQuality(tetE);
    float tetFq = tetQuality(tetF);
    return (tetAq + tetBq + tetCq + tetDq + tetEq + tetFq)
            / 4.2970697433826288147;
}

float hexQuality(in vec3 vp[8])
{
    vec3 tetA[] = vec3[](vp[0], vp[3], vp[5], vp[6]);
    vec3 tetB[] = vec3[](vp[1], vp[2], vp[7], vp[4]);

    float tetAQuality = tetQuality(tetA);
    float tetBQuality = tetQuality(tetB);
    return (tetAQuality + tetBQuality)
            / 2.0;
}
