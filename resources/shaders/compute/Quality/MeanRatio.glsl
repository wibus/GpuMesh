
float cornerQuality(in mat3 Fk)
{
    float Fk_det = determinant(Fk);
    if(Fk_det < 0.0)
        return 0.0;

    float Fk_frobenius2 =
        dot(Fk[0], Fk[0]) +
        dot(Fk[1], Fk[1]) +
        dot(Fk[2], Fk[2]);

    return 3.0 * pow(Fk_det, 2.0/3.0) / Fk_frobenius2;
}

float tetQuality(in vec3 vp[4])
{
    const mat3 Fr_INV = mat3(
        vec3(1, 0, 0),
        vec3(-0.5773502691896257645091, 1.154700538379251529018, 0),
        vec3(-0.4082482904638630163662, -0.4082482904638630163662, 1.224744871391589049099)
    );

    mat3 Tk0 = mat3(vp[0]-vp[1], vp[0]-vp[2], vp[0]-vp[3]);

    float qual0 = cornerQuality(Tk0 * Fr_INV);

    return qual0;
}

float priQuality(in vec3 vp[6])
{
    const mat3 Fr_INV = mat3(
        vec3(1.0, 0.0, 0.0),
        vec3(-0.5773502691896257645091, 1.154700538379251529018, 0.0),
        vec3(0.0, 0.0, 1.0)
    );

    mat3 Tk0 = mat3(vp[0]-vp[4], vp[0]-vp[2], vp[0]-vp[1]);
    mat3 Tk1 = mat3(vp[1]-vp[3], vp[1]-vp[5], vp[1]-vp[0]);
    mat3 Tk2 = mat3(vp[2]-vp[0], vp[2]-vp[4], vp[2]-vp[3]);
    mat3 Tk3 = mat3(vp[3]-vp[5], vp[3]-vp[1], vp[3]-vp[2]);
    mat3 Tk4 = mat3(vp[4]-vp[2], vp[4]-vp[0], vp[4]-vp[5]);
    mat3 Tk5 = mat3(vp[5]-vp[1], vp[5]-vp[3], vp[5]-vp[4]);

    float qual0 = cornerQuality(Tk0 * Fr_INV);
    float qual1 = cornerQuality(Tk1 * Fr_INV);
    float qual2 = cornerQuality(Tk2 * Fr_INV);
    float qual3 = cornerQuality(Tk3 * Fr_INV);
    float qual4 = cornerQuality(Tk4 * Fr_INV);
    float qual5 = cornerQuality(Tk5 * Fr_INV);

    return (qual0 + qual1 + qual2 + qual3 + qual4 + qual5) / 6.0;
}

float hexQuality(in vec3 vp[8])
{
    mat3 Tk0 = mat3(vp[0]-vp[1], vp[0]-vp[4], vp[0]-vp[2]);
    mat3 Tk1 = mat3(vp[1]-vp[0], vp[1]-vp[3], vp[1]-vp[5]);
    mat3 Tk2 = mat3(vp[2]-vp[0], vp[2]-vp[6], vp[2]-vp[3]);
    mat3 Tk3 = mat3(vp[3]-vp[1], vp[3]-vp[2], vp[3]-vp[7]);
    mat3 Tk4 = mat3(vp[4]-vp[0], vp[4]-vp[5], vp[4]-vp[6]);
    mat3 Tk5 = mat3(vp[5]-vp[1], vp[5]-vp[7], vp[5]-vp[4]);
    mat3 Tk6 = mat3(vp[6]-vp[2], vp[6]-vp[4], vp[6]-vp[7]);
    mat3 Tk7 = mat3(vp[7]-vp[3], vp[7]-vp[6], vp[7]-vp[5]);

    float qual0 = cornerQuality(Tk0);
    float qual1 = cornerQuality(Tk1);
    float qual2 = cornerQuality(Tk2);
    float qual3 = cornerQuality(Tk3);
    float qual4 = cornerQuality(Tk4);
    float qual5 = cornerQuality(Tk5);
    float qual6 = cornerQuality(Tk6);
    float qual7 = cornerQuality(Tk7);

    return (qual0 + qual1 + qual2 + qual3 + qual4 + qual5 + qual6 + qual7) / 8.0;
}
