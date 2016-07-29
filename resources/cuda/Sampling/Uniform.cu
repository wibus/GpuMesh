#include "Base.cuh"

#include <vector>


texture<float4, 3> TopLineTex;
texture<float4, 3> SideTriTex;
__constant__ mat4* TexTransform;


__device__ mat3 uniformMetricAt(const vec3& position, uint& cachedRefTet)
{
    vec3 coor = vec3(*TexTransform * vec4(position, 1.0));

    float4 topLine = tex3D(TopLineTex, coor.x, coor.y, coor.z);
    float4 sideTri = tex3D(SideTriTex, coor.x, coor.y, coor.z);

    mat3 metric = mat3(topLine.x, topLine.y, topLine.z,
                       topLine.y, sideTri.x, sideTri.y,
                       topLine.z, sideTri.y, sideTri.z);

    return metric;
}


__device__ metricAtFct uniformMetricAtPtr = uniformMetricAt;


// CUDA Drivers

void installCudaUniformSampler()
{
    metricAtFct d_metricAt = nullptr;
    cudaMemcpyFromSymbol(&d_metricAt, uniformMetricAtPtr, sizeof(metricAtFct));
    cudaMemcpyToSymbol(metricAt, &d_metricAt, sizeof(metricAtFct));

    // Setup texture reference parameters
    TopLineTex.normalized = true;
    TopLineTex.filterMode = cudaFilterModeLinear;
    TopLineTex.addressMode[0] = cudaAddressModeClamp;
    TopLineTex.addressMode[1] = cudaAddressModeClamp;
    TopLineTex.addressMode[2] = cudaAddressModeClamp;

    SideTriTex.normalized = true;
    SideTriTex.filterMode = cudaFilterModeLinear;
    SideTriTex.addressMode[0] = cudaAddressModeClamp;
    SideTriTex.addressMode[1] = cudaAddressModeClamp;
    SideTriTex.addressMode[2] = cudaAddressModeClamp;

    if(verboseCuda)
        printf("I -> CUDA \tUniform Discritizer installed\n");
}


mat4* d_texTransform = nullptr;

glm::ivec3 d_topLineExtents(0, 0, 0);
cudaArray* d_topLineArray = nullptr;

glm::ivec3 d_sideTriExtents(0, 0, 0);
cudaArray* d_sideTriArray = nullptr;

void updateCudaUniformTextures(
        const std::vector<glm::vec4>& topLineBuff,
        const std::vector<glm::vec4>& sideTriBuff,
        const glm::mat4& texTransform,
        const glm::ivec3 texDims)
{
    bool clearTex = (texDims.x == 0) || (texDims.y == 0) || (texDims.z == 0);


    // Send transform matrix
    if(d_texTransform == nullptr)
    {
        cudaMalloc(&d_texTransform, sizeof(mat4));
        cudaMemcpyToSymbol(TexTransform, &d_texTransform, sizeof(d_texTransform));
    }

    cudaMemcpy(d_texTransform, &texTransform, sizeof(texTransform), cudaMemcpyHostToDevice);

    cudaExtent extents = make_cudaExtent(texDims.x, texDims.y, texDims.z);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(
        32, 32, 32, 32, cudaChannelFormatKindFloat);

    cudaMemcpy3DParms copyParams = {0};
    copyParams.extent   = extents;
    copyParams.kind     = cudaMemcpyHostToDevice;


    if(d_topLineExtents != texDims)
    {
        cudaUnbindTexture(TopLineTex);
        cudaFreeArray(d_topLineArray);
        d_topLineArray = nullptr;

        cudaCheckErrors("Deallocating TopLine array");

        if(!clearTex)
        {
            cudaMalloc3DArray(&d_topLineArray, &channelDesc, extents);
            cudaCheckErrors("Allocating TopLineTex data");

            copyParams.dstArray = d_topLineArray;
            copyParams.srcPtr   = make_cudaPitchedPtr(
                (void *)topLineBuff.data(),
                extents.width*sizeof(vec4),
                extents.width, extents.height);

            cudaMemcpy3D(&copyParams);
            cudaCheckErrors("Copying TopLineTex data");

            cudaBindTextureToArray(TopLineTex, d_topLineArray, channelDesc);
            cudaCheckErrors("Binding TopLineTex to its array");
        }

        d_topLineExtents = texDims;
    }

    if(d_sideTriExtents != texDims)
    {
        cudaUnbindTexture(SideTriTex);
        cudaFreeArray(d_sideTriArray);
        d_sideTriArray = nullptr;

        cudaCheckErrors("Deallocating SideTri array");

        if(!clearTex)
        {
            cudaMalloc3DArray(&d_sideTriArray, &channelDesc, extents);
            cudaCheckErrors("Allocating SideTriTex data");

            copyParams.dstArray = d_sideTriArray;
            copyParams.srcPtr   = make_cudaPitchedPtr(
                (void *)sideTriBuff.data(),
                extents.width*sizeof(vec4),
                extents.width, extents.height);

            cudaMemcpy3D(&copyParams);
            cudaCheckErrors("Copying SideTriTex data");

            cudaBindTextureToArray(SideTriTex, d_sideTriArray, channelDesc);
            cudaCheckErrors("Binding SideTriTex to its array");
        }

        d_sideTriExtents = texDims;
    }
}
