/* Please Write the OpenCL Kernel(s) code here*/

__kernel void radial_blur(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    
    int samples[10] = {-8,-5,-3,-2,-1,1,2,3,5,8};
    
    int2 dir = (size / 2) - coord;
    
    float dist = sqrt((float)(dir.x * dir.x) + (float)(dir.y * dir.y));
    
    dir = dir / (int)dist;
    
    float3 sum = (float3)(0,0,0);
    
    float4 srcColor = read_imagef(input,sampler,coord);
    int sampleDist = 2;
    float sampleStrength = 3.2f;
    
    for(int i = 0; i < 10;i++){
        sum += read_imagef(input,sampler,coord + dir * samples[i] * sampleDist).xyz;
    }
    
    sum *= 1.0f / 11.0f;
    
    float t = dist * sampleStrength;
    
    t = clamp(t,0.0f,1.0f);
    
    float3 dstRGB = mix(srcColor.xyz,sum,t);
    
    float4 dstColor = (float4)(dstRGB,1.0f);
    
    write_imagef(output,coord,dstColor);
    
}