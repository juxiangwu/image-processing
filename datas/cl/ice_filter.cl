/* Please Write the OpenCL Kernel(s) code here*/

float rand(float2 co){
    float iptr = 1.0f;
    return fract(sin(dot(co.xy,(float2)(12.9898f,78.233f))) * 43758.5453f,&iptr);
}

__kernel void ice_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
 
    float2 v1 = (float2)(0.001f,0.001f);
    float2 v2 = (float2)(0.000f,0.000f);
    float iptr = 1.0f;
    float rnd_scale = 1.0f;
    float2 coordf = (float2)(coord.x,coord.y);
    float rnd_factor = 1.5f;
    float rnd = fract(sin(dot(coordf,v1)) + cos(dot(coordf,v2)) * rnd_scale,&iptr);
    int2 offset = (int2)((int)(rnd * rnd_factor * coord.x),(int)(rnd * rnd_factor * coord.y) );
    float3 srcRGB = read_imagef(input,sampler,offset).xyz;
    
    write_imagef(output,coord,(float4)(srcRGB,1.0f));
    
}