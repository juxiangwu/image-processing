/* Please Write the OpenCL Kernel(s) code here*/

__kernel void linearize_depth(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float n = 1.0f; //camera z near
   float f = 100.0f; // camera z far
   
   float4 srcColor = read_imagef(input,sampler,coord);
   
   float z = srcColor.x;
   
   float depth = 1 - (2.0f * n) / (f + n - z * (f - n));
   
   float3 zvec = (float3)(depth,depth,depth);
   
   const float LOG2 = 1.442695f;
   
   float fdistance = 10.0f;
   
   float fogColorStrength = exp2( -fdistance * fdistance * zvec * zvec * LOG2).x;
   
   fogColorStrength = clamp(fogColorStrength,0.0f,1.0f);
   
   float3 fogColor = (float3)(1.0f,1.0f,1.0f);
   
   float3 dstRGB = mix(fogColor,srcColor.xyz, 1 - fogColorStrength);
   
   write_imagef(output,coord,(float4)(dstRGB,1.0f));
   
}