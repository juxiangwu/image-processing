/* Please Write the OpenCL Kernel(s) code here*/
__kernel void brazil_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 src_rgba = read_imagef(input,sampler,coord);
   
   float xBlockSize = 0.01*0.1;
   float yBlockSize = xBlockSize * size.x / size.y;  // mutiply ratio
   float xCoord = (floor((coord.x-0.5)/xBlockSize)+0.5) * xBlockSize+0.5;
   float yCoord = (floor((coord.y-0.5)/yBlockSize)+0.5) * yBlockSize+0.5;
   float arg = 0.5f;
   
   float4 color = read_imagef(input,sampler,convert_int2((float2)(xCoord,yCoord)));
   color = (float4)(color.xyz+arg * 2.0f - 1.0f, color.w);
   
    float sum = (color.x + color.y + color.z) / 3.0f;

    float3 white  = (float3)(255.0f, 255.0f, 255.0f) / 255.0f;
    float3 yellow = (float3)(242.0f, 252.0f,   0.0f) / 255.0f;
    float3 green  = (float3)(  0.0f, 140.0f,   0.0f) / 255.0f;
    float3 brown  = (float3)( 48.0f,  19.0f,   6.0f) / 255.0f;
    float3 black  = (float3)(  0.0f,   0.0f,   0.0f) / 255.0f;

    if      (sum < 0.110f) color = (float4)(black,  color.w);
    else if (sum < 0.310f) color = (float4)(brown,  color.w);
    else if (sum < 0.513f) color = (float4)(green,  color.w);
    else if (sum < 0.965f) color = (float4)(yellow, color.w);
    else                  color = (float4)(white,  color.w);
   
  write_imagef(output,coord,color);
}