/* Please Write the OpenCL Kernel(s) code here*/
__kernel void mosaic_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   float arg = 0.075f;
   float4 color = read_imagef(input,sampler,coord);
   if (arg > 0.0f) {
    float xBlockSize = arg * 0.1f;
    float yBlockSize = xBlockSize * size.x / size.y;  // mutiply ratio
    float xCoord = (floor((coord.x-0.5f)/xBlockSize)+0.5f) * xBlockSize + 0.5f;
    float yCoord = (floor((coord.y-0.5f)/yBlockSize)+0.5f) * yBlockSize + 0.5f;
    float4 rgba = read_imagef(input,sampler,(float2)(xCoord,yCoord));
    write_imagef(output,convert_int2(coord),rgba);
  } else {
    write_imagef(output,convert_int2(coord),color);
  }
   
}