/* Please Write the OpenCL Kernel(s) code here*/
__kernel void hardrockcafe_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   
    float dx = 1.0f / size.x;
   float dy = 1.0f / size.y;
   
  float3 upperLeft   = gray_internal(read_imagef(input,sampler, convert_float2(coord) + (float2)(0.0f, -dy)));
  float3 upperCenter = gray_internal(read_imagef(input,sampler, convert_float2(coord) + (float2)(0.0f, -dy)));
  float3 upperRight  = gray_internal(read_imagef(input,sampler, convert_float2(coord) + (float2)( dx, -dy)));
  float3 left        = gray_internal(read_imagef(input,sampler, convert_float2(coord) + (float2)(-dx, 0.0f)));
  float3 center      = gray_internal(read_imagef(input,sampler, convert_float2(coord) + (float2)(0.0f, 0.0f)));
  float3 right       = gray_internal(read_imagef(input,sampler, convert_float2(coord) + (float2)( dx, 0.0f)));
  float3 lowerLeft   = gray_internal(read_imagef(input,sampler, convert_float2(coord) + (float2)(-dx,  dy)));
  float3 lowerCenter = gray_internal(read_imagef(input,sampler, convert_float2(coord) + (float2)(0.0f,  dy)));
  float3 lowerRight  = gray_internal(read_imagef(input,sampler, convert_float2(coord) + (float2)( dx,  dy)));
  
  // vertical convolution
  //[ -1, 0, 1,
  //  -2, 0, 2,
  //  -1, 0, 1 ]
  float3 vertical  = upperLeft   * -1.0f
                 + upperCenter *  0.0f
                 + upperRight  *  1.0f
                 + left        * -2.0f
                 + center      *  0.0f
                 + right       *  2.0f
                 + lowerLeft   * -1.0f
                 + lowerCenter *  0.0f
                 + lowerRight  *  1.0f;

  // horizontal convolution
  //[ -1, -2, -1,
  //   0,  0,  0,
  //   1,  2,  1 ]
  float3 horizontal = upperLeft   * -1.0f
                  + upperCenter * -2.0f
                  + upperRight  * -1.0f
                  + left        *  0.0f
                  + center      *  0.0f
                  + right       *  0.0f
                  + lowerLeft   *  1.0f
                  + lowerCenter *  2.0f
                  + lowerRight  *  1.0f;


  float v = (vertical.x > 0 ? vertical.x : -vertical.x);
  float h = (horizontal.x > 0 ? horizontal.x : -horizontal.x);
  float m = ( v + h) / 4.0f;
  
  float arg = 0.8f;
  
  float4 dst_color = (float4)(mix(color.xyz, (float3)(v, h, m), arg), color.w);
  
  write_imagef(output,convert_int2(coord),dst_color);
   
}