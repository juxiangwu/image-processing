/* Please Write the OpenCL Kernel(s) code here*/
float3 gray_internal(float4 color) {
  float y = dot(color.xyz, (float3)(0.2126f, 0.7152f, 0.0722f));
  return (float3)(y,y,y);
}

__kernel void cartoon2_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,convert_float2(coord));
   
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
   
   
  float r = (vertical.x > 0 ? vertical.x : -vertical.x) + (horizontal.x > 0 ? horizontal.x : -horizontal.x);
  float g = (vertical.y > 0 ? vertical.x : -vertical.y) + (horizontal.y > 0 ? horizontal.y : -horizontal.y);
  float b = (vertical.z > 0 ? vertical.x : -vertical.z) + (horizontal.z > 0 ? horizontal.z : -horizontal.z);
  if (r > 1.0f) r = 1.0f;
  if (g > 1.0f) g = 1.0f;
  if (b > 1.0f) b = 1.0f;
  
  float4 edged = (float4)(color.xyz - (float3)(r, g, b), color.w);
  
  float arg = 1.0f;
    
   write_imagef(output,coord,(float4)(mix(color.xyz, edged.xyz, arg), color.w));
}