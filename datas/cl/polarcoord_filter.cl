/* Please Write the OpenCL Kernel(s) code here*/
__kernel void polarcoord_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
  float2 coordRect = coord * convert_float2(size);
  float2 center = convert_float2(size) * 0.5f;
  float2 fromCenter = coordRect - center;

  float2 coordPolar = (float2)(
          atan2(fromCenter.x, fromCenter.y) * size.x / (2.0f * PI_F) + center.x,
          length(fromCenter) * 2.0f);
  float4 color;
  float arg = 0.2f;
  float2 tc = mix(coordRect, coordPolar, arg) / convert_float2(size);
   color = read_imagef(input, sampler,tc);
 /* if (all(isgreaterequal(tc, (float2)(0.0))) && all(islessequal(tc, (float2)(1.0)))) {
    color = read_imagef(input, sampler,tc);
  } else {
    color = (float4)(0.0, 0.0, 0.0, 1.0);
  }
  */
  write_imagef(output,convert_int2(coord),color);
}