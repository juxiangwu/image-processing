/* Please Write the OpenCL Kernel(s) code here*/
__kernel void bulge_pinch_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);
   
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   int2 center = (int2)(size.x / 2,size.y / 2);
   
   float2 coord_center = convert_float2(coord - center);
   
   float radius = size.y / 2.0f;
   float strength = 4.0f;
   float dist = length(convert_float2(coord_center));
   
   if (dist < radius) {
         float percent = dist / radius;
         if (strength > 0.0f) {
              coord_center *= mix(1.0f, smoothstep(0.0f, radius / dist, percent), strength * 0.75f);
         } else {
              coord_center *= mix(1.0f, pow(percent, 1.0f + strength * 0.75f) * radius / dist, 1.0f - percent);
         }
    }
    coord_center += convert_float2(center);
   
   float4 color = read_imagef(input,sampler,convert_int2(coord_center));
   
   
   
   write_imagef(output,convert_int2(coord),color);
}