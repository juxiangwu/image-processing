/* Please Write the OpenCL Kernel(s) code here*/
__kernel void swirl_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);
   
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   int2 center = (int2)(size.x / 2,size.y / 2);
   
   float2 coord_center = convert_float2(coord - center);
   
   float radius = size.y / 2.0f;
   float angle = 5.0f;
   float dist = length(convert_float2(coord_center));
   
   if (dist < radius) {
       
        float percent = (radius - dist) / radius;
            float theta = percent * percent * angle;
            float s = sin(theta);
            float c = cos(theta);
            coord_center = (float2)(
                coord_center.x * c - coord_center.y * s,
                coord_center.x * s + coord_center.y * c);
    }
    coord_center += convert_float2(center);
   
   float4 color = read_imagef(input,sampler,convert_int2(coord_center));
   
   
   
   write_imagef(output,convert_int2(coord),color);
}