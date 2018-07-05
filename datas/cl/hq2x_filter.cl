/* Please Write the OpenCL Kernel(s) code here*/

__kernel void hq2x_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
 
   float2 texture_size = convert_float2(dim);
   float4 tc1,tc2,tc3,tc4;
   
   float dx = texture_size.x / 2;//0.5f * (1.0f / texture_size.x);
   float dy = texture_size.y / 2;//0.5f * (1.0f / texture_size.y);
   
   float2 dg1 = (float2)(dx,dy);
   float2 dg2 = (float2)(-dx,dy);
   float2 ddx = (float2)(dx,0.0f);
   float2 ddy = (float2)(0.0f,dy);
   
   tc1 = (float4)(coord - dg1,coord - ddy);
   tc2 = (float4)(coord - dg2,coord + ddx);
   tc3 = (float4)(coord + dg1,coord + ddy);
   tc4 = (float4)(coord + dg2,coord - ddx);
   
   const float mx = 0.325f;
   const float k = -0.250f;
   const float max_w = 0.25f;
   const float min_w = -0.05f;
   const float lum_add = 0.5f;
   
   float3 c00 = read_imagef(input,sampler,tc1.xy).xyz;
   float3 c10 = read_imagef(input,sampler,tc1.zw).xyz;
   float3 c20 = read_imagef(input,sampler,tc2.xy).xyz;
   float3 c01 = read_imagef(input,sampler,tc4.zw).xyz;
   float3 c11 = read_imagef(input,sampler,coord).xyz;
   float3 c21 = read_imagef(input,sampler,tc2.zw).xyz;
   float3 c02 = read_imagef(input,sampler,tc4.xy).xyz;
   float3 c12 = read_imagef(input,sampler,tc3.zw).xyz;
   float3 c22 = read_imagef(input,sampler,tc3.xy).xyz;
   
   float3 dt = (float3)(1.0f,1.0f,1.0f);
   
   float md1 = dot(fabs(c00 - c22),dt);
   float md2 = dot(fabs(c02 - c20),dt);

   float w1 = dot(fabs(c22 - c11),dt) * md2;
   float w2 = dot(fabs(c02 - c11),dt) * md1;
   float w3 = dot(fabs(c00 - c11),dt) * md2;
   float w4 = dot(fabs(c20 - c11),dt) * md1;

   float t1 = w1 + w2;
   float t2 = w2 + w4;
   
   float ww = max(t1,t2) + 0.0001;
   
   c11 = (w1 * c00 + w2 * c20 + w3 * c22 + w4 * c02 + ww * c11) / (t1 + t2 + ww);
   
   float lc1 = k / (0.12f * dot(c10 + c12 + c11,dt) + lum_add);
   float lc2 = k / (0.12f * dot(c01 + c21 + c11,dt) + lum_add);
   
   w1 = clamp(lc1 * dot(fabs(c11 - c10),dt) + mx,min_w,max_w);
   w2 = clamp(lc2 * dot(fabs(c11 - c21),dt) + mx,min_w,max_w);
   w3 = clamp(lc1 * dot(fabs(c11 - c12),dt) + mx,min_w,max_w);      
   w4 = clamp(lc2 * dot(fabs(c11 - c01),dt) + mx,min_w,max_w);
   
   float3 final = w1 * c10 + w2 * c21 + w3 * c12 + w4 * c01 + (1.0f - w1 - w2 - w3 - w4) * c11;
   
   write_imagef(output,convert_int2(coord),(float4)(c11,1.0f));
}
