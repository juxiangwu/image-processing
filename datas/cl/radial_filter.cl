/* Please Write the OpenCL Kernel(s) code here*/

float2 deform( float2 p,float2 center){
    float2 uy;
    float time = 1.0f;
    float2 q = (float2)(sin(1.1 * time + p.x),sin(1.2 * time + p.y));
    float a = atan2(q.y,q.x);
    float r = sqrt(dot(q,q));
    
    uy.x = sin(0.0f + 1.0f * center.x) + p.x * sqrt(r * r + 1.0f);
    uy.y = sin(0.6f + 1.1f * center.y) + p.y * sqrt(r * r + 1.0f);
    
    return uy * 0.5f;
}

__kernel void radial_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   float2 position = (float2)(0,0);
   float2 resolution = (float2)(0.35f,0.35f);
   float2 p = -1.0f * convert_float2(dim) + 2.0f * (position + coord) / resolution;
   float2 s = p;
   
   float3 total = (float3)(0,0,0);
   
   float2 d = ((float2)(0.0f,0.0f) - p) / 40.0f;
   
   float w = 1.0f;
   
   for(int i = 0;i < 40;i++){
       float2 uy = deform(s,coord);
       float3 res = read_imagef(input,sampler,uy).xyz;
       res = smoothstep(0.1f,0.1f,res * res);
       total += w * res;
       w *= 0.99f;
       s += d;
   }
   
   total /= 40.0f;
  // float r = 1.5f / (1.0f + dot(p,p));
   float3 vvColor = (float3)(0.5f,0.5f,0.5f);
   float4 color = (float4)(total * vvColor,1.0f);
   write_imagef(output,convert_int2(coord),color);
   
}