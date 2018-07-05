/* Please Write the OpenCL Kernel(s) code here*/
#define PI_F 3.14159265358979323846f
float4 mangaCool(__read_only image2d_t input,float2 coord,int2 size,float arg){

  const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

  float dx = 1.0f / size.x;
  float dy = 1.0f / size.y;
  float c  = -1.0f / 8.0f; 
  
  
  float r = ((read_imagef(input,sampler, + (float2)(-dx, -dy)).x
          +   read_imagef(input,sampler,coord + (float2)(0.0, -dy)).x
          +   read_imagef(input,sampler,coord + (float2)( dx, -dy)).x
          +   read_imagef(input,sampler,coord + (float2)(-dx, 0.0)).x
          +   read_imagef(input,sampler,coord + (float2)( dx, 0.0)).x
          +   read_imagef(input,sampler,coord + (float2)(-dx,  dy)).x
          +   read_imagef(input,sampler,coord + (float2)(0.0,  dy)).x
          +   read_imagef(input,sampler,coord + (float2)( dx,  dy)).x) * c
          +    read_imagef(input,sampler,coord).x) * -19.2;

  float g = ((read_imagef(input,sampler,coord + (float2)(-dx, -dy)).y
          +   read_imagef(input,sampler,coord + (float2)(0.0, -dy)).y
          +   read_imagef(input,sampler,coord + (float2)( dx, -dy)).y
          +   read_imagef(input,sampler,coord + (float2)(-dx, 0.0)).y
          +   read_imagef(input,sampler,coord + (float2)( dx, 0.0)).y
          +   read_imagef(input,sampler,coord + (float2)(-dx,  dy)).y
          +   read_imagef(input,sampler,coord + (float2)(0.0,  dy)).y
          +   read_imagef(input,sampler,coord + (float2)( dx,  dy)).y) * c
          +   read_imagef(input,sampler, coord).y) * -9.6;

  float b = ((read_imagef(input,sampler,coord + (float2)(-dx, -dy)).z
          +   read_imagef(input,sampler,coord + (float2)(0.0, -dy)).z
          +   read_imagef(input,sampler,coord + (float2)( dx, -dy)).z
          +   read_imagef(input,sampler,coord + (float2)(-dx, 0.0)).z
          +   read_imagef(input,sampler,coord + (float2)( dx, 0.0)).z
          +   read_imagef(input,sampler,coord + (float2)(-dx,  dy)).z
          +   read_imagef(input,sampler,coord + (float2)(0.0,  dy)).z
          +   read_imagef(input,sampler,coord + (float2)( dx,  dy)).z) * c
          +   read_imagef(input,sampler, coord).z) * -4.0;

  if (r < 0.0) r = 0.0;
  if (g < 0.0) g = 0.0;
  if (b < 0.0) b = 0.0;
  if (r > 1.0) r = 1.0;
  if (g > 1.0) g = 1.0;
  if (b > 1.0) b = 1.0;

  float3 rgb = 1.0f - (float3)(r, g, b);
  
  
  float4 cool_color = (float4)(rgb - (arg), 1.0f);
  
  return cool_color;

}

__kernel void hengao_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,convert_int2(coord));
   float arg = 0.75f;
   
   if(arg == 0.5f){
    
     write_imagef(output,convert_int2(coord),color);
   }else if(arg == 1.0f){
       write_imagef(output,convert_int2(coord),(float4)(0,0,0,1));
   }else if(arg > 0.5f){
      int2 coordOffset = size / 2;
      float fd = 500.0 / tan((arg - 0.5f) * PI_F);

      float2 v = coord.xy - convert_float2(coordOffset);
      float d = length(v);
      float2 xy = v / d * fd * tan(clamp(d / fd, -0.5f * PI_F , 0.5f * PI_F )) + convert_float2(coordOffset);
      float2 tc = xy / convert_float2(size);
      if (all(isgreaterequal(tc, (float2)(0.0))) && all(islessequal(tc, (float2)(1.0)))) {
        color = mangaCool(input,coord,size,arg);
      } else {
        color = (float4)(0.0, 0.0, 0.0, 1.0);
      }
      write_imagef(output,convert_int2(coord),color);
   }else{
    int2 coordOffset = size / 2;
    float fd = 500.0 / tan((0.5 - arg) * PI_F);

    int2 v = convert_int2(coord.xy) - coordOffset;
    float d = length(convert_float2(v));
    float2  xy = convert_float2(v) / d * fd * atan(d/fd) + convert_float2(coordOffset);
    color = mangaCool(input,/*xy / convert_float2(size)*/xy,size,arg);
     write_imagef(output,convert_int2(coord),color);
   }
 
}