/* Please Write the OpenCL Kernel(s) code here*/
#define PI_F 3.14159265358979323846f
float4 brazil_internal(__read_only image2d_t input,
                              float2 coord,int2 size,float arg){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;
   
   float4 src_rgba = read_imagef(input,sampler,coord);
   
   float xBlockSize = 0.01*0.1;
   float yBlockSize = xBlockSize * size.x / size.y;  // mutiply ratio
   float xCoord = (floor((coord.x-0.5)/xBlockSize)+0.5) * xBlockSize+0.5;
   float yCoord = (floor((coord.y-0.5)/yBlockSize)+0.5) * yBlockSize+0.5;
  
   
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
   
   return color;
}


__kernel void hengaoposter_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,convert_int2(coord));
   float arg = 0.7f;
   
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
        color = brazil_internal(input,coord,size,arg);
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
    color = brazil_internal(input,/*xy / convert_float2(size)*/xy,size,arg);
     write_imagef(output,convert_int2(coord),color);
   }
}