/* Please Write the OpenCL Kernel(s) code here*/
#define PI_F 3.14159265358979323846f
float2 rand2(float2 coord){
    float iptr;
    float noiseX = (fract(sin(dot(coord,(float2)(12.9898f,78.233f))) * 43758.5453f,&iptr));
    float noiseY = (fract(sin(dot(coord,(float2)(12.9898f,78.233f) * 2.0f)) * 43758.5433f,&iptr));
    
    return (float2)(noiseX,noiseY) * 0.004f;
}

float compare_depths(float depth1,float depth2,float near,float far){
    float depthTolerance = far / 5.0f;
    float occlusionTolerance = far / 100.0f;
    float diff = (depth1 - depth2);
    
    if(diff <= 0.0f){
        return 0.0f;
    }
    if(diff > depthTolerance){
        return 0.0f;
    }
    
    if(diff < occlusionTolerance){
        return 0.0f;
    }
    
    return 1.0f;
}

float read_depth(float2 coord, float color_red,float near,float far){
    float z_b = color_red;
    float z_n = 2.0f * z_b - 1.0f;
    float z_e = 2.0f * near * far / (far + near - z_n * (far - near));
    
    return z_e;
}


__kernel void ssao_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   
   float near =  0.20f;
   float far = 1.0f;
   float2 text_coord = (float2)(coord.x / (float)dim.x,coord.y / (float)dim.y);
   float depth = read_depth(text_coord,color.x,near,far);
   
   float aspect = (float)dim.x / (float)dim.y;
   
   float2 noise = rand2(coord);
   float z_b = color.x;//(color.x + color.y + color.z) / 3.0f;
   
   float w = (1.0f / dim.x) / clamp(z_b,0.1f,1.0f) + (noise.x * (1.0f - noise.x));
   float h = (1.0f / dim.y) / clamp(z_b,0.1f,1.0f) + (noise.y * (1.0f - noise.y));
   
   float pw,ph;
   
   float ao = 0.0f;
   float s = 0.0f;
   
   float fade = 4.0f;
   int rings = 5;
   int samples = 3;
   float strength = 2.0f;
   float offset = 0.0f;
   float d;
   for(int i = 0;i < rings;i++){
       fade *= 0.5f;
       for(int j = 0;j < samples * rings;j++){
           if(j >= samples * i) break;
           float step = PI_F * 2.0f / ((float)samples * (float)(i));
           float r = 4.0f * i;
           pw = r * cos((float)j * step);
           ph = r * sin((float)j * step);
           color = read_imagef(input,sampler,(float2)(coord.x + pw * w,coord.y + ph * h));
           z_b = color.x;
           d = read_depth((float2)(coord.x + pw * w,coord.y + ph * h),z_b,near,far);
           
           ao += compare_depths(depth,d,near,far) * fade;
           
           s += 1.0f * fade;
       }
   }
   
   ao /= s;
   ao = clamp(ao,0.0f,1.0f);
   ao = 1.0f - ao;
   ao = offset + (1.0f - offset) * ao;
   ao = pow(ao,strength);
   
   write_imagef(output,convert_int2(coord),(float4)(ao,ao,ao,1.0f));
}