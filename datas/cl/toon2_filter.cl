/* Please Write the OpenCL Kernel(s) code here*/
float3 rgb2hsv(float r,float g,float b){
    float minv,maxv,delta;
    float3 res;
    
    minv = min(min(r,g),b);
    maxv = max(max(r,g),b);
    
    res.z = maxv; //v
    
    delta = maxv - minv;
    
    if(maxv != 0.0f){
        res.y = delta / maxv; // s
    }else{
        res.y = 0.0f;
        res.x = -1.0f;
        return res;
    }
    
    if(r == maxv){
        res.x = (g - b) / delta;
    }else if(g == maxv){
        res.x = 2.0f + (b - r) / delta;
    }else{
        res.x = 4.0f + (r - g) / delta;
    }
    
    res.x = res.x * 60.0f;
    
    if(res.x < 0.0f){
        res.x = res.x + 360.0f;
    }
    
    return res;
    
}

float3 hsv2rgb(float h,float s,float v){
    int i;
    float f,p,q,t;
    
    float3 res;
    
    if(s == 0.0f){
        res.x = res.y = res.z = v;
        return res;
    }
    
    h /= 60.0f;
    i = (int)(floor(h));
    
    f = h - (float)i;
    p = v * (1.0f - s);
    q = v * (1.0f - s * f);
    t = v * (1.0f - s * (1.0f - f));
    
    if(i == 0){
        res.x = v;
        res.y = t;
        res.z = p;
    }else if(i == 1){
        res.x = q;
        res.y = v;
        res.z = p;
    }else if(i == 2){
        res.x = p;
        res.y = v;
        res.z = t;
    }else if(i == 3){
        res.x = v;
        res.y = p;
        res.z = q;
    }
    
    return res;
}

#define HUE_LEV_COUNT 6
#define SAT_LEV_COUNT 7
#define VAL_LEV_COUNT 4


float nearest_level(float col,float mode,float * HueLevels,float* SatLenvels,float * ValLevels){
    int levelCount;
   
    if(mode == 0){
        levelCount = HUE_LEV_COUNT;
    }
    
    if(mode == 1){
        levelCount = SAT_LEV_COUNT;
    }
    
    if(mode == 2){
        levelCount = VAL_LEV_COUNT;
    }
    
    for(int i = 0;i < levelCount - 1;i++){
        if(mode == 0){
            if(col >= HueLevels[i] && col <= HueLevels[i + 1]){
                return HueLevels[i + 1];
            }
        }
        
        if(mode == 1){
            if(col >= SatLenvels[i] && col <= SatLenvels[i + 1]){
                return SatLenvels[i + 1];
            }
        }
        
        if(mode == 2){
            if(col >= ValLevels[i] && col <= ValLevels[i + 1]){
                return ValLevels[i + 1];
            }
        }
    }
    return 0.0f;
}

float avg_intensity(float4 pix){
    return (pix.x + pix.y + pix.z) / 3.0f;
}

float4 get_pixel(__read_only image2d_t input,__read_only sampler_t sampler,float2 coord,float x,float y){
    return read_imagef(input,sampler,coord + (float2)(x,y));
}

float isEdge(__read_only image2d_t input,__read_only sampler_t sampler,float2 coords,int2 size){
    float dxtex = 1.0f / (float)size.x;
    float dytey = 1.0f / (float)size.y;
    
    float pix[9];
    
    int k = -1;
    
    float delta;
    
    for(int i = -1;i <= 1;i++){
        for(int j = -1;j <= 1;j++){
            k++;
            pix[k] = avg_intensity(get_pixel(input,sampler,coords,(float)i * dxtex,(float)j * dytey));
        }
    }
    
    delta = (fabs(pix[1] - pix[7]) + fabs(pix[5] - pix[3]) + fabs(pix[0] - pix[8]) + fabs(pix[2] - pix[6])) / 4.0f;
    
    return clamp(5.5f * delta,0.0f,1.0f);
}

__kernel void toon2_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float HueLevels[HUE_LEV_COUNT] = {0.0f,80.0f,160.0f,240.0f,320.0f,360.0f};
   float SatLevels[SAT_LEV_COUNT] = {0.0f,0.15f,0.30f,0.45f,0.60f,0.80f,1.0f};
   float ValLevels[VAL_LEV_COUNT] = {0.0f,0.3f,0.6f,1.0f};
   
   float4 color = read_imagef(input,sampler,coord);
   float3 vHSV = rgb2hsv(color.x,color.y,color.z);
   vHSV.x = nearest_level(vHSV.x,0,HueLevels,SatLevels,ValLevels);
   vHSV.y = nearest_level(vHSV.y,1,HueLevels,SatLevels,ValLevels);
   vHSV.z = nearest_level(vHSV.z,2,HueLevels,SatLevels,ValLevels);
   
   float edg = isEdge(input,sampler,coord,dim);
   
   float3 vRGB = (edg >= 0.3f) ? (float3)(0.0f,0.0f,0.0f) : hsv2rgb(vHSV.x,vHSV.y,vHSV.z);
   
   write_imagef(output,convert_int2(coord),(float4)(vRGB,1.0f));
   
}