#define PI_F 3.14159265358979323846f
#define LOG2_F 1.442695f

void filter2d_internal(__read_only image2d_t input,
                       __write_only image2d_t output,
                       const int maskWidth,
                       const int maskHeight,
                        float * mask,int compute_aver);
void color_matrix_4x5_internal(__read_only image2d_t input,__write_only image2d_t output,float * mask);

__kernel void gray(__read_only image2d_t input, __write_only image2d_t output){

    const sampler_t sampler = CLK_FILTER_NEAREST |
                          CLK_NORMALIZED_COORDS_FALSE |
                          CLK_ADDRESS_CLAMP_TO_EDGE;

    int2 size = get_image_dim(input);
    int2 coord = (int2)(get_global_id(0),get_global_id(1));

    float4 pixel = (float4)(0,0,0,1);

    if(coord.x >= size.x || coord.y >= size.y){
        return;
    }


    pixel = read_imagef(input,sampler,coord);
    pixel.x = pixel.y = pixel.z = (pixel.x + pixel.y + pixel.z) / 3.0f;

    write_imagef(output,coord,pixel);
}

__kernel void sharpen(__read_only image2d_t input,__write_only image2d_t output){
    const int sharpen_mask [9] = {-0,-2,0,-2,9,-2,0,-2,0};
    const sampler_t sampler = CLK_FILTER_NEAREST |
                          CLK_NORMALIZED_COORDS_FALSE |
                          CLK_ADDRESS_CLAMP_TO_EDGE;

    int2 size = get_image_dim(input);
    int2 coord = (int2)(get_global_id(0),get_global_id(1));


    float4 color = (float4)(0,0,0,1);


    int2 c00 = (int2)(coord.x - 1,coord.y - 1);
    int2 c01 = (int2)(coord.x,coord.y - 1);
    int2 c02 = (int2)(coord.x + 1,coord.y - 1);

    int2 c10 = (int2)(coord.x - 1,coord.y );
    int2 c11 = (int2)(coord.x,coord.y);
    int2 c12 = (int2)(coord.x + 1,coord.y);

    int2 c20 = (int2)(coord.x - 1,coord.y + 1);
    int2 c21 = (int2)(coord.x,coord.y + 1);
    int2 c22 = (int2)(coord.x + 1,coord.y + 1);

    float4 p00 = read_imagef(input,sampler,c00);
    float4 p01 = read_imagef(input,sampler,c01);
    float4 p02 = read_imagef(input,sampler,c02);

    float4 p10 = read_imagef(input,sampler,c10);
    float4 p11 = read_imagef(input,sampler,c11);
    float4 p12 = read_imagef(input,sampler,c12);

    float4 p20 = read_imagef(input,sampler,c20);
    float4 p21 = read_imagef(input,sampler,c21);
    float4 p22 = read_imagef(input,sampler,c22);

    color.x = p00.x * sharpen_mask[0] +
              p01.x * sharpen_mask[1] +
              p02.x * sharpen_mask[2] +
              p10.x * sharpen_mask[3] +
              p11.x * sharpen_mask[4] +
              p12.x * sharpen_mask[5] +
              p20.x * sharpen_mask[6] +
              p21.x * sharpen_mask[7] +
              p22.x * sharpen_mask[8] ;


   color.y =  p00.y * sharpen_mask[0] +
              p01.y * sharpen_mask[1] +
              p02.y * sharpen_mask[2] +
              p10.y * sharpen_mask[3] +
              p11.y * sharpen_mask[4] +
              p12.y * sharpen_mask[5] +
              p20.y * sharpen_mask[6] +
              p21.y * sharpen_mask[7] +
              p22.y * sharpen_mask[8] ;


   color.z =  p00.z * sharpen_mask[0] +
              p01.z * sharpen_mask[1] +
              p02.z * sharpen_mask[2] +
              p10.z * sharpen_mask[3] +
              p11.z * sharpen_mask[4] +
              p12.z * sharpen_mask[5] +
              p20.z * sharpen_mask[6] +
              p21.z * sharpen_mask[7] +
              p22.z * sharpen_mask[8] ;


   write_imagef(output,coord,color);
}

__kernel void gaussian(__read_only image2d_t input, __write_only image2d_t output){

    const int gaussian_mask[9] = {1,2,1,2,4,2,1,2,1};
    const sampler_t sampler = CLK_FILTER_NEAREST |
                          CLK_NORMALIZED_COORDS_FALSE |
                          CLK_ADDRESS_CLAMP_TO_EDGE;

    int2 size = get_image_dim(input);
    int2 coord = (int2)(get_global_id(0),get_global_id(1));


    float4 color = (float4)(0,0,0,1);


    int2 c00 = (int2)(coord.x - 1,coord.y - 1);
    int2 c01 = (int2)(coord.x,coord.y - 1);
    int2 c02 = (int2)(coord.x + 1,coord.y - 1);

    int2 c10 = (int2)(coord.x - 1,coord.y );
    int2 c11 = (int2)(coord.x,coord.y);
    int2 c12 = (int2)(coord.x + 1,coord.y);

    int2 c20 = (int2)(coord.x - 1,coord.y + 1);
    int2 c21 = (int2)(coord.x,coord.y + 1);
    int2 c22 = (int2)(coord.x + 1,coord.y + 1);

    float4 p00 = read_imagef(input,sampler,c00);
    float4 p01 = read_imagef(input,sampler,c01);
    float4 p02 = read_imagef(input,sampler,c02);

    float4 p10 = read_imagef(input,sampler,c10);
    float4 p11 = read_imagef(input,sampler,c11);
    float4 p12 = read_imagef(input,sampler,c12);

    float4 p20 = read_imagef(input,sampler,c20);
    float4 p21 = read_imagef(input,sampler,c21);
    float4 p22 = read_imagef(input,sampler,c22);

    color.x = p00.x * gaussian_mask[0] +
              p01.x * gaussian_mask[1] +
              p02.x * gaussian_mask[2] +
              p10.x * gaussian_mask[3] +
              p11.x * gaussian_mask[4] +
              p12.x * gaussian_mask[5] +
              p20.x * gaussian_mask[6] +
              p21.x * gaussian_mask[7] +
              p22.x * gaussian_mask[8] ;


    color.y = p00.y * gaussian_mask[0] +
              p01.y * gaussian_mask[1] +
              p02.y * gaussian_mask[2] +
              p10.y * gaussian_mask[3] +
              p11.y * gaussian_mask[4] +
              p12.y * gaussian_mask[5] +
              p20.y * gaussian_mask[6] +
              p21.y * gaussian_mask[7] +
              p22.y * gaussian_mask[8] ;


    color.z = p00.z * gaussian_mask[0] +
              p01.z * gaussian_mask[1] +
              p02.z * gaussian_mask[2] +
              p10.z * gaussian_mask[3] +
              p11.z * gaussian_mask[4] +
              p12.z * gaussian_mask[5] +
              p20.z * gaussian_mask[6] +
              p21.z * gaussian_mask[7] +
              p22.z * gaussian_mask[8] ;

    float div = 16.0f;

    color.x = color.x / div;
    color.y = color.y / div;
    color.z = color.z / div;


    write_imagef(output,coord,color);
}


__kernel void filter_average(__read_only image2d_t input, __write_only image2d_t output){
    const int average_mask[9] = {1,1,1,1,1,1,1,1,1};
    const sampler_t sampler = CLK_FILTER_NEAREST |
                          CLK_NORMALIZED_COORDS_FALSE |
                          CLK_ADDRESS_CLAMP_TO_EDGE;

    int2 size = get_image_dim(input);
    int2 coord = (int2)(get_global_id(0),get_global_id(1));


    float4 color = (float4)(0,0,0,1);


    int2 c00 = (int2)(coord.x - 1,coord.y - 1);
    int2 c01 = (int2)(coord.x,coord.y - 1);
    int2 c02 = (int2)(coord.x + 1,coord.y - 1);

    int2 c10 = (int2)(coord.x - 1,coord.y );
    int2 c11 = (int2)(coord.x,coord.y);
    int2 c12 = (int2)(coord.x + 1,coord.y);

    int2 c20 = (int2)(coord.x - 1,coord.y + 1);
    int2 c21 = (int2)(coord.x,coord.y + 1);
    int2 c22 = (int2)(coord.x + 1,coord.y + 1);

    float4 p00 = read_imagef(input,sampler,c00);
    float4 p01 = read_imagef(input,sampler,c01);
    float4 p02 = read_imagef(input,sampler,c02);

    float4 p10 = read_imagef(input,sampler,c10);
    float4 p11 = read_imagef(input,sampler,c11);
    float4 p12 = read_imagef(input,sampler,c12);

    float4 p20 = read_imagef(input,sampler,c20);
    float4 p21 = read_imagef(input,sampler,c21);
    float4 p22 = read_imagef(input,sampler,c22);

    color.x = p00.x * average_mask[0] +
              p01.x * average_mask[1] +
              p02.x * average_mask[2] +
              p10.x * average_mask[3] +
              p11.x * average_mask[4] +
              p12.x * average_mask[5] +
              p20.x * average_mask[6] +
              p21.x * average_mask[7] +
              p22.x * average_mask[8] ;


    color.y = p00.y * average_mask[0] +
              p01.y * average_mask[1] +
              p02.y * average_mask[2] +
              p10.y * average_mask[3] +
              p11.y * average_mask[4] +
              p12.y * average_mask[5] +
              p20.y * average_mask[6] +
              p21.y * average_mask[7] +
              p22.y * average_mask[8] ;


    color.z = p00.z * average_mask[0] +
              p01.z * average_mask[1] +
              p02.z * average_mask[2] +
              p10.z * average_mask[3] +
              p11.z * average_mask[4] +
              p12.z * average_mask[5] +
              p20.z * average_mask[6] +
              p21.z * average_mask[7] +
              p22.z * average_mask[8] ;

    float div = 9.0f;

    color.x = color.x / div;
    color.y = color.y / div;
    color.z = color.z / div;


    write_imagef(output,coord,color);
}

__kernel void filter_laplace(__read_only image2d_t input, __write_only image2d_t output){
    const int laplace_mask[9] = {1,1,1,1,-8,1,1,1,1};
    const sampler_t sampler = CLK_FILTER_NEAREST |
                          CLK_NORMALIZED_COORDS_FALSE |
                          CLK_ADDRESS_CLAMP_TO_EDGE;

    int2 size = get_image_dim(input);
    int2 coord = (int2)(get_global_id(0),get_global_id(1));


    float4 color = (float4)(0,0,0,1);


    int2 c00 = (int2)(coord.x - 1,coord.y - 1);
    int2 c01 = (int2)(coord.x,coord.y - 1);
    int2 c02 = (int2)(coord.x + 1,coord.y - 1);

    int2 c10 = (int2)(coord.x - 1,coord.y );
    int2 c11 = (int2)(coord.x,coord.y);
    int2 c12 = (int2)(coord.x + 1,coord.y);

    int2 c20 = (int2)(coord.x - 1,coord.y + 1);
    int2 c21 = (int2)(coord.x,coord.y + 1);
    int2 c22 = (int2)(coord.x + 1,coord.y + 1);

    float4 p00 = read_imagef(input,sampler,c00);
    float4 p01 = read_imagef(input,sampler,c01);
    float4 p02 = read_imagef(input,sampler,c02);

    float4 p10 = read_imagef(input,sampler,c10);
    float4 p11 = read_imagef(input,sampler,c11);
    float4 p12 = read_imagef(input,sampler,c12);

    float4 p20 = read_imagef(input,sampler,c20);
    float4 p21 = read_imagef(input,sampler,c21);
    float4 p22 = read_imagef(input,sampler,c22);

    color.x = p00.x * laplace_mask[0] +
              p01.x * laplace_mask[1] +
              p02.x * laplace_mask[2] +
              p10.x * laplace_mask[3] +
              p11.x * laplace_mask[4] +
              p12.x * laplace_mask[5] +
              p20.x * laplace_mask[6] +
              p21.x * laplace_mask[7] +
              p22.x * laplace_mask[8] ;


    color.y = p00.y * laplace_mask[0] +
              p01.y * laplace_mask[1] +
              p02.y * laplace_mask[2] +
              p10.y * laplace_mask[3] +
              p11.y * laplace_mask[4] +
              p12.y * laplace_mask[5] +
              p20.y * laplace_mask[6] +
              p21.y * laplace_mask[7] +
              p22.y * laplace_mask[8] ;


    color.z = p00.z * laplace_mask[0] +
              p01.z * laplace_mask[1] +
              p02.z * laplace_mask[2] +
              p10.z * laplace_mask[3] +
              p11.z * laplace_mask[4] +
              p12.z * laplace_mask[5] +
              p20.z * laplace_mask[6] +
              p21.z * laplace_mask[7] +
              p22.z * laplace_mask[8] ;

//    float div = 9.0f;

//    color.x = color.x / div;
//    color.y = color.y / div;
//    color.z = color.z / div;


    write_imagef(output,coord,color);
}

__kernel void prewitt(__read_only image2d_t input, __write_only image2d_t output){

    const int prewitt_mask_h [9] = {1,1,1,0,0,0,-1,-1,-1};
    const int prewitt_mask_v [9] = {-1,0,1,-1,0,1,-1,0,1};

    const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

     int2 coord = (int2)(get_global_id(0),get_global_id(1));
     const int maskSize = 3;
     const int maskrows = maskSize / 2;
     const int maskcols = maskSize / 2;

     float4 color = (float4)(0,0,0,1);
     float4 colorv = (float4)(0,0,0,1);
     float4 colorh = (float4)(0,0,0,1);

     int mask_idx = 0;
     for(int y = -maskrows;y <= maskrows;++y){

         for(int x = -maskcols; x <= maskcols;++x){
             float4 srcColor = read_imagef(input,sampler,(int2)(x + coord.x,y + coord.y));

             colorh.x += srcColor.x * prewitt_mask_h[mask_idx];
             colorh.y += srcColor.y * prewitt_mask_h[mask_idx];
             colorh.z += srcColor.z * prewitt_mask_h[mask_idx];

             colorv.x += srcColor.x * prewitt_mask_v[mask_idx];
             colorv.y += srcColor.y * prewitt_mask_v[mask_idx];
             colorv.z += srcColor.z * prewitt_mask_v[mask_idx];

             color.x = colorh.x > colorv.x ? colorh.x : colorv.x;
             color.y = colorh.y > colorv.y ? colorh.y : colorv.y;
             color.z = colorh.z > colorv.z ? colorh.z : colorv.z;

//             color.x = colorh.x + colorv.x;
//             color.y = colorh.y + colorv.y;
//             color.z = colorh.z + colorv.z;

             mask_idx += 1;
         }
     }

     write_imagef(output,coord,color);
}

__kernel void kirsch(__read_only image2d_t input, __write_only image2d_t output){


    const int kirsh_mask_1[3][3] = {{5,5,5},{-3,0,-3},{-3,-3,-3}};
    const int kirsh_mask_2[3][3] = {{-3,5,5},{-3,0,5},{-3,-3,-3}};
    const int kirsh_mask_3[3][3] = {{-3,-3,5},{-3,0,5},{-3,-3,5}};
    const int kirsh_mask_4[3][3] = {{-3,-3,-3},{-3,0,5},{-3,5,5}};
    const int kirsh_mask_5[3][3] = {{-3,-3,-3},{-3,0,-3},{5,5,5}};
    const int kirsh_mask_6[3][3] = {{-3,-3,-3},{5,0,-3},{5,5,-3}};
    const int kirsh_mask_7[3][3] = {{5,-3,-3},{5,0,-3},{5,-3,-3}};
    const int kirsh_mask_8[3][3] = {{5,5,-3},{5,0,-3},{-3,-3,-3}};


    const sampler_t sampler = CLK_FILTER_NEAREST |
                         CLK_NORMALIZED_COORDS_FALSE |
                         CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    const int maskwidth = 3;
    const int maskheight = 3;
    const int maskrows = maskwidth / 2;
    const int maskcols = maskheight / 2;
    float3 sum[8] = {(float3)(0,0,0)};
    float4 color = (float4)(0,0,0,1);
    int maskIdx = 0;
    for(int y = 0;y <= maskrows;y++){
       for(int x = 0; x <= maskcols;x++){
            float4 srcColor = read_imagef(input,sampler,(int2)(x + coord.x,y + coord.y));

            sum[0].x +=  srcColor.x * kirsh_mask_1[x][y];
            sum[0].y +=  srcColor.y * kirsh_mask_1[x][y];
            sum[0].z +=  srcColor.z * kirsh_mask_1[x][y];

            sum[1].x +=  srcColor.x * kirsh_mask_2[x][y];
            sum[1].y +=  srcColor.y * kirsh_mask_2[x][y];
            sum[1].z +=  srcColor.z * kirsh_mask_2[x][y];

            sum[2].x +=  srcColor.x * kirsh_mask_3[x][y];
            sum[2].y +=  srcColor.y * kirsh_mask_3[x][y];
            sum[2].z +=  srcColor.z * kirsh_mask_3[x][y];

            sum[3].x +=  srcColor.x * kirsh_mask_4[x][y];
            sum[3].y +=  srcColor.y * kirsh_mask_4[x][y];
            sum[3].z +=  srcColor.z * kirsh_mask_4[x][y];

            sum[4].x +=  srcColor.x * kirsh_mask_5[x][y];
            sum[4].y +=  srcColor.y * kirsh_mask_5[x][y];
            sum[4].z +=  srcColor.z * kirsh_mask_5[x][y];

            sum[5].x +=  srcColor.x * kirsh_mask_6[x][y];
            sum[5].y +=  srcColor.y * kirsh_mask_6[x][y];
            sum[5].z +=  srcColor.z * kirsh_mask_6[x][y];

            sum[6].x +=  srcColor.x * kirsh_mask_7[x][y];
            sum[6].y +=  srcColor.y * kirsh_mask_7[x][y];
            sum[6].z +=  srcColor.z * kirsh_mask_7[x][y];

            sum[7].x +=  srcColor.x * kirsh_mask_8[x][y];
            sum[7].y +=  srcColor.y * kirsh_mask_8[x][y];
            sum[7].z +=  srcColor.z * kirsh_mask_8[x][y];

            maskIdx += 1;
         }
     }



    for(int i = 0;i < 8;i++){

        float maxX = 0;
        float maxY = 0;
        float maxZ = 0;
        if(maxX < sum[i].x)maxX = sum[i].x;

        if(maxX < 0) maxX = 0;

        if(maxX > 1) maxX = 1;

        if(maxY < sum[i].y)maxX = sum[i].y;

        if(maxY < 0) maxY = 0;

        if(maxY > 1) maxY = 1;

        if(maxZ < sum[i].z)maxX = sum[i].z;

        if(maxZ < 0) maxZ = 0;

        if(maxZ > 1) maxZ = 1;

        color.x = maxX;
        color.y = maxY;
        color.z = maxZ;
    }



    write_imagef(output,coord,color);

}

__kernel void filter2d(__read_only image2d_t input,
                       __write_only image2d_t output,
                       const int maskWidth,
                       const int maskHeight,
                       __global float * mask){

   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));

   const int maskrows = maskWidth / 2;
   const int maskcols = maskHeight / 2;

   float4 color = (float4)(0,0,0,1);
   int idx = 0;

   for(int y = -maskrows;y <= maskrows;++y){
      for(int x = -maskcols; x <= maskcols;++x){
        float4 srcColor = read_imagef(input,sampler,(int2)(x + coord.x,y + coord.y));
        color.x += srcColor.x * mask[idx];
        color.y += srcColor.y * mask[idx];
        color.z += srcColor.z * mask[idx];
        idx++;
      }
   }
  write_imagef(output,coord,color);
}

__kernel void selection_gray_filter(__read_only image2d_t input,
                                    __write_only image2d_t output){

}

__kernel void gamma_correction(__read_only image2d_t input,
                               __write_only image2d_t output){
     const sampler_t sampler = CLK_FILTER_NEAREST |
                               CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE;

     const int2 size = get_image_dim(input);

     int2 coord = (int2)(get_global_id(0),get_global_id(1));
    /*
     float4 color = (float4)(0,0,0,1);
     float gamma = 2.2f;
     float lut[256] = {0};
     for(int i = 0;i < 256;i++){
        float val = pow(i / 255.0f,gamma);
        lut[i] = val;
     }

     float4 srcColor = read_imagef(input,sampler,coord);
     color.x = srcColor.x * lut[convert_uchar_sat(srcColor.x * 255.0f)];
     color.y = srcColor.y * lut[convert_uchar_sat(srcColor.y * 255.0f)];
     color.z = srcColor.z * lut[convert_uchar_sat(srcColor.z * 255.0f)];

     write_imagef(output,coord,color);
    */
     float gamma = 2.0f;
     float3 srcRGB = read_imagef(input,sampler,coord).xyz;
     float3 dstRGB = pow(srcRGB,(float3)(1.0f / gamma,1.0f / gamma,1.0f / gamma));
     
     write_imagef(output,coord,(float4)(dstRGB,1.0f));
     
}

__kernel void media_filter(__read_only image2d_t input,
                               __write_only image2d_t output){
     const sampler_t sampler = CLK_FILTER_NEAREST |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));

    int tamanho = 5;
    float ratio = 2.0f;
    int linha_meio,coluna_meio;
    float cor1,cor2,cor3;
    float soma1 = 0.0f,soma2 = 0.0f,soma3 = 0.0f;
    float cor_resultante1,cor_resultante2,cor_resultante3;
    float largura = size.x;
    float altura = size.y;
   
    for(linha_meio = 0;linha_meio < tamanho;linha_meio++){
        for(coluna_meio = 0;coluna_meio < tamanho;coluna_meio++){
            float linha = linha_meio + (coord.x - ratio);
            float coluna = coluna_meio + (coord.y - ratio);
            
            if(linha < 0){
                linha = 0;
            }
            
            if(coluna < 0){
                coluna = 0;
            }
            
            if(coluna > largura){
                coluna = largura - 1;
            }
            
            float4 color = read_imagef(input,sampler,coord + (int2)(linha,coluna));
            soma1 += color.x;
            soma2 += color.y;
            soma3 += color.z;
        }
    }
    
    cor_resultante1 = (soma1 / (tamanho * tamanho));
    cor_resultante2 = (soma2 / (tamanho * tamanho));
    cor_resultante3 = (soma3 / (tamanho * tamanho));
    
    write_imagef(output,coord,(float4)(cor_resultante1,cor_resultante2,cor_resultante3,1.0f));
}

__kernel void rgb2hsi(__read_only image2d_t input,
                               __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));

    float4 color = (float4)(0,0,0,1);

    float r,g,b,num,den,dim,minVal,theta,H,S,I;

    float4 srcColor = read_imagef(input,sampler,coord);

    b = srcColor.x;
    g = srcColor.y;
    r = srcColor.z;

    num = 0.5 * ((r - g) + (r - b));
    den = sqrt((r - g) * (r - g) + (r - b) * (g - b));

    if(den == 0) {
        H = 0;
    }else{
        theta = acos(num / den);
        if(b > g){
            H = (2 * 3.14159265358979323846f - theta) / (2 * 3.14159265358979323846f);
        }else{
            H = theta / (2 * 3.14159265358979323846f);
        }
    }

    minVal = (b > g) ? g : b;
    minVal = (minVal > r) ? r : minVal;

    den = r + g + b;
    if(den == 0){
        S = 0;
    }else{
        S = 1 - 3 * minVal / den;
    }

    I = (r + g + b) / 3;

    color.x = H;
    color.y = S;
    color.z = I;

    write_imagef(output,coord,color);
}


__kernel void log_filter(__read_only image2d_t input,
                               __write_only image2d_t output){
     const sampler_t sampler = CLK_FILTER_NEAREST |
                               CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE;

     const int2 size = get_image_dim(input);

     int2 coord = (int2)(get_global_id(0),get_global_id(1));

     float4 color;

     float4 srcColor = read_imagef(input,sampler,coord);

     float3 logval = log(1.0f + srcColor.xyz);
     float scale = 2.0f;
     color = scale * (float4)(logval,1.0f);

     write_imagef(output,coord,color);
}


__kernel void contrast_filter(__read_only image2d_t input,
                               __write_only image2d_t output){
     const sampler_t sampler = CLK_FILTER_NEAREST |
                               CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE;

     const int2 size = get_image_dim(input);

     int2 coord = (int2)(get_global_id(0),get_global_id(1));

     float4 color;

     float4 srcColor = read_imagef(input,sampler,coord);
     float scale = -5.0f;
     float3 contrast = (exp(2 * (srcColor.xyz - 0.5f) * scale) - 1) / (exp(2 * (srcColor.xyz - 0.5f) * scale) + 1);

     color = scale * (float4)(contrast,1.0f);

     write_imagef(output,coord,color);
}

__kernel void contrast2_filter(__read_only image2d_t input,
                               __write_only image2d_t output){
     const sampler_t sampler = CLK_FILTER_NEAREST |
                               CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE;

     const int2 size = get_image_dim(input);

     int2 coord = (int2)(get_global_id(0),get_global_id(1));
     
     float arg = 0.5f;
     float4 color = read_imagef(input,sampler,coord);
     float slope = arg > 0.5f ? 1.0f/(2.0f - 2.0f * arg) : 2.0f * arg;
     float4 dstcolor = (float4)((color.xyz-0.5f)*slope+0.5f, color.w);
     
     write_imagef(output,coord,dstcolor);
}


__kernel void luminace_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));

    float4 color;

    float4 srcColor = read_imagef(input,sampler,coord);

    float gray = srcColor.x * 0.299f + srcColor.y * 0.587f + srcColor.z * 0.114f;

    float thresh = 0.32f;
    
    
    
    if(gray > thresh){
        color = (float4)(mix(srcColor,(float4)(1.0f),0.5f).xyz,1.0f);
    }else{
        color = (float4)((float3)(0.0f),1.0f);    
    }

    write_imagef(output,coord,color);
}

/*
    one dimesional guassian function
*/
float gaussian_dim1d_blur_compute(int i,float sigma){
    float sigmaq = sigma * sigma;
    float value = 0.0f;
    value = exp(-((i * i) / (2.0f * sigmaq))) / sqrt(2.0f * 3.14159265358979323846f * sigmaq);
    return value;
}

__kernel void horizontal_blur_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));

   
    float weight[31] = {0.0f};
    float sum = 0.0f;
    float sigma = 15.0f;
    
    for(int i = 1;i <= 31;i++){
        weight[i - 1] = gaussian_dim1d_blur_compute(i,sigma);
        sum += 2.0f * weight[i - 1];
    }
    
    for(int i = 0;i < 31;i++){
        weight[i] = weight[i] / sum;
    }

    
    float3 dstColor;
    dstColor = read_imagef(input,sampler,coord).xyz * weight[0];
    for(int i = 0;i < 31;i++){
       // if(coord.x != 0){
            int2 offset = (int2)(i ,0);
            dstColor += read_imagef(input,sampler,coord - offset).xyz * weight[i];
            dstColor += read_imagef(input,sampler,coord + offset).xyz * weight[i];
       // }
    }
    
    float4 color = (float4)(dstColor,1.0f);
    
    write_imagef(output,coord,color);
}

__kernel void vertical_blur_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));

   
    float weight[31] = {0.0f};
    float sum = 0.0f;
    float sigma = 15.0f;
    
    for(int i = 1;i <= 31;i++){
        weight[i - 1] = gaussian_dim1d_blur_compute(i,sigma);
        sum += 2.0f * weight[i - 1];
    }
    
    for(int i = 0;i < 31;i++){
        weight[i] = weight[i] / sum;
    }

    
    float3 dstColor;
    dstColor = read_imagef(input,sampler,coord).xyz * weight[0];
    for(int i = 0;i < 31;i++){
       // if(coord.x != 0){
            int2 offset = (int2)(0 ,i);
            dstColor += read_imagef(input,sampler,coord - offset).xyz * weight[i];
            dstColor += read_imagef(input,sampler,coord + offset).xyz * weight[i];
       // }
    }
    
    float4 color = (float4)(dstColor,1.0f);
    
    write_imagef(output,coord,color);
}

__kernel void vert_horiz_blur_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));

   
    float weight[31] = {0.0f};
    float sum = 0.0f;
    float sigma = 15.0f;
    
    for(int i = 1;i <= 31;i++){
        weight[i - 1] = gaussian_dim1d_blur_compute(i,sigma);
        sum += 2.0f * weight[i - 1];
    }
    
    for(int i = 0;i < 31;i++){
        weight[i] = weight[i] / sum;
    }

    
    float3 dstColor;
    dstColor = read_imagef(input,sampler,coord).xyz * weight[0];
    for(int i = 0;i < 31;i++){
       // if(coord.x != 0){
            int2 offset = (int2)(i ,i);
            dstColor += read_imagef(input,sampler,coord - offset).xyz * weight[i];
            dstColor += read_imagef(input,sampler,coord + offset).xyz * weight[i];
       // }
    }
    
    float4 color = (float4)(dstColor,1.0f);
    
    write_imagef(output,coord,color);
}

__kernel void bloom_old_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    
    float3 sum = (float3)(0,0,0);
    
    int mask_width = 4;
    int mask_height = 3;
    
    for(int i = -mask_width;i < mask_width;i++){
        for(int j = -mask_height;j < mask_height;j++){
            sum += read_imagef(input,sampler,coord + (int2)(j,i)).xyz * 0.2f;
        }
    }
    
    float4 srcColor= read_imagef(input,sampler,coord);
    float3 dstRGB;
    if(srcColor.x < 0.3f){
        dstRGB = sum * sum * 0.012f + srcColor.xyz;
    }else{
        if(srcColor.x < 0.5f){
            dstRGB = sum * sum * 0.009f + srcColor.xyz;
        }else{
            dstRGB = sum * sum * 0.0075f + srcColor.xyz;
        }
    }
    
    float4 color = (float4)(dstRGB,1.0f);
    
    write_imagef(output,coord,color);
}

__kernel void bloom_new_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    
    float4 srcColor = read_imagef(input,sampler,coord);
    
    float avg = (srcColor.x + srcColor.y + srcColor.z) / 3.0f;
    
    float3 sum = (float3)(0,0,0);
    
    float mask_width = 5;
    float mask_height = 5;
    
    float bloom_strength = 0.015;
    
    for(int i = -mask_width;i < mask_width; i++){
        for(int j = -mask_height; j < mask_height; j++){
            sum += read_imagef(input,sampler,coord + (int2)(i,j)).xyz * bloom_strength;
        }
    }
    
    float3 dstRGB;
    
    if(avg < 0.025f){
        dstRGB = srcColor.xyz + sum * 0.335f;
    }else if( avg < 0.10f){
        dstRGB = srcColor.xyz + (sum * sum) * 0.5f;
    }else if(avg < 0.88f){
        dstRGB = srcColor.xyz + (sum * sum) * 0.333f;
    }else if(avg >= 0.88f){
        dstRGB = srcColor.xyz + sum;
    }else{
        dstRGB = srcColor.xyz;
    }
    
    write_imagef(output,coord,(float4)(dstRGB,1.0f));
    
}

__kernel void bloom_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    float2 coord = (float2)(get_global_id(0),get_global_id(1));
    
    
    float4 sum = (float4)(0.0f,0.0f,0.0f,0.0f);
    float a = 0.15f;
    float g = 0.15f;
    float e = 0.25f;
    float b = 0.25f;
    float f = 0.35f;
    float c = 0.55f;
    float d = 0.45f;
    
    sum += read_imagef(input,sampler,coord + (float2)(-3,-4) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-3,-3) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-2,-3) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-1,-3) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-0,-3) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(1,-3) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(2,-3) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(3,-3) * a) * g;
    
    sum += read_imagef(input,sampler,coord + (float2)(-4,-2) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-3,-2) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-2,-2) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-1,-2) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-0,-2) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(1,-2) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(2,-2) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(3,-2) * a) * g;
    
    sum += read_imagef(input,sampler,coord + (float2)(-4,-1) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-3,-1) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-2,-1) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-1,-1) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-0,-1) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(1,-1) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(2,-1) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(3,-1) * a) * g;
    
    sum += read_imagef(input,sampler,coord + (float2)(-4,0) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-3,0) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-2,0) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-1,0) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-0,0) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(1,0) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(2,0) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(3,0) * a) * g;
    
    sum += read_imagef(input,sampler,coord + (float2)(-4,1) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-3,1) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-2,1) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-1,1) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-0,1) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(1,1) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(2,1) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(3,1) * a) * g;

    sum += read_imagef(input,sampler,coord + (float2)(-4,2) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-3,2) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-2,2) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-1,2) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(-0,2) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(1,2) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(2,2) * a) * g;
    sum += read_imagef(input,sampler,coord + (float2)(3,2) * a) * g;
    
    float4 color = read_imagef(input,sampler,coord);
    
    if(color.x < e){
        float4 rgba = sum * sum * b + color;
        rgba.w = 1.0f;
        write_imagef(output,convert_int2(coord),rgba);
    }else{
        if(color.x < f){
            float4 rgba = sum * sum * c + color;
            rgba.w = 1.0f;
            write_imagef(output,convert_int2(coord),rgba);
        }else{
            float4 rgba = sum * sum * d + color;
            rgba.w = 1.0f;
            write_imagef(output,convert_int2(coord),rgba);
        }
    }
}



__kernel void radial_blur(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    
    int samples[10] = {-8,-5,-3,-2,-1,1,2,3,5,8};
    
    int2 dir = (size / 2) - coord;
    
    float dist = sqrt((float)(dir.x * dir.x) + (float)(dir.y * dir.y));
    
    dir = dir / (int)dist;
    
    float3 sum = (float3)(0,0,0);
    
    float4 srcColor = read_imagef(input,sampler,coord);
    int sampleDist = 2;
    float sampleStrength = 3.2f;
    
    for(int i = 0; i < 10;i++){
        sum += read_imagef(input,sampler,coord + dir * samples[i] * sampleDist).xyz;
    }
    
    sum *= 1.0f / 11.0f;
    
    float t = dist * sampleStrength;
    
    t = clamp(t,0.0f,1.0f);
    
    float3 dstRGB = mix(srcColor.xyz,sum,t);
    
    float4 dstColor = (float4)(dstRGB,1.0f);
    
    write_imagef(output,coord,dstColor);
    
}

__kernel void linearize_depth(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float n = 1.0f; //camera z near
   float f = 100.0f; // camera z far
   
   float4 srcColor = read_imagef(input,sampler,coord);
   
   float z = srcColor.x;
   
   float depth = 1 - (2.0f * n) / (f + n - z * (f - n));
   
   float3 zvec = (float3)(depth,depth,depth);
   
   const float LOG2 = 1.442695f;
   
   float fdistance = 10.0f;
   
   float fogColorStrength = exp2( -fdistance * fdistance * zvec * zvec * LOG2).x;
   
   fogColorStrength = clamp(fogColorStrength,0.0f,1.0f);
   
   float3 fogColor = (float3)(1.0f,1.0f,1.0f);
   
   float3 dstRGB = mix(fogColor,srcColor.xyz, 1 - fogColorStrength);
   
   write_imagef(output,coord,(float4)(dstRGB,1.0f));
   
}

float3 tone_map(float3 hdrRGB,float exposure){
    
    float3 dstRGB = 1.0f - exp2(-hdrRGB * exposure);
    return dstRGB;
    
}


__kernel void tone_map_depth(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    
    float exposure = 2.5f;
    
    float4 color = read_imagef(input,sampler,coord);
    
    float4 rgba = 1.0f - exp2(-color * exposure);
    rgba.w = 1.0f;
    
    write_imagef(output,coord,rgba);
}

float rand(float2 co){
    float iptr = 1.0f;
    return fract(sin(dot(co.xy,(float2)(12.9898f,78.233f))) * 43758.5453f,&iptr);
}



__kernel void ice_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
 
    float2 v1 = (float2)(0.001f,0.001f);
    float2 v2 = (float2)(0.000f,0.000f);
    float iptr = 1.0f;
    float rnd_scale = 1.0f;
    float2 coordf = (float2)(coord.x,coord.y);
    float rnd_factor = 1.5f;
    float rnd = fract(sin(dot(coordf,v1)) + cos(dot(coordf,v2)) * rnd_scale,&iptr);
    int2 offset = (int2)((int)(rnd * rnd_factor * coord.x),(int)(rnd * rnd_factor * coord.y) );
    float3 srcRGB = read_imagef(input,sampler,offset).xyz;
    
    write_imagef(output,coord,(float4)(srcRGB,1.0f));
    
}

__kernel void protanopia_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
 
   /*
    float16 protanopia_mask = (float16)(0.20f,0.99f,-0.19f,0.0f,
                                        0.16f,0.79f,0.04f,0.0f,
                                        0.01f,-0.01f,1.00f,0.0f,
                                        0.0f,0.0f,0.0f,1.0f);
                                        */
    float4 protanopia_mask_1 = (float4)(0.20f,0.99f,-0.19f,0.0f);
    float4 protanopia_mask_2 = (float4)(0.16f,0.79f,0.04f,0.0f);
    float4 protanopia_mask_3 = (float4)(0.01f,-0.01f,1.00f,0.0f);
    float4 protanopia_mask_4 = (float4)(0.0f,0.0f,0.0f,1.0f);
    
    float4 protanopia_mask[4] = {protanopia_mask_1,protanopia_mask_2,protanopia_mask_3,protanopia_mask_4};
   
    //float4 v1 = protanopia_mask[0];
    float4 srcRGBA = read_imagef(input,sampler,coord);
    
    float4 dstRGBA;
    float sum[4] = {0.0f};
    for(int i = 0;i < 4;i++){
        
        sum[i] += dot(protanopia_mask[i],srcRGBA);
    }
    
    dstRGBA = (float4)(sum[0],sum[1],sum[2],sum[3]);
    
    write_imagef(output,coord,dstRGBA);
    
   // for(;;);
}

__kernel void reflection_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
 
    float4 p_mat[4];
   
    float4 srcRGBA = read_imagef(input,sampler,coord);
    
    srcRGBA *= M_SQRT2_F / length(srcRGBA);
     
    p_mat[0] = (float4)(1.0f,0.0f,0.0f,0.0f) - (srcRGBA * srcRGBA.x);
    p_mat[1] = (float4)(0.0f,1.0f,0.0f,0.0f) - (srcRGBA * srcRGBA.y);
    p_mat[2] = (float4)(0.0f,0.0f,1.0f,0.0f) - (srcRGBA * srcRGBA.z);
    p_mat[3] = (float4)(0.0f,0.0f,0.0f,1.0f) - (srcRGBA * srcRGBA.w);

    float4 dstRGBA;
    
    float4 x_vec = (float4)(0.5f,0.5f,0.5f,1.0f);
    
    dstRGBA.x = dot(p_mat[0],x_vec);
    dstRGBA.y = dot(p_mat[1],x_vec);
    dstRGBA.z = dot(p_mat[2],x_vec);
    dstRGBA.w = dot(p_mat[3],x_vec);

    write_imagef(output,coord,dstRGBA);
    
}

__kernel void brick_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    
    float3 srcRGB = read_imagef(input,sampler,coord).xyz;
    
    float gray = (srcRGB.x + srcRGB.y + srcRGB.z) / 3;
    
    float thresh = 128.0f / 255.0f;
    
    gray = gray >= thresh ? 255 : 0;
    
    float4 dstRGBA = (float4)(gray,gray,gray,1.0f);
    

    write_imagef(output,coord,dstRGBA);
    
}

__kernel void feather_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    int width = size.x;
    int height = size.y;
    float ratio = (width > height ? width / height : height / width);
    
    int cx = size.x >> 1;
    int cy = size.y >> 1;
    
    float featherSize = 0.25f;
    int maxval = cx * cx + cy * cy;
    int minval = (int)(maxval * (1 - featherSize));
    
    int diff = maxval - minval;
  
    float3 srcRGB = read_imagef(input,sampler,coord).xyz;
    
    int dx = cx - coord.x;
    int dy = cx - coord.y;
    
    if(size.x > size.y){
        dx = (dx * ratio);
    }else{
        dy = (dy * ratio);
    }
    
    int distSq = dx * dx + dy * dy;
    
    float v = ((float)distSq / diff);
    
    float4 dstRGBA = (float4)((srcRGB + v) ,1.0f);
    

    write_imagef(output,coord,dstRGBA);
    
}


__kernel void scale_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_LINEAR |
                              CLK_NORMALIZED_COORDS_TRUE |
                              CLK_ADDRESS_CLAMP;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    
    int width = size.x;
    int height = size.y;
    
    float scale = 0.25f;
    float scalex = 1.0f / (scale * width);
    float scaley = 1.0f / (scale * height);
    
    float2 scale_coord = convert_float2(coord) * (float2)(scalex,scaley);
    
    float4 color = read_imagef(input,sampler,scale_coord);
    
    write_imagef(output,coord,color);
}

__kernel void rotate_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    
    int width = size.x;
    int height = size.y;
    
    int xc = width / 2;
    int yc = height / 2;
    
    float angle = 45.0f;
    float theta = angle * PI_F / 180.0f;
    
    float xpos = (coord.x - xc) * cos(theta) - (coord.y - yc) * sin(theta) + xc;
    float ypos = (coord.x - xc) * sin(theta) + (coord.y - yc) * cos(theta) + yc;
    
    int2 pos = convert_int2((float2)(xpos,ypos));
    if(pos.x >= 0 && pos.x < width && pos.y >= 0 && pos.y < height){
        float4 color = read_imagef(input,sampler,pos);
    
        write_imagef(output,coord,color);
    }
}

__kernel void add_filter(__read_only image2d_t input,__read_only image2d_t input2,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    if(coord.x >= 0 && coord.x < size.x && coord.y >= 0 && coord.y < size.y){
        float4 color = read_imagef(input,sampler,coord) + read_imagef(input2,sampler,coord);
        color.w = 1.0f;

        write_imagef(output,coord,color);
    }
}

__kernel void sub_filter(__read_only image2d_t input,__read_only image2d_t input2,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    if(coord.x >= 0 && coord.x < size.x && coord.y >= 0 && coord.y < size.y){
        float4 color = read_imagef(input,sampler,coord) - read_imagef(input2,sampler,coord);
        color.w = 1.0f;

        write_imagef(output,coord,color);
    }
}



__kernel void add_weighted_filter(__read_only image2d_t input,__read_only image2d_t input2,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP;

    const int2 size = get_image_dim(input);
    
    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    
    float alpha = 0.45f;
    float beta = 0.55f;
    float gamma = 2.0f;
    
    if(coord.x >= 0 && coord.x < size.x && coord.y >= 0 && coord.y < size.y){
        float3 color = read_imagef(input,sampler,coord).xyz * 255.0f * alpha + read_imagef(input2,sampler,coord).xyz * 255.0f * beta + gamma;
        
        color.xyz = color.xyz / 255.0f;
        write_imagef(output,coord,(float4)(color,1.0f));
    }
}

__kernel void xor_filter(__read_only image2d_t input,__read_only image2d_t input2,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP;

    const int2 size = get_image_dim(input);
    
    int2 coord = (int2)(get_global_id(0),get_global_id(1));
 
    if(coord.x >= 0 && coord.x < size.x && coord.y >= 0 && coord.y < size.y){
      
        float3 rgb1 = read_imagef(input,sampler,coord).xyz * 255.0f;
        float3 rgb2 = read_imagef(input2,sampler,coord).xyz * 255.0f;
        
        int3 rgbi1 = convert_int3(rgb1);
        int3 rgbi2 = convert_int3(rgb2);
        
        int3 rgb = (rgbi1 ^ rgbi2);
      
        write_imagef(output,coord,(float4)(convert_float3(rgb) / 255.0f,1.0f));
    }
}

__kernel void or_filter(__read_only image2d_t input,__read_only image2d_t input2,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP;

    const int2 size = get_image_dim(input);
    
    int2 coord = (int2)(get_global_id(0),get_global_id(1));
 
    if(coord.x >= 0 && coord.x < size.x && coord.y >= 0 && coord.y < size.y){
      
        float3 rgb1 = read_imagef(input,sampler,coord).xyz * 255.0f;
        float3 rgb2 = read_imagef(input2,sampler,coord).xyz * 255.0f;
        
        int3 rgbi1 = convert_int3(rgb1);
        int3 rgbi2 = convert_int3(rgb2);
        
        int3 rgb = (rgbi1 | rgbi2);
      
        write_imagef(output,coord,(float4)(convert_float3(rgb) / 255.0f,1.0f));
    }
}

__kernel void and_filter(__read_only image2d_t input,__read_only image2d_t input2,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP;

    const int2 size = get_image_dim(input);
    
    int2 coord = (int2)(get_global_id(0),get_global_id(1));
 
    if(coord.x >= 0 && coord.x < size.x && coord.y >= 0 && coord.y < size.y){
      
        float3 rgb1 = read_imagef(input,sampler,coord).xyz * 255.0f;
        float3 rgb2 = read_imagef(input2,sampler,coord).xyz * 255.0f;
        
        int3 rgbi1 = convert_int3(rgb1);
        int3 rgbi2 = convert_int3(rgb2);
        
        int3 rgb = (rgbi1 & rgbi2);
      
        write_imagef(output,coord,(float4)(convert_float3(rgb) / 255.0f,1.0f));
    }
}

__kernel void multi_filter(__read_only image2d_t input,__read_only image2d_t input2,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP;

    const int2 size = get_image_dim(input);
    
    int2 coord = (int2)(get_global_id(0),get_global_id(1));
 
    if(coord.x >= 0 && coord.x < size.x && coord.y >= 0 && coord.y < size.y){
      
        float3 rgb1 = read_imagef(input,sampler,coord).xyz;
        float3 rgb2 = read_imagef(input2,sampler,coord).xyz;
 
        float3 rgb = (rgb1 * rgb2);
      
        write_imagef(output,coord,(float4)(rgb,1.0f));
    }
}

__kernel void div_filter(__read_only image2d_t input,__read_only image2d_t input2,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP;

    const int2 size = get_image_dim(input);
    
    int2 coord = (int2)(get_global_id(0),get_global_id(1));
 
    if(coord.x >= 0 && coord.x < size.x && coord.y >= 0 && coord.y < size.y){
      
        float3 rgb1 = read_imagef(input,sampler,coord).xyz;
        float3 rgb2 = read_imagef(input2,sampler,coord).xyz;
   
        float3 rgb = (rgb1 / rgb2);
      
        write_imagef(output,coord,(float4)(rgb,1.0f));
    }
}

__kernel void mix_filter(__read_only image2d_t input,__read_only image2d_t input2,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP;

    const int2 size = get_image_dim(input);
    
    int2 coord = (int2)(get_global_id(0),get_global_id(1));
 
    if(coord.x >= 0 && coord.x < size.x && coord.y >= 0 && coord.y < size.y){
        float reflection = 0.5f;
        float3 rgb1 = read_imagef(input,sampler,coord).xyz;
        float3 rgb2 = read_imagef(input2,sampler,coord).xyz;
   
        float3 rgb = mix(rgb1,rgb2,reflection);
      
        write_imagef(output,coord,(float4)(rgb,1.0f));
    }
}

__kernel void emboss_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP;

    const int2 size = get_image_dim(input);
    int width = size.x;
    int height = size.y;
    int2 coord = (int2)(get_global_id(0),get_global_id(1));
     
    const int maskrows = 3 / 2;
    const int maskcols = 3 / 2;
    float emboss_mask[9] = {2,0,0,0,-1,0,0,0,-1};
   float3 color = (float3)(0,0,0);
   int idx = 0;
  
    for(int y = -maskrows;y <= maskrows;++y){
      for(int x = -maskcols; x <= maskcols;++x){
        float3 srcColor = read_imagef(input,sampler,(int2)(x + coord.x,y + coord.y)).xyz;
        color += srcColor * emboss_mask[idx];
        idx++;
      }
    }
    
  //  color = color / 3.0f;
  //  color.x = color.y = color.z = (color.x + color.y + color.z) / 3.0f;
 
   write_imagef(output,coord,(float4)(color,1.0f));
    
}

__kernel void mask_blur_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_LINEAR |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);
    int width = size.x;
    int height = size.y;
    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    float2 coordf = convert_float2(coord);
    float3 rgb = read_imagef(input,sampler,coord).xyz;
    int radius = 5;
    float3 color;
    float2 mask_coord;
    int d = (radius * 2 + 1);
    int dsize = d * d;
    
    float mask_kernel [25] = {
        0,0,1,0,0,
        0,1,3,1,0,
        1,3,7,3,1,
        0,1,3,1,0,
        0,0,1,0,0
    };
   
    const int maskrows = radius / 2;
    const int maskcols = radius / 2;
    int idx = 0;
    if(coord.x >= radius && coord.x < width - radius && coord.y >= radius && coord.y < height){
      for(int y = -maskrows;y <= maskrows;++y){
          for(int x = -maskcols; x <= maskcols;++x){
            float3 srcColor = read_imagef(input,sampler,(int2)(x + coord.x,y + coord.y)).xyz * mask_kernel[idx];
            color += srcColor;
            idx++;
          }
        }
    
       color = color / dsize;
    
        write_imagef(output,coord,(float4)(color,0.5f));
    }else{
        write_imagef(output,coord,(float4)(rgb,1.0f));
    }
}


__kernel void color_overlay_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_LINEAR |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);
    int width = size.x;
    int height = size.y;
    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    
    float4 overlay_color = (float4)(0.0f,0.0f,0.5f,0.5f);
    
    
    float4 src_color = read_imagef(input,sampler,coord);
    
    float4 dst_color; //= mix(src_color,overlay_color,1.0f);
   
    dst_color = (float4)(mix(src_color.xyz / max(src_color.w,0.00390625f),overlay_color.xyz / max(overlay_color.w,0.00390625f),overlay_color.w) *  src_color.w,src_color.w);

   
    write_imagef(output,coord,dst_color);
}


__kernel void brighten_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_LINEAR |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);
    int width = size.x;
    int height = size.y;
    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    
    float factor = -0.2f;
    
    float4 color = (float4)(clamp(read_imagef(input,sampler,coord).xyz + factor,0.0f,1.0f),1.0f);
    
    write_imagef(output,coord,color);
}

__kernel void alpha_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_LINEAR |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);
    int width = size.x;
    int height = size.y;
    int2 coord = (int2)(get_global_id(0),get_global_id(1));

    float factor = -0.5f;

    float4 color = read_imagef(input,sampler,coord);
    color.w = clamp(factor,0.0f,1.0f);
    write_imagef(output,coord,color);
}

__kernel void saturate_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_LINEAR |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);
    int width = size.x;
    int height = size.y;
    int2 coord = (int2)(get_global_id(0),get_global_id(1));

    float factor = -0.5f;
    

    float4 color = read_imagef(input,sampler,coord);
    float gray = (color.x + color.y + color.z) / 3.0f;
    color.x = gray + factor * (color.x - gray);
    color.y = gray + factor * (color.y - gray);
    color.z = gray + factor * (color.z - gray);
    write_imagef(output,coord,color);
}

__kernel void sepia_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_LINEAR |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);
    int width = size.x;
    int height = size.y;
    int2 coord = (int2)(get_global_id(0),get_global_id(1));

    float4 rgb = read_imagef(input,sampler,coord);
    
    float r = rgb.x,g = rgb.y,b = rgb.z;
    
    rgb.x = (r * 0.393f) + (g * 0.769f) + (b * 0.189f);
    rgb.y = (r * 0.349f) + (g * 0.686f) + (b * 0.168f);
    rgb.z = (r * 0.272f) + (g * 0.534f) + (b * 0.131f);
    
    
    write_imagef(output,coord,rgb);
}

__kernel void adjust_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_LINEAR |
                              CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);
    int width = size.x;
    int height = size.y;
    int2 coord = (int2)(get_global_id(0),get_global_id(1));

    float3 rgb = read_imagef(input,sampler,coord).xyz;
    
    float r = 0.0f;
    float g = 0.2f;
    float b = 0.0f;
    
    rgb.x += r;
    rgb.y += g;
    rgb.z += b;
    
    
    write_imagef(output,coord,(float4)(rgb,1.0f));
}

void filter2d_internal(__read_only image2d_t input,
                       __write_only image2d_t output,
                       const int maskWidth,
                       const int maskHeight,
                        float * mask,int compute_aver){

   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));

   const int maskrows = maskWidth / 2;
   const int maskcols = maskHeight / 2;

   float4 color = (float4)(0,0,0,1.0f);
   int idx = 0;

   for(int y = -maskrows;y <= maskrows;++y){
      for(int x = -maskcols; x <= maskcols;++x){
        float4 srcColor = read_imagef(input,sampler,(int2)(x + coord.x,y + coord.y));
          color.xyz += srcColor.xyz * mask[idx];
        idx++;
      }
   }
   if(compute_aver){
     color.xyz = color.xyz / (maskWidth * maskHeight);
   }
  write_imagef(output,coord,color);
}

__kernel void desaturate_luminance_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    
    float color_matrix[] = {
        0.2764723f, 0.9297080f, 0.0938197f, 0, -37.1f,
	    0.2764723f, 0.9297080f, 0.0938197f, 0, -37.1f,
		0.2764723f, 0.9297080f, 0.0938197f, 0, -37.1f,
		0.0f, 0.0f, 0.0f, 1.0f, 0.0f
    }; 
 //   int maskWidth = 5;
//    int maskHeight = 5;
    
   // filter2d_internal(input,output,maskWidth,maskHeight,color_matrix,0);
   color_matrix_4x5_internal(input,output,color_matrix);
}

__kernel void brownie_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    
    float color_matrix[] = {
        0.5997023498159715f,0.34553243048391263f,-0.2708298674538042f,0,47.43192855600873f,
	   -0.037703249837783157f,0.8609577587992641f,0.15059552388459913f,0,-36.96841498319127f,
		0.24113635128153335f,-0.07441037908422492f,0.44972182064877153f,0,-7.562075277591283f,
	   0.0f,0.0f,0.0f,1.0f,0.0f

    }; 
//    int maskWidth = 4;
//    int maskHeight = 4;
    
//    filter2d_internal(input,output,maskWidth,maskHeight,color_matrix,1);//ok
    //filter2d_internal(input,output,maskWidth,maskHeight,color_matrix,1);//ok
     color_matrix_4x5_internal(input,output,color_matrix);
}

__kernel void sepia2_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    
    float color_matrix[] = {
       0.393f, 0.7689999f, 0.18899999f,0.0f,0.0f,
	   0.349f, 0.6859999f, 0.16799999f,0.0f,0.0f,
	   0.272f, 0.5339999f, 0.13099999f,0.0f,0.0f,
	   0.0f,0.0f,0.0f,1.0f,0.0f
    }; 
    //int maskWidth = 3;
    //int maskHeight = 3;
    color_matrix_4x5_internal(input,output,color_matrix);
   // filter2d_internal(input,output,maskWidth,maskHeight,color_matrix,0);//ok
}

__kernel void hue_filter(__read_only image2d_t input,
                         __write_only image2d_t output){
    float angle = 45.0f;
    float rotation = angle / 180.0f * PI_F;
    
    float lumR = 0.213f;
    float lumG = 0.715f;
    float lumB = 0.072f;
    
    
    
    float color_matrix[] = {
       lumR+cos(rotation)*(1-lumR)+sin(rotation)*(-lumR),lumG+cos(rotation)*(-lumG)+sin(rotation)*(-lumG),lumB+cos(rotation)*(-lumB)+sin(rotation)*(1-lumB),0.0f,0.0f,
	   lumR+cos(rotation)*(-lumR)+sin(rotation)*(0.143f),lumG+cos(rotation)*(1-lumG)+sin(rotation)*(0.140f),lumB+cos(rotation)*(-lumB)+sin(rotation)*(-0.283f),0.0f,0.0f,
	   lumR+cos(rotation)*(-lumR)+sin(rotation)*(-(1-lumR)),lumG+cos(rotation)*(-lumG)+sin(rotation)*(lumG),lumB+cos(rotation)*(1-lumB)+sin(rotation)*(lumB),0.0f,0.0f,
       0.0f,0.0f,0.0f,1.0f,0.0f
	  
    }; 
    //int maskWidth = 3;
    //int maskHeight = 3;
    
    //filter2d_internal(input,output,maskWidth,maskHeight,color_matrix,0);//ok
     color_matrix_4x5_internal(input,output,color_matrix);
}

__kernel void hue2_filter(__read_only image2d_t input,
                         __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   const float4 kRGBToYPrime = (float4) (0.299, 0.587, 0.114, 0.0);
   const float4 kRGBToI     = (float4) (0.595716, -0.274453, -0.321263, 0.0);    
   const float4 kRGBToQ     = (float4) (0.211456, -0.522591, 0.31135, 0.0);
 
   const float4 kYIQToR   = (float4) (1.0, 0.9563, 0.6210, 0.0);
   const float4 kYIQToG   = (float4) (1.0, -0.2721, -0.6474, 0.0);
   const float4 kYIQToB   = (float4) (1.0, -1.1070, 1.7046, 0.0);
   
   float4 color = read_imagef(input,sampler,coord);
   
   float YPrime  = dot (color, kRGBToYPrime);
   float I      = dot (color, kRGBToI);
   float Q      = dot (color, kRGBToQ);
   
   float hue     = atan2 (Q, I);
   float chroma  = sqrt (I * I + Q * Q);
   
   float hueAdjust = 0.0f;
   
   hue += (-hueAdjust);
   
   Q = chroma * sin (hue);
   I = chroma * cos (hue);
     
     // Convert back to RGB
   float4 yIQ = (float4) (YPrime, I, Q, 0.0);
   color.x = dot (yIQ, kYIQToR);
   color.y = dot (yIQ, kYIQToG);
   color.z = dot (yIQ, kYIQToB);
   
   write_imagef(output,coord,color);
}

__kernel void vintage_pinhole_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    
    float color_matrix[] = {
            0.6279345635605994f,0.3202183420819367f,-0.03965408211312453f,0.0f,9.3651285835294123f,
			0.02578397704808868f,0.6441188644374771f,0.03259127616149294f,0.0f,7.462829176470591f,
			0.0466055556782719f,-0.0851232987247891f,0.5241648018700465f,0.0f,5.159190588235296f,
			0.0f,0.0f,0.0f,1.0f,0.0f
    }; 
    //int maskWidth = 5;
    //int maskHeight = 3;
    
    //filter2d_internal(input,output,maskWidth,maskHeight,color_matrix,1);//ok
         color_matrix_4x5_internal(input,output,color_matrix);

}

__kernel void techni_color_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    
    float color_matrix[20] = {
            1.9125277891456083f,-0.8545344976951645f,-0.09155508482755585f,0,11.793603434377337f,
			-0.3087833385928097f,1.7658908555458428f,-0.10601743074722245f,0,-70.35205161461398f,
			-0.231103377548616f,-0.7501899197440212f,1.847597816108189f,0,30.950940869491138f,
			0,0,0,1,0
    }; 
    //int maskWidth = 5;
    //int maskHeight = 4;
    
   // filter2d_internal(input,output,maskWidth,maskHeight,color_matrix,1);//ok,0 also ok
    color_matrix_4x5_internal(input,output,color_matrix);
}

__kernel void kodachrome_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    
    float color_matrix[20] = {
           1.1285582396593525f,-0.3967382283601348f,-0.03992559172921793f,0,63.72958762196502f,
			-0.16404339962244616f,1.0835251566291304f,-0.05498805115633132f,0,24.732407896706203f,
			-0.16786010706155763f,-0.5603416277695248f,1.6014850761964943f,0,35.62982807460946f,
			0,0,0,1,0
    }; 
   // int maskWidth = 5;
    //int maskHeight = 4;
    
   // filter2d_internal(input,output,maskWidth,maskHeight,color_matrix,1);//both
   color_matrix_4x5_internal(input,output,color_matrix);
}

__kernel void polariod_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    
    float color_matrix[20] = {
           1.438f,-0.062f,-0.062f,0,0,
			-0.122f,1.378f,-0.122f,0,0,
			-0.016f,-0.016f,1.483f,0,0,
			0,0,0,1,0
    }; 
   // int maskWidth = 5;
    //int maskHeight = 4;
    
    //filter2d_internal(input,output,maskWidth,maskHeight,color_matrix,0);//ok
    color_matrix_4x5_internal(input,output,color_matrix);
}

__kernel void prewitt_horizonta_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    float divider = 3.0f;
    float color_matrix[9] = {
           1/divider, 1/divider, 1/divider,
           0, 0, 0,
           -1/divider, -1/divider, -1/divider
    }; 
    int maskWidth = 3;
    int maskHeight = 3;
    
    filter2d_internal(input,output,maskWidth,maskHeight,color_matrix,0);//ok
}

__kernel void prewitt_vertical_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    float divider = 3.0f;
    float color_matrix[9] = {
          -1/divider, 0, 1/divider,
          -1/divider, 0, 1/divider,
          -1/divider, 0, 1/divider
    }; 
    int maskWidth = 3;
    int maskHeight = 3;
    
    filter2d_internal(input,output,maskWidth,maskHeight,color_matrix,0);//ok
}

__kernel void posterize2_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    
    const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   int level = 255;
   float numAreas = 255.0f / level;
   float numValues = 255.0f / (level - 1);
   
   float3 rgb = clamp((((read_imagef(input,sampler,coord).xyz / numAreas) * numValues) ),0,1.0f);
   
   write_imagef(output,coord,(float4)(rgb,1.0f));
}

__kernel void posterize3_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    
    const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   int level = 1;
   int colors = 40;
   float numAreas = 255.0f / level;
   float numValues = 255.0f / (level - 1);
   
   local float levels[256];
   
   for(int i = 0;i < 256;i++){
       if(i < colors * level){
           levels[i] = colors * (level - 1) / 255.0f;
       }else {
           levels[i] = colors * level / 255.0f;
           ++level;
       }
   }
   
   float3 src_rgb = read_imagef(input,sampler,coord).xyz;
   
   float3 dst_rgb;
   dst_rgb.x = levels[convert_int(src_rgb.x * 255)];
   dst_rgb.y = levels[convert_int(src_rgb.y * 255)];
   dst_rgb.z = levels[convert_int(src_rgb.z * 255)];
   
   write_imagef(output,coord,(float4)(dst_rgb,1.0f));
}

__kernel void solarize_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    
    const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float amount = 0.25f;
   
   float4 rgb = read_imagef(input,sampler,coord);
   
   if(rgb.x > amount) rgb.x = 1.0 - rgb.x;
   if(rgb.y > amount) rgb.y = 1.0 - rgb.y;
   if(rgb.z > amount) rgb.z = 1.0 - rgb.z;
   
   write_imagef(output,coord,rgb);

}

__kernel void split_red_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    
    const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float amount = 0.25f;
   
   float4 rgb = read_imagef(input,sampler,coord);
   
   rgb.yz = 0;
   
   write_imagef(output,coord,rgb);

}

__kernel void split_green_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    
    const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float amount = 0.25f;
   
   float4 rgb = read_imagef(input,sampler,coord);
   
   rgb.xz = 0;
   
   write_imagef(output,coord,rgb);

}

__kernel void split_blue_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    
    const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float amount = 0.25f;
   
   float4 rgb = read_imagef(input,sampler,coord);
   
   rgb.xy = 0;
   
   write_imagef(output,coord,rgb);

}

__kernel void median2_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   float color_mask[25] = {1, 0, 0, 0, 1, 
                         0, 1, 0, 1, 0, 
                         1, 1, 1, 1, 1, 
                         0, 1, 0, 1, 0, 
                         1, 0, 0, 0, 1 }; 
   
   int maskWidth = 5;
   int maskHeight = 5;
 
   filter2d_internal(input,output,maskWidth,maskHeight,color_mask,1);//ok
}

__kernel void pencil_sketch_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
                              
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float3 src_rgb = read_imagef(input,sampler,coord).xyz;
   float3 invert_rgb = 1.0f - src_rgb;
   
   float3 mask [9] = {0.1f, 0.2f, 0.1f, 0.2f, 0.1f, 0.2f, 0.1f, 0.2f, 0.1f};
   
   const int maskrows = 3 / 2;
   const int maskcols = 3 / 2;

   float4 color = (float4)(0,0,0,1.0f);
   int idx = 0;

   for(int y = -maskrows;y <= maskrows;++y){
      for(int x = -maskcols; x <= maskcols;++x){
        float4 srcColor = read_imagef(input,sampler,(int2)(x + coord.x,y + coord.y));
          color.xyz += srcColor.xyz * mask[idx];
        idx++;
      }
   }
   color.xyz = color.xyz / 9;
   
   float3 dst_rgb = clamp(invert_rgb + color.xyz,0,1);
  // dst_rgb.x = dst_rgb.y = dst_rgb.z = (dst_rgb.x + dst_rgb.y + dst_rgb.z) / 3;
   write_imagef(output,coord,(float4)(dst_rgb,1.0f));
}

__kernel void emboss2_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   float color_mask[9] = {-2,-1,0,
                          -1,1,1,
                          0,1,2}; 
   
   int maskWidth = 3;
   int maskHeight = 3;
 
   filter2d_internal(input,output,maskWidth,maskHeight,color_mask,1);//both ok
}

__kernel void luminosity_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   float color_mask[9] = {0,0,0,
                          -1,1,-1,
                          0,0,0}; 
   
   int maskWidth = 3;
   int maskHeight = 3;
 
   filter2d_internal(input,output,maskWidth,maskHeight,color_mask,0);
}

float bezier_curve_pow3(float x, float p0, float p1, float p2, float p3){

   //基于三次贝塞尔曲线 
   return p0 * pow((1 - x), 3) + 3 * p1 * x * pow((1 - x), 2) + 3 * p2 * x * x * (1 - x) + p3 * pow(x, 3);
}

float3 calDark(float x, float y, float3 p,float middleX,float middleY,float startDistance,float maxDistance,float lastLevel){
  //计算距中心点距离
  float dist = length((float2)(x - middleX,y - middleY));//distance([x, y], [middleX, middleY]);
  float currBilv = (dist - startDistance) / (maxDistance - startDistance);
  if(currBilv < 0) currBilv = 0;
   //应该增加暗度
    return  bezier_curve_pow3(currBilv, 0, 0.02, 0.3, 1) * p * lastLevel / 255;
  }

__kernel void dark_corner_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   int R = 3;//from 1~10
   int lastLevel = 20;//0 ~ 255
   
   int width = size.x;
   int height = size.y;
   
   int lenght = R * 2 + 1;
   
   float middleX = width * 2 / 3;
   float middleY = height * 1/ 2;
   
   float maxDistance = distance(middleX ,middleY);
                //开始产生暗角的距离
   float startDistance = maxDistance * (1 - R / 10);
   
   float4 color = read_imagef(input,sampler,coord);
   
   float3 darkness = calDark(coord.x,coord.y,color.xyz,middleX,middleY,startDistance,maxDistance,lastLevel);
   
   color.xyz -= darkness;
   
   write_imagef(output,coord,color);
   
}

__kernel void oil_paint2_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   int R = 16;
   int xLength = 2 * R + 1;
   
   float4 color = read_imagef(input,sampler,coord) * 255;
   float gray  = (color.x + color.y + color.z) / 3;
   
   float every = (gray / R) * R;
   
   color.x = color.y = color.z = every;
   
   write_imagef(output,coord,color / 255);
}

float luminance(float4 color){
    const float3 w = (float3)(0.2125, 0.7154, 0.0721);
    return dot(color.xyz, w);
}

__kernel void toon_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 rgb = read_imagef(input,sampler,coord);
   
    float topLeft = luminance(read_imagef(input,sampler,(int2)(coord.x - 1,coord.y - 1)));
    // top
    float top = luminance(read_imagef(input,sampler,(int2)(coord.x,coord.y - 1)));
    // top right
    float topRight = luminance(read_imagef(input,sampler,(int2)(coord.x + 1,coord.y - 1)));
    // left
    float left = luminance(read_imagef(input,sampler,(int2)(coord.x - 1,coord.y)));
    // center
    float center = luminance(read_imagef(input,sampler,(int2)(coord.x,coord.y)));
    // right
    float right = luminance(read_imagef(input,sampler,(int2)(coord.x + 1,coord.y)));
    // bottom left
    float bottomLeft = luminance(read_imagef(input,sampler,(int2)(coord.x - 1,coord.y + 1)));
    // bottom
    float bottom = luminance(read_imagef(input,sampler,(int2)(coord.x,coord.y + 1)));
    // bottom right
    float bottomRight = luminance(read_imagef(input,sampler,(int2)(coord.x + 1,coord.y + 1)));
    
    
    float h = -topLeft-2.0*top-topRight+bottomLeft+2.0*bottom+bottomRight;
    float v = -bottomLeft-2.0*left-topLeft+bottomRight+2.0*right+topRight;

    float mag = length((float2)(h, v));
    float threshold = 0.2f;
    float quantizationLevels = 10;    
    float3 posterizedImageColor = floor((rgb.xyz * quantizationLevels) + 0.5f) / quantizationLevels;
    float thresholdTest = 1.0f - step(threshold, mag);
   
    rgb.xyz = rgb.xyz * thresholdTest;
    
    write_imagef(output,coord,rgb);
}

__kernel void directional_nonmaximum_suppression_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);
   int width = size.x;
   int height = size.y;
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float upperThreshold = 0.80;
   float lowerThreshold = 0.25;
   
   float4 currentGradientAndDirection = read_imagef(input,sampler,(coord));
   float2 vUv = (float2)(3,3);
   float2 gradientDirection = ((currentGradientAndDirection.yz * 2.0f) - 1.0f) * (float2)(1.0f/width, 1.0f/height);
   float firstSampledGradientMagnitude = read_imagef(input,sampler,convert_int2(gradientDirection + vUv)).x;
   float secondSampledGradientMagnitude = read_imagef(input,sampler,convert_int2(vUv - gradientDirection)).x;
   float multiplier = step(firstSampledGradientMagnitude, currentGradientAndDirection.x);
   multiplier = multiplier * step(secondSampledGradientMagnitude, currentGradientAndDirection.x);
   float thresholdCompliance = smoothstep(lowerThreshold, upperThreshold, currentGradientAndDirection.x);
   multiplier = multiplier * thresholdCompliance;
   
   write_imagef(output,coord,(float4)(multiplier,multiplier,multiplier,1.0f));
}

__kernel void box_blur_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float blurSize =256.0f; 
   float imageWidth = size.x;
   
   float4 sum;
   
   for(int i = 0; i < 40; i++){
      sum += read_imagef(input,sampler, coord + (int2)(0, convert_int((i-20) * blurSize / imageWidth)));
   }
  /*
   for(int i = 0; i < 40; i++){
      sum += read_imagef(input,sampler, coord + (int2)(convert_int((i-20) * blurSize / size.y),0));
   }
   */
   sum = sum / 40;
   
   write_imagef(output,coord,sum);
}


__kernel void color_matrix_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 color_matrix = (float4)(1,0.5,0.72,1);
   
   float intensity = 0.8f;
   
   float4 color = read_imagef(input,sampler,coord);
   
   float4 dst_color = color * color_matrix;
   
   dst_color = intensity * dst_color + ( (1.0f-intensity) * color );
   
   write_imagef(output,coord,dst_color);
   
}

__kernel void hexagonal_blur_1(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float blurSize = 256.0f;
   float imageWidth = size.x;
   float imageHeight = size.y;
   
   float2 offset = (float2)(blurSize/imageWidth, blurSize/imageHeight);
   
   float4 color;
   
   for(int i = 0; i < 30; i++){
      color += 1.0f/30.0f * read_imagef(input,sampler, coord + convert_int2((float2)(0.0, offset.y * i)));
   }
   
   write_imagef(output,coord,color);
}

__kernel void hexagonal_blur_2(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float blurSize = 256.0f;
   float imageWidth = size.x;
   float imageHeight = size.y;
   
   float2 offset = (float2)(blurSize/imageWidth, blurSize/imageHeight);
   
   float4 color;
   
   for(int i = 0; i < 30; i++){
      color += 1.0f/30.0f * read_imagef(input,sampler, coord + convert_int2((float2)( offset.x * i, offset.y * i)));
   }
   
   write_imagef(output,coord,color);
}

__kernel void hexagonal_blur_3(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float blurSize = 256.0f;
   float imageWidth = size.x;
   float imageHeight = size.y;
   
   float2 offset = (float2)(blurSize/imageWidth, blurSize/imageHeight);
   
   float4 color2,color1,color3;
   
   for(int i = 0; i < 30; i++){
      color1 += 1.0f/30.0f * read_imagef(input,sampler, coord - convert_int2((float2)( offset.x * i, offset.y * i)));
   }
   
   for(int i = 0; i < 30; i++){
      color2 += 1.0f/30.0f * read_imagef(input,sampler, coord + convert_int2((float2)( offset.x * i, -offset.y * i)));
   }
   
   for(int i = 0; i < 30; i++){
      color3 += 1.0f/30.0f * read_imagef(input,sampler, coord + convert_int2((float2)( offset.x * i, -offset.y * i)));
   }
   
   
   write_imagef(output,coord,(color1 + color2 + color3) / 3);
}

__kernel void sketch_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 rgb = read_imagef(input,sampler,coord);
   
    float topLeft = luminance(read_imagef(input,sampler,(int2)(coord.x - 1,coord.y - 1)));
    // top
    float top = luminance(read_imagef(input,sampler,(int2)(coord.x,coord.y - 1)));
    // top right
    float topRight = luminance(read_imagef(input,sampler,(int2)(coord.x + 1,coord.y - 1)));
    // left
    float left = luminance(read_imagef(input,sampler,(int2)(coord.x - 1,coord.y)));
    // center
    float center = luminance(read_imagef(input,sampler,(int2)(coord.x,coord.y)));
    // right
    float right = luminance(read_imagef(input,sampler,(int2)(coord.x + 1,coord.y)));
    // bottom left
    float bottomLeft = luminance(read_imagef(input,sampler,(int2)(coord.x - 1,coord.y + 1)));
    // bottom
    float bottom = luminance(read_imagef(input,sampler,(int2)(coord.x,coord.y + 1)));
    // bottom right
    float bottomRight = luminance(read_imagef(input,sampler,(int2)(coord.x + 1,coord.y + 1)));
    
    
 
    float h = -topLeft-2.0f*top-topRight+bottomLeft+2.0f*bottom+bottomRight;
	float v = -bottomLeft-2.0f*left-topLeft+bottomRight+2.0f*right+topRight;

	float mag = 1.0f - length((float2)(h, v));

    rgb.x = rgb.y = rgb.z = mag;
    rgb.w = 1.0f;
    
    write_imagef(output,coord,rgb);

}

__kernel void autumn_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   
   color.x =  color.x + color.y * 1.25f - color.z * 1.25f;
   
   write_imagef(output,coord,color);
}

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

__kernel void denoise_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 center_color = read_imagef(input,sampler,coord);
   
   float exponent = 0.5f;
   float strength = 1.0f;
   float total = 0.0f;
   float4 color;
   for (int x = -4; x <= 4; x += 1) {
     for (int y = -4; y <= 4; y += 1) {
           float4 sample = read_imagef(input,sampler, (coord + (int2)(x, y)));
           float dot_res = dot(sample.xyz - center_color.xyz, (float3)(0.25,0.25,0.25));
           float weight = 1.0f - ((dot_res > 0)? dot_res : -dot_res);
           weight = pow(weight, exponent);
           color += sample * weight;
           total += weight;
     }
   }
   
  write_imagef(output,coord,color / total);
}

__kernel void vibrance_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   
   float amount = 0.5f;
  
   float average = (color.x + color.y + color.z) / 3.0f;
   float mx = max(color.x, max(color.y, color.z));
   float amt = (mx - average) * (-amount * 3.0f);
   color.xyz = mix(color.xyz, (float3)(mx,mx,mx), amt);
   
  write_imagef(output,coord,color);
}



__kernel void brazil_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 src_rgba = read_imagef(input,sampler,coord);
   
   float xBlockSize = 0.01*0.1;
   float yBlockSize = xBlockSize * size.x / size.y;  // mutiply ratio
   float xCoord = (floor((coord.x-0.5)/xBlockSize)+0.5) * xBlockSize+0.5;
   float yCoord = (floor((coord.y-0.5)/yBlockSize)+0.5) * yBlockSize+0.5;
   float arg = 0.5f;
   
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
   
  write_imagef(output,coord,color);
}

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

__kernel void croquis_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   
   float arg = 1.0f;
  
   float dx = 1.0f / size.x;
   float dy = 1.0f / size.y;
   float c  = -1.0f / 8.0f;
   
    
   float r = ((read_imagef(input,sampler, coord + (float2)(-dx, -dy)).x
          +    read_imagef(input,sampler, coord + (float2)(0.0, -dy)).x
          +    read_imagef(input,sampler, coord + (float2)( dx, -dy)).x
          +    read_imagef(input,sampler, coord + (float2)(-dx, 0.0)).x
          +    read_imagef(input,sampler, coord + (float2)( dx, 0.0)).x
          +    read_imagef(input,sampler, coord + (float2)(-dx,  dy)).x
          +    read_imagef(input,sampler, coord + (float2)(0.0,  dy)).x
          +    read_imagef(input,sampler, coord + (float2)( dx,  dy)).x) * c
          +    read_imagef(input,sampler, coord).x) * -2.0f; 
   
    float g = ((read_imagef(input,sampler, coord + (float2)(-dx, -dy)).y
          +   read_imagef(input,sampler, coord+ (float2)(0.0, -dy)).y
          +   read_imagef(input,sampler, coord + (float2)( dx, -dy)).y
          +   read_imagef(input,sampler, coord + (float2)(-dx, 0.0)).y
          +   read_imagef(input,sampler, coord + (float2)( dx, 0.0)).y
          +   read_imagef(input,sampler, coord + (float2)(-dx,  dy)).y
          +   read_imagef(input,sampler, coord + (float2)(0.0,  dy)).y
          +   read_imagef(input,sampler, coord + (float2)( dx,  dy)).y) * c
          +   read_imagef(input,sampler, coord).y) * -24.0;
          
    float b = ((read_imagef(input,sampler, coord + (float2)(-dx, -dy)).z
          +   read_imagef(input,sampler, coord + (float2)(0.0, -dy)).z
          +   read_imagef(input,sampler, coord + (float2)( dx, -dy)).z
          +   read_imagef(input,sampler, coord + (float2)(-dx, 0.0)).z
          +   read_imagef(input,sampler, coord + (float2)( dx, 0.0)).z
          +   read_imagef(input,sampler, coord + (float2)(-dx,  dy)).z
          +   read_imagef(input,sampler, coord + (float2)(0.0,  dy)).z
          +   read_imagef(input,sampler, coord + (float2)( dx,  dy)).z) * c
          +   read_imagef(input,sampler, coord).z) * -24.0;
          
   float brightness = (r * 0.3f + g * 0.59f + b * 0.11f);
   brightness = 1.0f - brightness;
   if (brightness < 0.0f) brightness = 0.0f;
   if (brightness > 1.0f) brightness = 1.0f;
   r = g = b = brightness;  
   
   float3 rgb = (float3)(r, g, b);
   float4 dst_color = (float4)(rgb - (1.0f - arg),1.0f);
   
   write_imagef(output,convert_int2(coord),dst_color);
 
}

__kernel void hardrockcafe_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   
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


  float v = (vertical.x > 0 ? vertical.x : -vertical.x);
  float h = (horizontal.x > 0 ? horizontal.x : -horizontal.x);
  float m = ( v + h) / 4.0f;
  
  float arg = 0.8f;
  
  float4 dst_color = (float4)(mix(color.xyz, (float3)(v, h, m), arg), color.w);
  
  write_imagef(output,convert_int2(coord),dst_color);
   
}

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



__kernel void lego_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,convert_int2(coord));
   
   float arg = 0.75f;
   
   if(arg > 0.0f){
    float xBlockSize = arg * 0.1f;
    float yBlockSize = xBlockSize * size.x / size.y;  // mutiply ratio
    float xCoord = (floor((coord.x - 0.5) / xBlockSize) + 0.5f) * xBlockSize + 0.5f;
    float yCoord = (floor((coord.y - 0.5)/ yBlockSize) + 0.5f) * yBlockSize + 0.5f;
    float4 rgba = read_imagef(input,sampler, convert_int2((float2)(xCoord, yCoord)));
    float sum = (rgba.x + rgba.y + rgba.z) / 3.0f;
    float3 one = (float3)(255.0f, 255.0f, 255.0f) / 255.0f;
    float3 two = (float3)(242.0f, 252.0f, 0.0f) / 255.0f;
    float3 three = (float3)(0.0f, 140.0f, 0.0f) / 255.0f;
    float3 four = (float3)(48.0f, 19.0f, 6.0f) / 255.0f;
    float3 five = (float3)(0.0f, 0.0f, 0.0f) / 255.0f;
/*
黒　緑　黄色　白
1   255 255 255
2   242 252 0
3   0   140 0
4   48  19  6
5   0   0   0
*/
    if      (sum < 0.05){ 
      rgba = (float4)(five,   1.0f);
    }
    else if (sum < 0.65) {
      rgba = (float4)(four,   1.0f);
    }
    else if (sum < 1.40) {
      rgba = (float4)(three, 1.0f);
    }
    else if (sum < 2.15) {
     rgba = (float4)(two,  1.0f);
    }
    else{                 
       rgba = (float4)(one,  1.0f);
    }
   // rgba = color;
    write_imagef(output,convert_int2(coord),rgba);
   }else{
       write_imagef(output,convert_int2(coord),color);
   }
   
}

__kernel void monochrome_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   float arg = 0.75f;
   float4 color = read_imagef(input,sampler,coord);
   float y = dot(color.xyz, (float3)(0.299f, 0.587f, 0.114f));
   float4 dst_color = (float4)(mix(color.xyz, (float3)(y), arg), 1.0f);
   
   write_imagef(output,convert_int2(coord),dst_color);
}

__kernel void monoedge_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   float arg = 0.75f;
   
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
          +    read_imagef(input,sampler,coord).x) * -2;

  float g = ((read_imagef(input,sampler,coord + (float2)(-dx, -dy)).y
          +   read_imagef(input,sampler,coord + (float2)(0.0, -dy)).y
          +   read_imagef(input,sampler,coord + (float2)( dx, -dy)).y
          +   read_imagef(input,sampler,coord + (float2)(-dx, 0.0)).y
          +   read_imagef(input,sampler,coord + (float2)( dx, 0.0)).y
          +   read_imagef(input,sampler,coord + (float2)(-dx,  dy)).y
          +   read_imagef(input,sampler,coord + (float2)(0.0,  dy)).y
          +   read_imagef(input,sampler,coord + (float2)( dx,  dy)).y) * c
          +   read_imagef(input,sampler, coord).y) * -24;

  float b = ((read_imagef(input,sampler,coord + (float2)(-dx, -dy)).z
          +   read_imagef(input,sampler,coord + (float2)(0.0, -dy)).z
          +   read_imagef(input,sampler,coord + (float2)( dx, -dy)).z
          +   read_imagef(input,sampler,coord + (float2)(-dx, 0.0)).z
          +   read_imagef(input,sampler,coord + (float2)( dx, 0.0)).z
          +   read_imagef(input,sampler,coord + (float2)(-dx,  dy)).z
          +   read_imagef(input,sampler,coord + (float2)(0.0,  dy)).z
          +   read_imagef(input,sampler,coord + (float2)( dx,  dy)).z) * c
          +   read_imagef(input,sampler, coord).z) * -24.0;

  if (r < 0.0) r = 0.0;
  if (g < 0.0) g = 0.0;
  if (b < 0.0) b = 0.0;
  if (r > 1.0) r = 1.0;
  if (g > 1.0) g = 1.0;
  if (b > 1.0) b = 1.0;
  
  float brightness = (r*0.3 + g*0.59 + b*0.11);
  if (brightness < 0.0) brightness = 0.0;
  if (brightness > 1.0) brightness = 1.0;
  r = g = b = brightness;  

  float3 rgb = (float3)(r, g, b);
  
  
  float4 dst_color = (float4)(rgb - (1.0f - arg), 1.0f);
  
  write_imagef(output,convert_int2(coord),dst_color);
}

__kernel void mosaic_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   float arg = 0.075f;
   float4 color = read_imagef(input,sampler,coord);
   if (arg > 0.0f) {
    float xBlockSize = arg * 0.1f;
    float yBlockSize = xBlockSize * size.x / size.y;  // mutiply ratio
    float xCoord = (floor((coord.x-0.5f)/xBlockSize)+0.5f) * xBlockSize + 0.5f;
    float yCoord = (floor((coord.y-0.5f)/yBlockSize)+0.5f) * yBlockSize + 0.5f;
    float4 rgba = read_imagef(input,sampler,(float2)(xCoord,yCoord));
    write_imagef(output,convert_int2(coord),rgba);
  } else {
    write_imagef(output,convert_int2(coord),color);
  }
   
}


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

__kernel void skin_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);

   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   
   float4 color = read_imagef(input,sampler,coord);
   
   float r = color.x;
   float g = color.y;
   float b = color.z;
   
   float xy = (r - g);
   xy = xy > 0 ? xy : -xy;
   
   if((r <= 45.0f / 255.0f) || 
      (g <= 40.0f / 255.0f) || 
      (b <= 20.0f / 255.0f) ||
      (r <= g) ||
      (r <= b) ||
      ((r - min(g,b)) <= 15.0f / 255.0f) ||
      (xy <= 15.0f / 255.0f)){
      color.x = color.y = color.z = 0;
   }
   write_imagef(output,convert_int2(coord),color);
}

__kernel void vignette_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float amount = 2.0f;
   float2 vUv = (float2)(100,100);
   
   float dist = distance(vUv,(float2)(dim.x / 2,dim.y / 2));
   
   float4 color = read_imagef(input,sampler,coord);
   float size = 256.0f;
   color.xyz *= smoothstep(0.8f,size * 0.799f,dist * (amount + size));
   
   write_imagef(output,convert_int2(coord),color);
   
}

float lum(float3 color){
    return 0.3f * color.x + 0.59f * color.y + 0.11f * color.z;
}

__kernel void color_clip_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   
   float L = lum(color.xyz);
   
   float n = min(min(color.x,color.y),color.z);
   float x = max(max(color.x,color.y),color.z);
   
   if(n < 0.0f){
      color.xyz = L +(((color.xyz - L) * L) / (L - n));
   }
   
   if(x > 1.0f){
      color.xyz = L + (((color.xyz - L) * (1 - L)) / (x - L));
   }
   
   color.xyz += (0.5f - L);
   
   write_imagef(output,convert_int2(coord),color);
   
}

__kernel void bias_clip_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);

   float amount = 0.75f;
   
   color.xyz *= clamp(color.xyz / (( 1.0f / amount - 1.9f) * (0.9f - color.xyz) + 1.0f),0.0f,1.0f);
   
   write_imagef(output,convert_int2(coord),color);
}



__kernel void duo_tone_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   
   float3 dark_color = (float3)(1,1,0);
   float3 light_color = (float3)(0,0,1);
   
   float gray = lum(color.xyz);
   
   float luminance = dot(color.xyz,gray);
   
   color.xyz = clamp(mix(dark_color,light_color,luminance),0.0f,1.0f);
   
   write_imagef(output,coord,color);
   
}

__kernel void opacity_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   
   float opacity_value = 0.5f;
   
   color.w *= opacity_value;
   
   write_imagef(output,coord,color);
   
}

__kernel void tan_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_TRUE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   
   
   float scale = 2.0f;
   float2 new_coord =tan(coord * scale);
   float4 color = read_imagef(input,sampler,new_coord);
   
   write_imagef(output,convert_int2(coord),color);
   
}

float3 mod289_3(float3 x){
    return x - floor(x * (1.0f / 289.0f)) * 289.0f;
}

float2 mod289_2(float2 x){
    return x - floor(x * (1.0f / 289.0f)) * 289.0f;
}

float3 permute(float3 x){
    return mod289_3(((x * 34.0f) + 1.0f) * x);
}

float snoise(float2 v){
    const float4 C = (float4)(0.211324865405187f,  // (3.0-sqrt(3.0))/6.0
                      0.366025403784439f,  // 0.5*(sqrt(3.0)-1.0)
                     -0.577350269189626f,  // -1.0 + 2.0 * C.x
                      0.024390243902439f); // 1.0 / 41.0;
                      
    
    float2 i = floor(v + dot(v,C.yy));
    float2 x0 = v - i + dot(i,C.xx);
    
    float2 i1 ;//= step(x0.y,x0.x);
    
    i1 = (x0.x > x0.y) ? (float2)(1.0f,0.0f) : (float2)(0.0f,1.0f);
    
    float4 x12 = x0.xyxy + C.xxzz;
    
    i = mod289_2(i);
    
    float3 p = permute(permute(i.y + (float3)(0.0f,i1.y,1.0f)) + i.x + (float3)(0.0f,i1.x,1.0f));
    
    float3 m = max(0.5f - (float3)(dot(x0,x0),dot(x12.xy,x12.xy),dot(x12.zw,x12.zw)),(float3)(0.0f,0.0f,0.0f));
    
    m = m * m;
    m = m * m;
    float3 iptr; //0x1.fffffep-1f;
    float3 x = 2.0f * fract(p * C.www,&iptr) - 1.0f;
    float3 h = fabs(x) - 0.5f;

    float3 ox = floor(x + 0.5f);
    float3 a0 = x - ox;
    
    m *= 1.79284291400159f - 0.85373472095314f * ( a0*a0 + h*h ); 
    
    float3 g;
    g.x = a0.x * x0.x + h.x + x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    
    return 130.0f * dot(m,g);
}

__kernel void cipher_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   float noise_scale = 2.0f;
   float noise_offset = 3.0f;
   float4 n = snoise(coord * noise_scale + noise_offset);
   
   write_imagef(output,convert_int2(coord),color * 0.5f + n * 0.25f + 0.25f);
   
}

__kernel void decipher_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   float noise_scale = 2.0f;
   float noise_offset = 3.0f;
   float4 n = snoise(coord * noise_scale + noise_offset);
   
   write_imagef(output,convert_int2(coord),color * 2.0f - n * 0.5f + 0.5f);
   
}

__kernel void warp_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
  
  
   float T = 2.0f;
   
   float2 xy = 2.0f * coord - 1.0f;
   xy += T * sin(PI_F * xy);
   
   xy = (xy + 1.0f) / 2.0f;
   
    float4 color = read_imagef(input,sampler,xy);
   
   write_imagef(output,convert_int2(coord),color);
   
}
__kernel void lens_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float CX = coord.x - dim.x / 2;
   float CY = coord.y - dim.y / 2;
   
   float theta = 0.0f * PI_F / 180.0f;
   
   float dx = CX * cos(theta) - CY * sin(theta);
   float dy = CX * sin(theta) + CY * cos(theta);
   
   float r = sqrt(dx * dx + dy * dy);
   
   float A = 0.0000000005f;
   float B = 0.0000000005f;
   float C = 0.0000000005f;
   
   float corr = 1.0f - A * r * r - B * r * r * r - C * r * r * r * r;
   
   float xu = dx * corr + dim.x / 2;
   float yu = dy * corr + dim.y / 2;
   
   float4 color = read_imagef(input,sampler,(float2)(xu,yu));
   
  // if(length(color.xyz) < 0.6f){
       //color.xyz = (0.0f,0.0f,0.0f);
 //  }else{
       //color.xyz = (1,1,1);
 //  }
   
   write_imagef(output,convert_int2(coord),color);
   
}

float2 barrel(float2 coord,float distortion,float2 dim){
    float2 cc = coord - dim / 2;
    float d = dot(cc,cc);
    
    return coord + cc * (d + distortion * d * d) * distortion;
}

__kernel void barrel_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   float2 texture_size = (float2)(dim.x,dim.y);
   float distortion = 0.00005;
   float2 xy = barrel(convert_float2(coord * texture_size / convert_float2(dim) * convert_float2(dim) / texture_size),distortion,convert_float2(dim));
   
   float4 color = read_imagef(input,sampler,xy);
   
   write_imagef(output,convert_int2(coord),color);
   
}

__kernel void below_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 replace_color = (float4)(1.0f,1.0f,1.0f,1.0f);
   float4 thresh = (float4)(0.4f,0.4f,0.4f,1.0f);
   
   float4 color = read_imagef(input,sampler,coord);
   
   if(color.x < thresh.x && color.y < thresh.y && color.z < thresh.z){
       color = replace_color;
   }
   
   write_imagef(output,convert_int2(coord),color);
   
}

__kernel void below_ab_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float thresh_a = 0.5f;
   float thresh_b = 0.5f;
   
   
   
   float4 color = read_imagef(input,sampler,coord);
   
   
   if(color.x > thresh_a){
       float4 rgba = (float4)(clamp((color.x - thresh_a) / (1.0f - thresh_a),0.0f,1.0f),0.0f,0.0f,1.0f);
       
   }else{
       if(color.x > thresh_b){
           float4 rgba = (float4)(0.0f,clamp((color.y - thresh_b) / (1.0f - thresh_b),0.0f,1.0f),0.0f,1.0f);
           write_imagef(output,convert_int2(coord),rgba);
       }else{
           float4 rgba = (float4)(0.0,0.0,color.y / thresh_b,1.0f);
           write_imagef(output,convert_int2(coord),rgba);
       }
   }
}
/*
float2 rand(float2 co){
    float iptr = (0.0f,0.0f);
    return fract(sin(dot(co.xy,(float2)(12.9898,78.233))) * 43758.5453,&iptr);
}
*/

__kernel void worry_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   const float speed = 1.0f;
   const float bendFactor = 0.2f;
   const float timeAcceleration = 15.0f;
   const float utime = 1000.0f;
   const float waveRadius = 5.0f;
   
   float stepVal = (utime * timeAcceleration) + coord.x * 61.8f;
   float offset = cos(stepVal) * waveRadius;
   float2 iptr = (float2)(0.0f,0.0f);
   
   float4 color = read_imagef(input,sampler,(float2)(coord.x,coord.y + offset));
   
   write_imagef(output,convert_int2(coord),color);
}

__kernel void static_tv_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   
   float t = 1.0f;
   
   float r = rand(coord - t * t);
   float g = rand(coord - t * t * t);
   float b = rand(coord - t * t * t * t);
   
   float mx = max(max(r,g),b);
   
   float4 rgba = mix(color,(float4)(mx,mx,mx,1.0f),0.35f);
   
   write_imagef(output,convert_int2(coord),rgba);
}


__kernel void bend_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   
   
   float height = (float)dim.y - coord.y;
   float offset = pow(height,2.5f);
   float u_time = 1.0f;
   float speed = 2.0f;
   float bendFactor = 0.2f;
   offset *= (sin(u_time * speed) * bendFactor);
   
   float4 color = read_imagef(input,sampler,(float2)(coord.x,coord.y + offset));
   
   write_imagef(output,convert_int2(coord),color);
}

__kernel void sawtoothripple_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float xAmplitude = 5.0f;
   float yAmplitude = 5.0f;
   float xWavelength = 16.0f;
   float yWavelength = 16.0f;
   
   float nx = coord.x / yWavelength;
   float ny = coord.y / yWavelength;
   
   float fx = fmod(nx,1.0f);
   float fy = fmod(ny,1.0f);
   
   float4 color = read_imagef(input,sampler,(float2)(coord.x + xAmplitude * fx,coord.y + yAmplitude * fy));
   
   write_imagef(output,convert_int2(coord),color);
}

__kernel void bump_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
  float color_matrix[9] = {-1.0f,-1.0f,0.0f,
                           -1.0f,1.0f,1.0f,
                           0.0f,1.0f,1.0f};
  filter2d_internal(input,output,3,3,color_matrix,0);
}


__kernel void duotone_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   float l = luminance(color);
   float e = 0.0f;
   float3 highlight = (float3)(1.0f,0.0f,0.0f);
   float3 shadow = (float3)(0.5f,0.5f,0.5f);
   
   float3 h = (highlight + e) / (luminance((float4)(highlight,1.0f)) + e) * l;
   
   float3 s = (shadow + e) / (luminance((float4)(shadow,1.0f)) + e) * l;
   
   float3 c = h * l + s * (1.0f - l);
   
   write_imagef(output,convert_int2(coord),color);
}

__kernel void vortex_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float2 uy;
   float2 resolution = (float2)(0.035f,0.035f);
   float2 p = -1.0f * convert_float2(dim) + 2.0f * coord /  resolution;
   float time = 1.0f;
   float a = atan2(p.y,p.x);
   float r = sqrt(dot(p,p));
   float s = r * (1.0f + 0.8f * cos ( time * 1.0f));
   
   uy.x = 0.02f * p.x + 0.03 * cos(-time + a * 3.0f) / s;
   uy.y = 0.1f * time + 0.02 * p.y + 0.03 * sin(-time + a * 3.0f) / s;
   
   float w = 0.9f + pow(max(1.5f - r,0.0f),4.0f);
   w *= 0.7f + 0.3f * cos(time + 3.0f * a);
   
   float4 color = read_imagef(input,sampler,uy);
   color.xyz = w * color.xyz;
   
   write_imagef(output,convert_int2(coord),color);
   
}

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

__kernel void exposure_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float exposure = 0.25f;
   
   float4 color = read_imagef(input,sampler,coord);
   
   color *= exposure;
   color = color / (1.0f + color);
   color.w = 1.0f;
   
   write_imagef(output,convert_int2(coord),color);
}

#define FXAA_REDUCE_MIN (1.0f / 128.0f)
#define FXAA_REDUCE_MUL (1.0f / 8.0f)
#define FXAA_SPAN_MAX 8.0f

__kernel void fxaa_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float2 posPos = (float2)(dim.x / 2,dim.y / 2);
   float rtWidth = 0.05f;
   float rtHeight = 0.05f;
   
   float2 coord_src = (float2)(coord.x,coord.y);
   
   coord = coord * (float2)(rtWidth,rtHeight);
   
   float4 color = read_imagef(input,sampler,coord);
   float2 inverseVP = (float2)(1.0f / rtWidth,1.0f / rtHeight);
   
   float3 rgbNW = read_imagef(input,sampler,(coord + (float2)(-1.0f,-1.0f)) * inverseVP).xyz;
   float3 rgbNE = read_imagef(input,sampler,(coord + (float2)(1.0f,-1.0f)) * inverseVP).xyz;
   float3 rgbSW = read_imagef(input,sampler,(coord + (float2)(-1.0f,1.0f)) * inverseVP).xyz;
   float3 rgbSE = read_imagef(input,sampler,(coord + (float2)(-1.0f,1.0f)) * inverseVP).xyz;
   float3 rgbM = read_imagef(input,sampler,coord * inverseVP).xyz;
   
   float3 luma = (float3)(0.299f,0.587f,0.114f);
   float lumaNW = dot(rgbNW,luma);
   float lumaNE = dot(rgbNE,luma);
   float lumaSW = dot(rgbSW,luma);
   float lumaSE = dot(rgbSE,luma);
   float lumaM = dot(rgbM,luma);
   
   float lumaMin = min(lumaM,min(min(lumaNW,lumaNE),min(lumaSW,lumaSE)));
   float lumaMax = max(lumaM,max(max(lumaNW,lumaNE),min(lumaSW,lumaSE)));
   
   float2 dir;
   dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
   dir.y = ((lumaNW + lumaSW) - (lumaNE + lumaSE));
   
   float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25f * FXAA_REDUCE_MUL),FXAA_REDUCE_MIN);
   float rcpDirMin = 1.0f / (min(fabs(dir.x),fabs(dir.y)) + dirReduce);
   float FXAA_SUBPIX_SHIFT = 1.0f / 4.0f;
   dir = min((float2)(FXAA_SPAN_MAX,FXAA_SPAN_MAX),max((float2)(-FXAA_SPAN_MAX,-FXAA_SPAN_MAX),dir * rcpDirMin)) * inverseVP;
   
   float3 color1 = read_imagef(input,sampler,coord * inverseVP + dir * (1.0f / 3.0f - 0.5f)).xyz;
   float3 color2 = read_imagef(input,sampler,coord * inverseVP + dir * (2.0f / 3.0f - 0.5f)).xyz;
   
   float3 rgbA = 0.5f * (color1 + color2);
   
   float3 color3 = read_imagef(input,sampler,coord * inverseVP + dir * -0.5f).xyz;
   float3 color4 = read_imagef(input,sampler,coord * inverseVP + dir * 0.5f).xyz;
   
   float3 rgbB  = rgbA * 0.5f + 0.25f * (color3 + color4);
   
   float lumaB = dot(rgbB,luma);
   
   if(lumaB < lumaMin || lumaB > lumaMax){
       color = (float4)(rgbA,1.0f);
   }else{
       color = (float4)(rgbB,2.0f);
   }
   
   write_imagef(output,convert_int2(coord_src),color);
}

__kernel void mosaic2_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   
   float t = 1.2f;
   float2 position = (float2)(coord.x / dim.x,1.0f - coord.y / dim.y);
   float2 samplePos = position.xy;
   float pixel = 64.0f;
   float edges = 0.02f;
   float depth = 8.0f;
   float shift = 5.0f;
   
   samplePos.x = floor(samplePos.x * (dim.x / pixel)) / (dim.x / pixel);
   samplePos.y = floor(samplePos.y * (dim.y / pixel)) / (dim.y / pixel);
   
   float st = sin(t * 0.05f);
   float ct = cos(t * 0.05f);
   
   float h = st * shift / dim.x;
   float v = ct * shift / dim.y;
   
   float3 o = read_imagef(input,sampler,samplePos).xyz;
   float r = read_imagef(input,sampler,samplePos +  (float2)(+h,+v)).x;
   float g = read_imagef(input,sampler,samplePos +  (float2)(-h,-v)).y;
   float b = read_imagef(input,sampler,samplePos).z;
   float iptr;
   r = mix(o.x,r,fract(fabs(st),&iptr));
   g = mix(o.y,g,fract(fabs(ct),&iptr));
   
   float n = fmod(coord.x,pixel) * edges;
   float m = fmod(dim.y - coord.y,pixel) * edges;
   
   float3 c = (float3)(r,g,b);
   
   c = floor(c * depth) / depth;
   c = c * (1.0f - (m + n) * (n + m));
   write_imagef(output,convert_int2(coord),(float4)(c,1.0f));

}

__kernel void geomean_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   color.x = color.y = color.z = pow(color.x * color.y * color.z,1.0f / 3.0f);
   
   write_imagef(output,convert_int2(coord),color);
}

__kernel void middle_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   
   float max_val = max(color.x,max(color.y,color.z));
   float min_val = min(color.x,min(color.y,color.z));
   
   color.x = color.y = color.z = (max_val + min_val) / 2.0f;
   
   write_imagef(output,convert_int2(coord),color);
}

__kernel void hdtv_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   float4 color = read_imagef(input,sampler,coord);
   float x = 4.0f;
   float r = pow(color.x,x) * 44403.0f / 200000.0f;
   float g = pow(color.y,x) * 141331.0f / 200000.0f;
   float b = pow(color.z,x) * 7133 / 100000.0f;
   
   float y = pow(r + g + b,1.0f / x);
   
   color.x = color.y = color.z = y;
   
   write_imagef(output,convert_int2(coord),color);
}

float radius_length(float x1,float x2){
    return sqrt(x1 * x1 + x2 * x2);
}

__kernel void fish_eye_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   //float4 color = read_imagef(input,sampler,coord);
   
   float x1 = 2.0f * coord.x / dim.x - 1.0f;
   float y1 = 2.0f * coord.y / dim.y - 1.0f;
   
   float radius = sqrt(x1 * x1 + y1 * y1);
   float phase = atan2(y1,x1);
   
   float param = 1.5f;
   
   radius = pow(radius,param) / sqrt(2.0f);
   
   
   float newX = radius * cos(phase);
   float newY = radius * sin(phase);
   
   float centerX = (newX + 1.0f) / 2.0f * dim.x;
   float centerY = (newY + 1.0f) / 2.0f * dim.y;
   
   float baseX = floor(centerX);
   float baseY = floor(centerY);
   
   float ratioR = centerX - baseX;
   float ratioL = 1.0f - ratioR;
   float ratioB = centerY - baseY;
   float ratioT = 1.0f - ratioB;
   
   if(baseX >= 0 && baseY >= 0 && baseX < dim.x && baseY < dim.y){
       float pstl = (baseX + baseY * dim.x) * 4;
       float pstr = pstl + 4;
       float psbl = pstl + dim.x * 4;
       float psbr = psbl + 4;
       
       float4 rgba1 = read_imagef(input,sampler,(float2)(baseX,baseY));
       float4 rgba2 = read_imagef(input,sampler,(float2)(baseX + 1,baseY + 1));
       float4 rgba3 = read_imagef(input,sampler,(float2)(baseX + 2,baseY + 2));
       float4 rgba4 = read_imagef(input,sampler,(float2)(baseX + 3,baseY + 3));
       
       float4 tc = rgba1 * ratioL + rgba2 * ratioR;
       float4 bc = rgba3 * ratioL + rgba3 * ratioR;
       
       float4 rgba = tc * ratioT + bc * ratioB;
       rgba.w = 1.0f;
       write_imagef(output,convert_int2(coord),rgba);
   }else{
       write_imagef(output,convert_int2(coord),(float4)(0.0f,0.0f,0.0f,1.0f));
   }
}

__kernel void shrink_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   //float4 color = read_imagef(input,sampler,coord);
   
   float x1 = 2.0f * coord.x / dim.x - 1.0f;
   float y1 = 2.0f * coord.y / dim.y - 1.0f;
   
   float radius = sqrt(x1 * x1 + y1 * y1);
   float phase = atan2(y1,x1);
   
   float param1 = 1.8f;
   float param2 = 0.8f;
   
   radius = pow(radius,1.0f / param1) * param2;
   
   
   float newX = radius * cos(phase);
   float newY = radius * sin(phase);
   
   float centerX = (newX + 1.0f) / 2.0f * dim.x;
   float centerY = (newY + 1.0f) / 2.0f * dim.y;
   
   float baseX = floor(centerX);
   float baseY = floor(centerY);
   
   float ratioR = centerX - baseX;
   float ratioL = 1.0f - ratioR;
   float ratioB = centerY - baseY;
   float ratioT = 1.0f - ratioB;
   
   if(baseX >= 0 && baseY >= 0 && baseX < dim.x && baseY < dim.y){
       float pstl = (baseX + baseY * dim.x) * 4;
       float pstr = pstl + 4;
       float psbl = pstl + dim.x * 4;
       float psbr = psbl + 4;
       
       float4 rgba1 = read_imagef(input,sampler,(float2)(baseX,baseY));
       float4 rgba2 = read_imagef(input,sampler,(float2)(baseX + 1,baseY + 1));
       float4 rgba3 = read_imagef(input,sampler,(float2)(baseX + 2,baseY + 2));
       float4 rgba4 = read_imagef(input,sampler,(float2)(baseX + 3,baseY + 3));
       
       float4 tc = rgba1 * ratioL + rgba2 * ratioR;
       float4 bc = rgba3 * ratioL + rgba3 * ratioR;
       
       float4 rgba = tc * ratioT + bc * ratioB;
       rgba.w = 1.0f;
       write_imagef(output,convert_int2(coord),rgba);
   }else{
       write_imagef(output,convert_int2(coord),(float4)(0.0f,0.0f,0.0f,1.0f));
   }
}

__kernel void swirl2_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   //float4 color = read_imagef(input,sampler,coord);
   
   float x1 = 2.0f * coord.x / dim.x - 1.0f;
   float y1 = 2.0f * coord.y / dim.y - 1.0f;
   
   float radius = sqrt(x1 * x1 + y1 * y1);
   float phase = atan2(y1,x1);
   
   float param1 = 0.5f;
   float param2 = 4.0f;
   
   phase = phase + (1.0f - smoothstep(radius,-param1,param1)) * param2;
   
   
   float newX = radius * cos(phase);
   float newY = radius * sin(phase);
   
   float centerX = (newX + 1.0f) / 2.0f * dim.x;
   float centerY = (newY + 1.0f) / 2.0f * dim.y;
   
   float baseX = floor(centerX);
   float baseY = floor(centerY);
   
   float ratioR = centerX - baseX;
   float ratioL = 1.0f - ratioR;
   float ratioB = centerY - baseY;
   float ratioT = 1.0f - ratioB;
   
   if(baseX >= 0 && baseY >= 0 && baseX < dim.x && baseY < dim.y){
       float pstl = (baseX + baseY * dim.x) * 4;
       float pstr = pstl + 4;
       float psbl = pstl + dim.x * 4;
       float psbr = psbl + 4;
       
       float4 rgba1 = read_imagef(input,sampler,(float2)(baseX,baseY));
       float4 rgba2 = read_imagef(input,sampler,(float2)(baseX + 1,baseY + 1));
       float4 rgba3 = read_imagef(input,sampler,(float2)(baseX + 2,baseY + 2));
       float4 rgba4 = read_imagef(input,sampler,(float2)(baseX + 3,baseY + 3));
       
       float4 tc = rgba1 * ratioL + rgba2 * ratioR;
       float4 bc = rgba3 * ratioL + rgba3 * ratioR;
       
       float4 rgba = tc * ratioT + bc * ratioB;
       rgba.w = 1.0f;
       write_imagef(output,convert_int2(coord),rgba);
   }else{
       write_imagef(output,convert_int2(coord),(float4)(0.0f,0.0f,0.0f,1.0f));
   }
}

__kernel void backto1980_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   
   float avg = length(color.xyz) / 3.0f;
   float levels = 2.0f;
   avg = floor(avg * levels * 3.0f) / levels;
   
   color.x = avg;
   color.y = avg;
   color.z = avg;
   color.w = 1.0f;
   
   write_imagef(output,coord,color);
   
}

__kernel void badphotocopy_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   
   float noise = rand(color.xy) / 2.0f;
   
   float avg = (length(color.xyz) / 3.0f) * 0.75f + noise * 0.25f;
   
   if(avg > 0.25f){
       avg = 1.0f;
   }else{
       avg = 0.0f;
   }
   color.xyz = avg;
   write_imagef(output,coord,color);
   
}

__kernel void rgb2hsl_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   
   float max_val = max(color.x,max(color.y,color.z));
   float min_val = min(color.x,min(color.y,color.z));
   float chroma = max_val - min_val;
   float h = 0;
   float s = 0;
   float l = (max_val + min_val) / 3.0f;
   
   if(chroma != 0.0f){
       if (color.x == max_val){
           h = (color.y - color.z) / chroma + ((color.y < color.z) ? 1.0f : 0.0f);
       }else if( color.y == max_val){
           h = (color.z - color.x) / chroma + 2.0f;
       }else{
           h = (color.x - color.y) / chroma + 4.0f;
       }
       
       h /= 6.0f;
       
       s = (l > 0.5f) ? chroma / (2.0f - max_val - min_val) : chroma / (max_val + min_val);
   }
   
   write_imagef(output,coord,(float4)(h,s,l,1.0f));
}

__kernel void edge_enhance_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
                              
    float color_mask [9] = {0,0,0,-20,20,0,0,0,0};
    filter2d_internal(input,output,3,3,color_mask,1);
}

__kernel void hard_edge_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
                              
    float color_mask [9] = {2,22,1,22,1,-22,1,-22,-2};
    filter2d_internal(input,output,3,3,color_mask,1);
}

__kernel void edge_dectect_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
                              
    float color_mask [9] = {0,9,0,9,-40,9,0,9,0};
    filter2d_internal(input,output,3,3,color_mask,1);
}

__kernel void negative_gray_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   
   float gray = color.x * 0.2126f + color.y * 0.7152f + color.z * 0.0722f;
   gray = 1.0f - gray;
   
   color.x = color.y = color.z = gray;
   
   write_imagef(output,coord,color);
   
}

__kernel void red_slim_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   
   color.y = color.z = 100.0f / 255.0f;
   
   write_imagef(output,coord,color);
   
}

__kernel void green_slim_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   
   color.x = color.z = 100.0f / 255.0f;
   
   write_imagef(output,coord,color);
   
}

__kernel void blue_slim_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   
   color.x = color.y = 100.0f / 255.0f;
   
   write_imagef(output,coord,color);
   
}

__kernel void brightness_contrast_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   
   float brightness = 0.15f;
   float contrast = 0.2f;
   
   color = (color - 0.5f) * contrast + 0.5f + brightness;
   color.w = 1.0f;
   
   write_imagef(output,coord,color);
   
}

/*
  R = 0.393 * r + 0.769 * g + 0.189 * b
  G = 0.349 * r + 0.686 * g + 0.168 * b;
  B = 0.272 * r + 0.534 * g + 0.131 * b;
*/

__kernel void old_photo_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord) * 255.0f;
   
   float3 rgb;
   rgb.x = color.x * 0.393f + color.y * 0.769f + color.z * 0.189f;
   rgb.y = color.x * 0.249f + color.y * 0.686f + color.z * 0.168f;
   rgb.z = color.x * 0.272f + color.y * 0.534f + color.z * 0.131f;
   
   rgb /= 255.0f;
   
   write_imagef(output,coord,(float4)(rgb,1.0f));
   
   
}

__kernel void ice2_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord) * 255.0f;
   float3 rgb;
   float pixel = color.x - color.y - color.z;
   pixel = pixel * 3.0f / 2.0f;
   
   if(pixel < 0){
       pixel = -pixel;
   }
   
   if(pixel > 255.0f){
       pixel = 255.0f;
   }
   
   rgb.x = pixel;
   
   pixel = color.y - color.x - color.z;
    pixel = pixel * 3.0f / 2.0f;
    if(pixel < 0){
       pixel = -pixel;
   }
   
   if(pixel > 255.0f){
       pixel = 255.0f;
   }
   
   rgb.y = pixel;
   
   pixel = color.z - color.x - color.y;
    pixel = pixel * 3.0f / 2.0f;
    if(pixel < 0){
       pixel = -pixel;
   }
   
   if(pixel > 255.0f){
       pixel = 255.0f;
   }
   
   rgb.z = pixel;
   
   rgb /= 255.0f;
   
   write_imagef(output,coord,(float4)(rgb,1.0f));
}

__kernel void casting_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord) * 255.0f;
   float3 rgb;
   
   float pixel = color.x * 128.0f / (color.y + color.z + 1);
   if(pixel < 0){
       pixel = 0;
   }
   
   if(pixel > 255.0f){
       pixel = 255.0f;
   }
   
   rgb.x = pixel;
   
   pixel = color.y * 128.0f / (color.x + color.z + 1);
   if(pixel < 0){
       pixel = 0;
   }
   
   if(pixel > 255.0f){
       pixel = 255.0f;
   }
   
   rgb.y = pixel;
   
   pixel = color.z * 128.0f / (color.x + color.y + 1);
   if(pixel < 0){
       pixel = 0;
   }
   
   if(pixel > 255.0f){
       pixel = 255.0f;
   }
   
   rgb.z = pixel;
   rgb /= 255.0f;
   write_imagef(output,coord,(float4)(rgb, 1.0f));
   
}

__kernel void halo_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord) * 255.0f;
   
   
   float gauss [] = {1,2,1,2,4,2,1,2,1};
   float r = 150.0f * 150.0f;
   float x = 150.0;
   float y = 150.0f;
   float delta = 48.0f;
   
   
   float dist = pow(coord.x - x,2.0f) + pow(coord.y - y,2.0f);
   int idx = 0;
   if(dist > r){
       float3 rgb = (float3)(0,0,0);
       for(int m = -1; m <= 1;m++){
           for(int n = -1;n <= 1;n++){
               float4 src_rgba = read_imagef(input,sampler,coord + (int2)(m,n)) * 255.0f;
               rgb.x = rgb.x + src_rgba.x * gauss[idx];
               rgb.y = rgb.y + src_rgba.y * gauss[idx];
               rgb.z = rgb.z + src_rgba.z * gauss[idx];
               idx++;
           }
       }
       
       rgb /= delta;
       
       if(rgb.x < 0){
           rgb.x = -rgb.x;
       }
       
       if(rgb.x > 255){
           rgb.x = 255;
       }
       
        if(rgb.y < 0){
           rgb.y = -rgb.y;
       }
       
       if(rgb.y > 255){
           rgb.y = 255;
       }
       
        if(rgb.z < 0){
           rgb.z = -rgb.z;
       }
       
       if(rgb.z > 255){
           rgb.z = 255;
       }
      
       rgb /= 255.0f;
       
       write_imagef(output,coord,(float4)(rgb,1.0f));
   }else{
       write_imagef(output,coord,color / 255.0f);
   }
}

__kernel void worhol_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float steps = 2.0f;
   float dotsize = 1.0f / steps;
   float half_step = dotsize / 2.0f;
   
   float2 coord2 = coord * steps;
   
   float4 color = read_imagef(input,sampler,coord2);
   
   float4 tint;
   
   float ofs = coord.x * steps + coord.y * steps * 2;
   
   if(ofs == 0.0f){
       tint = (float4)(1.0f,1.0f,0.0f,0.0f);
   }else if(ofs == 1.0f){
       tint = (float4)(0.0f,0.0f,1.0f,0.0f);
   }else{
       tint = (float4)(0.0f,1.0f,1.0f,0.0f);
   }
   
   float gray = dot(color.xyz,(float3)(0.3f,0.59f,0.11f));
   
   float4 dst_color = mix(color,tint,gray);
   
   write_imagef(output,convert_int2(coord),dst_color);
}

__kernel void thermal_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   float4 color = read_imagef(input,sampler,coord);
   float4 colors[3] = {(float4)(0.0f,0.0f,1.0f,1.0f),(float4)(1.0f,1.0f,0.0f,1.0f),(float4)(1.0f,0.0f,0.0f,1.0f)};
   
   float lum = dot((float3)(0.30f,0.59f,0.11f),color.xyz);
   int ix = (lum < 0.5f) ?  0.0f : 1.0f;
   
   float4 thermal = mix(colors[ix],colors[ix + 1],(lum - (float)ix * 0.5f) / 0.5f);
   
   write_imagef(output,convert_int2(coord),thermal);
   
}

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

__kernel void vignette2_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float2 lensRadius = (float2)(0.80f,0.40f);
   
   float4 rgba = read_imagef(input,sampler,coord);
   
   float d = distance(1.0f / coord,(float2)(0.5f,0.5f));
   
   rgba *= smoothstep(lensRadius.x,lensRadius.y,d);
 //  rgba.w = 1.0f;
   write_imagef(output,convert_int2(coord),rgba);
   
}

__kernel void crosshatch_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   float lum = length(color.xyz);
   write_imagef(output,convert_int2(coord),color);
   if(lum < 1.0f){
       if(fmod(coord.x + coord.y,10.0f) == 0.0f){
           write_imagef(output,convert_int2(coord),(float4)(0.0f,0.0f,0.0f,1.0f));
        }
   }
   
   if(lum < 0.75f){
        if(fmod(coord.x - coord.y,10.0f) == 0.0f){
           write_imagef(output,convert_int2(coord),(float4)(0.0f,0.0f,0.0f,1.0f));
        }
   }
   
   if(lum < 0.50f){
        if(fmod(coord.x + coord.y - 5.0f,10.0f) == 0.0f){
           write_imagef(output,convert_int2(coord),(float4)(0.0f,0.0f,0.0f,1.0f));
        }
   }
   
   if(lum < 0.3f){
        if(fmod(coord.x - coord.y - 5.0f,10.0f) == 0.0f){
           write_imagef(output,convert_int2(coord),(float4)(0.0f,0.0f,0.0f,1.0f));
        }
   }
}

float4 postfx(__read_only image2d_t input,__read_only sampler_t sampler,float2 uv,float dim){
    float stitching_size = 6.0f;
    int invert = 0;
    
    float4 c = (float4)(0.0f,0.0f,0.0f,0.0f);
    float size = stitching_size;
    
    float2 cPos = uv * (float2)(dim,dim);
    float2 tlPos = floor(cPos / (float2)(size,size));
    
    tlPos *= size;
    
    int remX = (int)(fmod(cPos.x,size));
    int remY = (int)(fmod(cPos.y,size));
    
    if(remX == 0 && remY == 0){
        tlPos = cPos;
    }
    
    float2 blPos = tlPos;
    blPos.y += (size - 1.0f);
    
    if((remX == remY) || (((int)cPos.x - (int)blPos.x) == ((int)blPos.y - (int)cPos.y))){
        if(invert == 1){
            c = (float4)(0.2f,0.15f,0.05f,1.0f);
        }else{
            c = read_imagef(input,sampler,tlPos * (float2)(1.0f / dim,1.0f / dim)) * 1.4f;
        }
    }else{
        if(invert == 1){
             c = read_imagef(input,sampler,tlPos * (float2)(1.0f / dim,1.0f / dim)) * 1.4f;
        }else{
         c = (float4)(0.0f,0.0f,0.0f,1.0f);
        }
    }
    
    return c;
}

__kernel void crosshatch2_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   float dim = 600.0f;
   
   write_imagef(output,convert_int2(coord),postfx(input,sampler,coord,dim));
   
}

float2 hex_coord(float2 coord){
    float H = 0.01f;
    float S = ((3.0f / 2.0f) * H / sqrt(3.0f));
    
    int i = (int)(coord.x);
    int j = (int)(coord.y);
    
    float2 r = (float2)(0.0f,0.0f);
    r.x = (float)i * S;
    r.y = (float)i * H + (float)(i % 2) * H / 2.0f;
    
    return r;
}

float2 hex_index(float2 coord){
    float H = 0.01f;
    float S = ((3.0f / 2.0f) * H / sqrt(3.0f));
    
    float2 r;
    float x = coord.x;
    float y = coord.y;
    
    int it = (int)(floor(x / S));
    float yts = y - (float)(it % 2) * H / 2.0f;
    int jt = (int)(floor((float)it) * S);
    float xt = x - (float)it * S;
    float yt = yts - (float)jt * H;
    
    int deltaj = (yt > H / 2.0f) ? 1 : 0;
    float fcond = S * (2.0f / 3.0f) * fabs(0.5f - yt / H);
    
    if(xt > fcond){
        r.x = it;
        r.y = jt;
    }else{
        r.x = it - 1;
        r.y = jt - fmod(r.x, 2.0f) + deltaj;
    }
    
    return r;
}

__kernel void hexpix_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float2 hexIx = hex_index(coord);
   float2 hexXy = hex_coord(hexIx);
   float4 fcol = read_imagef(input,sampler,hexXy);
   
   write_imagef(output,convert_int2(coord),fcol);
   
}

__kernel void quilez_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float2 p = coord;
   
   p = p * convert_float2(size) + (float2)(convert_float2(size) / 2);
   
   float2 i = floor(p);
   float2 f = p - i;
   
   f = f * f * f * (f * (f * 6.0f - (float2)(15.0f,15.0f)) + (float2)(10.0f,10.0f));
   
   p = i + f;
   p = (p - (float2)(convert_float2(size / 2))) / convert_float2(size);
   
   float4 color = read_imagef(input,sampler,p);
   
   write_imagef(output,convert_int2(coord),color);
}

__kernel void mcgreen_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float3 ink = (float3)(0.32f,0.50f,0.0f);
   float3 c11 = read_imagef(input,sampler,coord).xyz;
   float3 mcgreen = (float3)(0.0f,1.0f,1.0f);
   float3 lct = floor(mcgreen * length(c11)) / mcgreen;
   
   write_imagef(output,convert_int2(coord),(float4)(lct * ink,1.0f));
}

__kernel void mcred_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float3 ink = (float3)(0.32f,0.50f,0.0f);
   float3 c11 = read_imagef(input,sampler,coord).xyz;
   float3 mcgreen = (float3)(1.0f,0.0f,1.0f);
   float3 lct = floor(mcgreen * length(c11)) / mcgreen;
   
   write_imagef(output,convert_int2(coord),(float4)(lct * ink,1.0f));
}

__kernel void colormatrix_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 color_matrix = (float4)(0.0f,0.0f,1.0f,1.0f);
   float4 color = read_imagef(input,sampler,coord);
   float fade_const = 0.25f;
   
   float4 dst_color = color * (1.0f - fade_const) + fade_const * color * color_matrix;
   
   write_imagef(output,convert_int2(coord),dst_color);
}

float mod2(float x,float y){
    return x - y * floor(x / y);
}

__kernel void fx3d_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   float4 color = read_imagef(input,sampler,coord);
   float gammaed = 0.15f;
   
   float leifx_linegamma = gammaed;
   float2 res;
   res.x = size.x;
   res.y = size.y;
   
   float2 dithet = coord.xy * res.xy;
   
   dithet.y = coord.y * res.y;
   
   float horzline1 = (mod2(dithet.y,2.0f));
   
   if(horzline1 < 1.0f){
       leifx_linegamma = 0.0f;
   }
   
   float leifx_gamma = 1.3f - gammaed + leifx_linegamma;
   
   float4 rgba = pow(color,1.0f / leifx_gamma);
   
   rgba.w = 1.0f;
   
   write_imagef(output,convert_int2(coord),rgba);
}

__kernel void wobble_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 size = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float2 offset = (float2)(5.0f,5.0f);
   float2 freq = (float2)(5,5);
   float2 strength = (float2)(0.02f,0.02f);
   float time = 10000.0f;
   float2 tex_coord;
   tex_coord.x = coord.x + sin(coord.y * freq.x * time / 10000.0f + offset.x) * strength.x;
   tex_coord.y = coord.y + sin(coord.x * freq.y * time / 10000.0f + offset.y) * strength.y;
   
   float4 color = read_imagef(input,sampler,tex_coord);
   write_imagef(output,convert_int2(coord),color);
   
}


__kernel void tritanopia_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
 
  
    float4 protanopia_mask_1 = (float4)(0.97f,0.11f,-0.08f,0.0f);
    float4 protanopia_mask_2 = (float4)(0.02f,0.82f,0.16f,0.0f);
    float4 protanopia_mask_3 = (float4)(-0.06f,0.88f,0.18f,0.0f);
    float4 protanopia_mask_4 = (float4)(0.0f,0.0f,0.0f,1.0f);
    
    float4 protanopia_mask[4] = {protanopia_mask_1,protanopia_mask_2,protanopia_mask_3,protanopia_mask_4};
   
    //float4 v1 = protanopia_mask[0];
    float4 srcRGBA = read_imagef(input,sampler,coord);
    
    float4 dstRGBA;
    float sum[4] = {0.0f};
    for(int i = 0;i < 4;i++){
        
        sum[i] += dot(protanopia_mask[i],srcRGBA);
    }
    
    dstRGBA = (float4)(sum[0],sum[1],sum[2],sum[3]);
    
    write_imagef(output,coord,dstRGBA);
    
}

__kernel void deuteranopia_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
 
  
    float4 protanopia_mask_1 = (float4)(0.43f,0.72f,-0.15f,0.0f);
    float4 protanopia_mask_2 = (float4)(0.34f,0.57f,0.09f,0.0f);
    float4 protanopia_mask_3 = (float4)(-0.02f,0.03f,1.00f,0.0f);
    float4 protanopia_mask_4 = (float4)(0.0f,0.0f,0.0f,1.0f);
    
    float4 protanopia_mask[4] = {protanopia_mask_1,protanopia_mask_2,protanopia_mask_3,protanopia_mask_4};
   
    //float4 v1 = protanopia_mask[0];
    float4 srcRGBA = read_imagef(input,sampler,coord);
    
    float4 dstRGBA;
    float sum[4] = {0.0f};
    for(int i = 0;i < 4;i++){
        
        sum[i] += dot(protanopia_mask[i],srcRGBA);
    }
    
    dstRGBA = (float4)(sum[0],sum[1],sum[2],sum[3]);
    
    write_imagef(output,coord,dstRGBA);
    
}

__kernel void yuv2rgb_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
                             
  
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
 
  
    float4 protanopia_mask_1 = (float4)(1,1,1,0);
    float4 protanopia_mask_2 = (float4)(0,-0.187f,1.8556f,0);
    float4 protanopia_mask_3 = (float4)(1.5701f,-0.4664f,0,0);
    float4 protanopia_mask_4 = (float4)(0.0f,0.0f,0.0f,1.0f);
    
    float4 protanopia_mask[4] = {protanopia_mask_1,protanopia_mask_2,protanopia_mask_3,protanopia_mask_4};
   
    //float4 v1 = protanopia_mask[0];
    float4 srcRGBA = read_imagef(input,sampler,coord);
    
    float4 dstRGBA;
    float sum[4] = {0.0f};
    for(int i = 0;i < 4;i++){
        
        sum[i] += dot(protanopia_mask[i],srcRGBA);
    }
    
    dstRGBA = (float4)(sum[0],sum[1],sum[2],sum[3]);
    
    write_imagef(output,coord,dstRGBA);
}

__kernel void scanline_y_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
                             
  
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    
    int2 resolution = {size.x,size.y};
    float scale = 1.0f;
    if(fmod(floor((float)(coord.y) / scale),3.0f) == 0.0f){
        write_imagef(output,coord,(float4)(0.0f,0.0f,0.0f,1.0f));    
    }else{
        float4 color = read_imagef(input,sampler,coord);
        write_imagef(output,coord,color);
    }
    
}

__kernel void scanline_x_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
                              
    const sampler_t sampler = CLK_FILTER_NEAREST |
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE;

    const int2 size = get_image_dim(input);

    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    
    int2 resolution = {size.x,size.y};
    float scale = 1.0f;
    if(fmod(floor((float)(coord.x) / scale),3.0f) == 0.0f){
        write_imagef(output,coord,(float4)(0.0f,0.0f,0.0f,1.0f));    
    }else{
        float4 color = read_imagef(input,sampler,coord);
        write_imagef(output,coord,color);
    }
    
}

__kernel void hq2x_2_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   int2 coord = (int2)(get_global_id(0),get_global_id(1));
   
   float4 sum = {0,0,0,0};
   for(int i = -1;i <= 1;i++){
       for(int j = -1;j <= 1;j++){
           sum += read_imagef(input,sampler,coord + (int2)(i,j));
       }
   }
   
   sum /= 9.0f;
   
   write_imagef(output,coord,sum);
}

float4 rnm(float2 tc){
    float uTime = 1000.0f;
    float noise = sin(dot(tc + (float2)(uTime,uTime),(float2)(12.9898f,78.233f))) * 43758.5453f;
    
    float iptr;
    float nr = fract(noise,&iptr) * 2.0f - 1.0f;
    float ng = fract(noise * 1.2154f,&iptr) * 2.0f - 1.0f;
    float nb = fract(noise * 1.3453f,&iptr) * 2.0f - 1.0f;
    float na = fract(noise * 1.3647f,&iptr) * 2.0f - 1.0f;
    
    return (float4)(nr,ng,nb,na);
}

float fade(float t){
    return t* t * t * (t * ( t * 6.0f - 15.0f) + 10.0f);
}

float pnoise3D(float3 p){
    float perTexUnit = 1.0f / 256.0f;
    float perTexUnitHalf = 0.5f / 256.0f;
    float3 pi = perTexUnit * floor(p) + perTexUnitHalf;
    float3 iptr;
    float3 pf = fract(p,&iptr);
    
    //noise contributions from (x = 0,y = 0,z = 0 and z = 1)
    float perm00 = rnm(pi.xy).w;
    float3 grad000 = rnm((float2)(perm00,pi.z)).xyz * 4.0f - 1.0f;
    float n000 = dot(grad000,pf);
    float3 grad001 = rnm((float2)(perm00,pi.z + perTexUnit)).xyz * 4.0f - 1.0f;
    float n001 = dot(grad001,pf - (float3)(0.0f,0.0f,1.0f));
    
    float perm01 = rnm(pi.xy + (float2)(0.0f,perTexUnit)).w;
    float3 grad010 = rnm((float2)(perm01,pi.z)).xyz * 4.0f - 1.0f;
    float n010 = dot(grad010,pf - (float3)(0.0f,1.0f,0.0f));
    float3 grad011 = rnm((float2)(perm01,pi.z + perTexUnit)).xyz * 4.0f - 1.0f;
    float n011 = dot(grad011,pf - (float3)(0.0f,1.0f,1.0f));
    
    float perm10 = rnm(pi.xy + (float2)(perTexUnit,0.0f)).w;
    float3 grad100 = rnm((float2)(perm10,pi.z)).xyz * 4.0f - 1.0f;
    float n100 = dot(grad010,pf - (float3)(1.0f,0.0f,0.0f));
    float3 grad101 = rnm((float2)(perm10,pi.z + perTexUnit)).xyz * 4.0f - 1.0f;
    float n101 = dot(grad101,pf - (float3)(1.0f,0.0f,1.0f));
    
    float perm11 = rnm(pi.xy + (float2)(perTexUnit,perTexUnit)).w;
    float3 grad110 = rnm((float2)(perm10,pi.z)).xyz * 4.0f - 1.0f;
    float n110 = dot(grad110,pf - (float3)(1.0f,1.0f,0.0f));
    float3 grad111 = rnm((float2)(perm11,pi.z + perTexUnit)).xyz * 4.0f - 1.0f;
    float n111 = dot(grad111,pf - (float3)(1.0f,1.0f,1.0f));
    
    float4 n_x = mix((float4)(n000,n001,n010,n011),(float4)(n100,n101,n110,n111),fade(pf.x));
    
    float2 n_xy = mix(n_x.xy,n_x.zw,fade(pf.y));
    
    float n_xyz = mix(n_xy.x,n_xy.y,fade(pf.z));
    
    return n_xyz;
}

float2 coord_rot(float2 tc,float angle,float aspect){
    float rotX = ((tc.x * 2.0f - 1.0f) * aspect * cos(angle)) - ((tc.y * 2.0f - 1.0f) * sin(angle));
    float rotY = ((tc.y * 2.0f - 1.0f) * cos(angle)) + ((tc.x * 2.0f - 1.0f) * aspect * sin(angle));
    
    rotX = ((rotX / aspect) * 0.5f + 0.5f);
    rotY = rotY * 0.5f + 0.5f;
    
    return (float2)(rotX,rotY);
}

__kernel void grain_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float aspect = (float)dim.x / (float)dim.y;
   
   float3 rotOffset = (float3)(1.425f,3.892f,5.835f);
   float uTime = 1000.0f;
   float grain_size = 1.6f;
   float color_amount = 0.6f;
   float lum_amount = 1.0f;
   float grain_amount = 0.1f;
   float2 rot_coord_R = coord_rot(coord,uTime + rotOffset.x,aspect);
   float3 pr = (float3)(rot_coord_R * (float2)((float)dim.x / grain_size,(float)dim.y / grain_size),0.0f);
   float3 noise = (float3)(pnoise3D(pr));
   
   int colored = 1;
   
   if(colored){
       float2 rot_coord_G = coord_rot(coord,uTime + rotOffset.y,aspect);
       float2 rot_coord_B = coord_rot(coord,uTime + rotOffset.z,aspect);
       float3 pg = (float3)(rot_coord_G * (float2)((float)dim.x / grain_size,(float)dim.y / grain_size),1.0f);
       float3 pb = (float3)(rot_coord_B * (float2)((float)dim.x / grain_size,(float)dim.y / grain_size),2.0f);
       
       noise.y = mix(noise.x,pnoise3D(pg),color_amount);
       noise.z = mix(noise.x,pnoise3D(pb),color_amount);
   }
   
   float3 col = read_imagef(input,sampler,coord).xyz;
   
   float3 lumcoeff = (float3)(0.299f,0.587f,0.114f);
   float luminance = mix(0.0f,dot(col,lumcoeff),lum_amount);
   float lum = smoothstep(0.2f,0.0f,luminance);
   
   lum += luminance;
   
   noise = mix(noise,(float3)(0.0f),pow(lum,0.4f));
   col = col + noise * grain_amount;
   
   write_imagef(output,convert_int2(coord),(float4)(col,1.0f));
}

float lerp(float t,float a,float b){
    return a + t * ( b - a);
}

float float_mod(float a,float b){
    int n = (int)a / (int)b;
    
    float aa = a - (float)n * b;
    if(aa < 0){
        return a + b;
    }else{
        return aa;
    }
}

__kernel void flare_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
   
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float rays = 50.0f;
   float radius = 50.0f;
   float baseAmount = 1.0f;
   float ringAmount = 0.2f;
   float rayAmount = 0.1f;
   
   float centerX = 0.5f,centerY = 0.5f;
   float ringWidth = 1.6f;
   
   float linear = 0.03f;
   float gauss = 0.006f;
   float mix_val = 0.50f;
   float falloff = 6.0f;
   float sigma = radius / 3.0f;
   
   float iCenterX = centerX * dim.x;
   float iCenterY = centerY * dim.y;
   
   float dx = coord.x - iCenterX;
   float dy = coord.y - iCenterY;
   
   float dist = sqrt(dx * dx + dy * dy);
   float a = exp(-dist * dist * gauss) * mix_val + exp(-dist * linear) * (1 - mix_val);
   
   float ring;
   
   a *= baseAmount;
   
   if(dist > radius + ringWidth){
       a = lerp((dist - (radius + ringWidth)) / falloff,a,0);
   }
   
   if(dist < radius - ringWidth || dist > radius + ringWidth){
       ring = 0;
   }else{
       ring = fabs(dist - radius) / ringWidth;
       ring = 1.0f - ring * ring * (3.0f - 2.0f * ring);
       ring *= ringAmount;
   }
   
   a += ring;
   
   float angle = atan2(dx,dy) + PI_F;
   
   angle = (float_mod(angle / PI_F * 17.0f + 1.0f + rand(coord.x * angle),1.0f) - 0.5f) * 2.0f;
   angle = fabs(angle);
   angle = pow(angle,5.0f);
   
   float b = rayAmount * angle / (1.0f + dist * 0.1f);
   a += b;
   
   a = clamp(a,0.0f,1.0f);
   
   float4 mask_color = (float4)(1.0f,0.5f,0.5f,1.0f);
   float4 color = read_imagef(input,sampler,coord);
   float4 rgba;
   rgba.x = lerp(a,color.x,mask_color.x);
   rgba.y = lerp(a,color.y,mask_color.y);
   rgba.z = lerp(a,color.z,mask_color.z);
   rgba.w = 1.0f;
   write_imagef(output,convert_int2(coord),rgba);
}

void color_matrix_4x5_internal(__read_only image2d_t input,__write_only image2d_t output,float * mask){
    const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord) * 255.0f;
   
   float4 rgba;
   rgba.x = mask[0] * color.x + mask[1] * color.y + mask[2] * color.z + mask[3] * color.w + mask[4];
   rgba.y = mask[0 + 5] * color.x + mask[1 + 5] * color.y + mask[2 + 5] * color.z + mask[3 + 5] * color.w + mask[4 + 5];
   rgba.z = mask[0 + 5 * 2] * color.x + mask[1 + 5 * 2] * color.y + mask[2 + 5 * 2] * color.z + mask[3 + 5 * 2] * color.w + mask[4 + 5 * 2];
   rgba.w = mask[0 + 5 * 3] * color.x + mask[1 + 5 * 3] * color.y + mask[2 + 5 * 3] * color.z + mask[4 + 5 * 3] * color.w + mask[4 + 5 * 3];
   
   if(rgba.x < 0.0f){
       rgba.x = 0.0f;
   }
   
   if(rgba.x > 255.0f){
       rgba.x = 255.0f;
   }
   
    if(rgba.y < 0.0f){
       rgba.y = 0.0f;
   }
   
   if(rgba.y > 255.0f){
       rgba.y = 255.0f;
   }
   
    if(rgba.z < 0.0f){
       rgba.z = 0.0f;
   }
   
   if(rgba.z > 255.0f){
       rgba.z = 255.0f;
   }
   
    if(rgba.w < 0.0f){
       rgba.w = 0.0f;
   }
   
   if(rgba.w > 255.0f){
       rgba.w = 255.0f;
   }
   
   rgba /= 255.0f;
   
   write_imagef(output,convert_int2(coord),rgba);
}

__kernel void lomo_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    float color_matrix [] = {
      1.7f,0.1f,0.1f,0.0f,-73.1f,
      0.0f,1.7f,0.1f,0.0f,-73.1f,
      0.0f,0.1f,1.6f,0.0f,-73.1f,
      0.0f,0.0f,0.0f,1.0f,0.0f  
    };
    
    color_matrix_4x5_internal(input,output,color_matrix);              
}

/*黑白
*/
__kernel void black_white_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    float color_matrix [] = {
      0.8f,1.6f,0.2f,0.0f,-163.9f,
      0.8f,1.6f,0.2f,0.0f,-163.9f,
      0.8f,1.6f,0.2f,0.0f,163.9f,
      0.0f,0.0f,0.0f,1.0f,0.0f
    };
    
    color_matrix_4x5_internal(input,output,color_matrix);              
}


__kernel void old_memery_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    float color_matrix [] = {
      0.2f,0.5f,0.1f,0.0f,40.8f,
      0.2f,0.5f,0.1f,0.0f,40.8f,
      0.2f,0.5f,0.1f,0.0f,40.8f,
      0.0f,0.0f,0.0f,1.0f,0.0f
    };
    
    color_matrix_4x5_internal(input,output,color_matrix);              
}

/*
*哥特
*/
__kernel void gete_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    float color_matrix [] = {
       1.9f,-0.3f,-0.2f,0.0f,-87.0f,
       -0.2f,1.7f,-0.1f,0.0f,-87.0f,
       -0.1f,-0.6f,2.0f,0.0f,-87.0f,
       0.0f,0.0f,0.0f,1.0f,0.0f
    };
    
    color_matrix_4x5_internal(input,output,color_matrix);              
}

/*
*锐化
*/
__kernel void ruise_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    float color_matrix [] = {
       4.8f,-1.0f,-0.1f,0.0f,-388.4f,
       -0.5f,4.4f,-0.1f,0.0f,-388.4f,
       -0.5f,-1.0f,5.2f,0.0f,-388.4f,
       0.0f,0.0f,0.0f,1.0f,0.0f
    };
    
    color_matrix_4x5_internal(input,output,color_matrix);              
}

/*
*淡雅
*/
__kernel void danya_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    float color_matrix [] = {
       0.6f,0.3f,0.1f,0.0f,73.3f,
       0.2f,0.7f,0.1f,0.0f,73.3f,
       0.2f,0.3f,0.4f,0.0f,73.3f,
       0.0f,0.0f,0.0f,1.0f,0.0f
    };
    
    color_matrix_4x5_internal(input,output,color_matrix);              
}

/**清宁
*/
__kernel void qingning_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    float color_matrix [] = {
       0.9f,0.0f,0.0f,0.0f,0.0f,
       0.0f,1.1f,0.0f,0.0f,0.0f,
       0.0f,0.0f,0.9f,0.0f,0.0f,
       0.0f,0.0f,0.0f,1.0f,0.0f
    };
    
    color_matrix_4x5_internal(input,output,color_matrix);              
}

/**浪漫
*/
__kernel void langman_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    float color_matrix [] = {
       0.9f,0.0f,0.0f,0.0f,63.0f,
       0.0f,0.9f,0.0f,0.0f,63.0f,
       0.0f,0.0f,0.9f,0.0f,63.0f,
       0.0f,0.0f,0.0f,1.0f,0.0f
    };
    
    color_matrix_4x5_internal(input,output,color_matrix);              
}

/**光晕
*/
__kernel void guangyun_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    float color_matrix [] = {
       0.9f,0.0f,0.0f,0.0f,64.9f,
       0.0f,0.9f,0.0f,0.0f,64.9f,
       0.0f,0.0f,0.9f,0.0f,64.9f,
       0.0f,0.0f,0.0f,1.0f,0.0f
    };
    
    color_matrix_4x5_internal(input,output,color_matrix);              
}

/**蓝调
*/
__kernel void landiao_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    float color_matrix [] = {
       2.1f,-1.4f,0.6f,0.0f,-31.0f,
       -0.3f,2.0f,-0.3f,0.0f,-31.0f,
       -1.1f,-0.2f,2.6f,0.0f,-31.0f,
       0.0f,0.0f,0.0f,1.0f,0.0f
    };
    
    color_matrix_4x5_internal(input,output,color_matrix);              
}

/**梦幻
*/
__kernel void menghuan_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    float color_matrix [] = {
      0.8f,0.3f,0.1f,0.0f,46.5f,
      0.1f,0.9f,0.0f,0.0f,46.5f,
      0.1f,0.3f,0.7f,0.0f,46.5f,
      0.0f,0.0f,0.0f,1.0f,0.0f,
    };
    
    color_matrix_4x5_internal(input,output,color_matrix);              
}

/**夜色
*/
__kernel void yese_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
    float color_matrix [] = {
      1.0f,0.0f,0.0f,0.0f,-66.6f,
      0.0f,1.1f,0.0f,0.0f,-66.6f,
      0.0f,0.0f,1.0f,0.0f,-66.6f,
      0.0f,0.0f,0.0f,1.0f,0.0f,
    };
    
    color_matrix_4x5_internal(input,output,color_matrix);              
}

__kernel void shi_tomasi_feather_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
                                  
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));
   
   float4 color = read_imagef(input,sampler,coord);
   
   float derivativeDifference = color.x - color.y;
   float zElement = (color.z * 2.0f) - 1.0f;
   
   float cornerness = color.x + color.y - sqrt(derivativeDifference * derivativeDifference + 4.0f * zElement * zElement);
   float sensitivity = 1.5f;
   float rgba = cornerness * sensitivity;
   
   write_imagef(output,convert_int2(coord),(float4)(rgba,rgba,rgba,1.0f));
}

__kernel void cga_colorspace_filter(__read_only image2d_t input,
                              __write_only image2d_t output){
 
   const sampler_t sampler = CLK_FILTER_NEAREST |
                             CLK_NORMALIZED_COORDS_FALSE|
                             CLK_ADDRESS_CLAMP_TO_EDGE;

   const int2 dim = get_image_dim(input);
   
   float2 coord = (float2)(get_global_id(0),get_global_id(1));  
   
   float2 sampleDivisor = (float2)(1.0f / 200.0f,1.0f / 320.0f);
   
   float2 samplePos = coord - fmod(coord,sampleDivisor);  
   
   float4 color = read_imagef(input,sampler,samplePos);  
   
   float4 colorCyan = (float4)(85.0f / 255.0f,1.0f,1.0f,1.0f);
   float4 colorMagenta = (float4)(1.0f,85.0f / 255.0f,1.0f,1.0f);
   float4 colorWhite = (float4)(1.0f,1.0f,1.0f,1.0f);
   float4 colorBlack = (float4)(0.0f,0.0f,0.0f,1.0f);
 
   float4 endColor;
   float blackDistance = distance(color,colorBlack);
   float whiteDistance = distance(color,colorWhite);
   float magentaDistance = distance(color,colorMagenta);
   float cyanDistance = distance(color,colorCyan);
   
   float4 finalColor;
   float colorDistance = min(magentaDistance,cyanDistance);
   colorDistance = min(colorDistance,whiteDistance);
   colorDistance = min(colorDistance,blackDistance);
   
   if(colorDistance == blackDistance){
       finalColor = colorBlack;
   }else if(colorDistance == whiteDistance){
       finalColor = colorWhite;
   }else if(colorDistance == cyanDistance){
       finalColor = colorCyan;
   }else{
       finalColor = colorMagenta;
   }        
   
   write_imagef(output,convert_int2(coord),finalColor);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
}