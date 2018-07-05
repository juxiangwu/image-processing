/* Please Write the OpenCL Kernel(s) code here*/

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
