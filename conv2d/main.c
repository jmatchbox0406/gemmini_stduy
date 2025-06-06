#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#define BANK_ROWS 1024
#define BANK_NUM 2
#define DIM 8
#define ACC_ROWS 512
#define GARBAGE_ADDR 0xFFFFFFFF

#define INPUT_CH 3
#define INPUT_C 6
#define INPUT_R 7
#define KERNEL_C 2
#define KERNEL_R 3

int32_t *data_i;
int32_t *weight;
// Declare scratchpad memory
int32_t scratchpad[DIM * BANK_ROWS * BANK_NUM];
float accumulator_sram[ACC_ROWS][DIM];

void gemmini_extended_mvout(int32_t* dram_addr, uint32_t sp_addr, uint32_t cols, uint32_t rows) {
    //printf("[mvout] sp_addr: %u, cols: %u, rows: %u", sp_addr, cols, rows);
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < cols; c++) {
            uint32_t idx = r * DIM + c;
            uint32_t sp_idx = sp_addr * DIM + idx;
            if (sp_idx < DIM * BANK_ROWS) {
                dram_addr[idx] = scratchpad[sp_idx];
                //printf("  dram_addr[%u] = %d", idx, scratchpad[sp_idx]);
            }
        }
    }
}

void gemmini_extended_mvin(const int32_t* dram_addr, uint32_t sp_addr, uint32_t cols, uint32_t rows) {
    printf("[mvin] sp_addr: %u, cols: %u, rows: %u\n", sp_addr, cols, rows);
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < cols; c++) {
            uint32_t idx = r * DIM + c;
            uint32_t sp_idx = sp_addr * DIM + idx;
            if (sp_idx < DIM * BANK_ROWS * BANK_NUM) {
                if (dram_addr != NULL) {
                    int flat_idx = (int)(dram_addr - data_i + r*cols+c);
                    //printf("  dram_addr=%x, data_i = %x , flat_idx = %d\n", dram_addr, data_i, flat_idx);
                    int r = flat_idx / (INPUT_C * INPUT_CH);
                    int c = (flat_idx / INPUT_CH) % INPUT_C;
                    int ch = flat_idx % INPUT_CH;
                    printf("  scratchpad[%02u] : Input[%02d][%02d][%02d] (from dram_addr[%u] = %d)", sp_idx/8, ch, r, c, idx, dram_addr[idx]);
                }
                else printf("  scratchpad[%u] = 0 (zero-padded)", sp_idx/8);
                printf("\n");
                scratchpad[sp_idx] = (dram_addr != NULL) ? dram_addr[idx] : 0;
            }
        }
    }
}

void gemmini_extended_mvin2(const int32_t* dram_addr, uint32_t sp_addr, uint32_t cols, uint32_t rows, uint32_t dim_o_ch) {
    printf("[mvin2] sp_addr: %u, cols: %u, rows: %u\n", sp_addr, cols, rows);
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < cols; c++) {
            uint32_t idx = r * DIM + c;
            uint32_t sp_idx = sp_addr * DIM + idx;
            if (sp_idx < DIM * BANK_ROWS * BANK_NUM) {
                if (dram_addr != NULL) {
                    int flat_idx = (int)(dram_addr - weight + r*cols+c);
                    // printf("  dram_addr=%x, weight = %x , flat_idx = %d\n", dram_addr, weight, flat_idx);
                    int out_ch = flat_idx % dim_o_ch;
                    int tmp1 = flat_idx / dim_o_ch;
                    int in_ch = tmp1 % INPUT_CH;
                    int tmp2 = tmp1 / INPUT_CH;
                    int k_c = tmp2 % KERNEL_C;
                    int k_r = tmp2 / KERNEL_C;
                    printf("  scratchpad[%02u] : Weight[%02d][%02d][%02d][%02d] (from dram_addr[%u] = %x)", sp_idx/8, in_ch, k_r, k_c, out_ch, idx, dram_addr[idx]);
                }
                else printf("  scratchpad[%u] = 0 (zero-padded)", sp_idx/8);
                printf("\n");
                scratchpad[sp_idx] = (dram_addr != NULL) ? dram_addr[idx] : 0;
            }
        }
    }
}


// New function: gemmini_extended_compute_preloaded
void gemmini_extended_compute_preloaded(uint32_t A_sp_addr, uint32_t BD_sp_addr,
    uint32_t A_cols, uint32_t A_rows,
    uint32_t BD_cols, uint32_t BD_rows) {
    printf("[gemmini_extended_compute_preloaded] A_sp_addr: %u, BD_sp_addr: %u, A_cols: %u, A_rows: %u, BD_cols: %u, BD_rows: %u\n",
    A_sp_addr, BD_sp_addr, A_cols, A_rows, BD_cols, BD_rows);

    for (uint32_t i = 0; i < A_rows; i++) {
        for (uint32_t j = 0; j < BD_cols; j++) {
            int32_t sum = 0;
            for (uint32_t k = 0; k < A_cols; k++) {
                uint32_t A_index = A_sp_addr * DIM + i * A_cols + k;
                uint32_t B_index = BD_sp_addr * DIM + k * BD_cols + j;

                if (A_index < DIM * BANK_ROWS * BANK_NUM && B_index < DIM * BANK_ROWS * BANK_NUM) {
                    int32_t A_val = scratchpad[A_index];
                    int32_t B_val = scratchpad[B_index];
                    sum += A_val * B_val;
                    printf("SP[%u][%u](%d) * SP[%u][%u](%d) = %x\n",
                    A_index / DIM, A_index % DIM, A_val,
                    B_index / DIM, B_index % DIM, B_val,
                    A_val * B_val);
                }
            }

            if (i < ACC_ROWS && j < DIM) {
                accumulator_sram[i][j]+= sum;
            }
        }
    }
}

// Function to simulate GEMMINI compute accumulated operation
void gemmini_extended_compute_accumulated(uint32_t A_sp_addr, uint32_t BD_sp_addr, uint32_t A_cols, uint32_t A_rows, uint32_t BD_cols, uint32_t BD_rows, uint32_t w_sp_addr_start)
{
    printf("[gemmini_extended_compute_accumulated] A_sp_addr: %u, BD_sp_addr: %u, A_cols: %u, A_rows: %u, BD_cols: %u, BD_rows: %u \n", A_sp_addr, BD_sp_addr, A_cols, A_rows, BD_cols, BD_rows);

    for (uint32_t i = 0; i < A_rows; i++) {
        for (uint32_t j = 0; j < BD_cols; j++) {
            int32_t sum = 0;
            for (uint32_t k = 0; k < A_cols; k++) {
                uint32_t A_index = A_sp_addr * DIM + i * A_cols + k;
                uint32_t B_index = (w_sp_addr_start + BD_sp_addr) * DIM + k * BD_cols + j;

                if (A_index < DIM * BANK_ROWS * BANK_NUM && B_index < DIM * BANK_ROWS * BANK_NUM) {
                    int32_t A_val = scratchpad[A_index];
                    int32_t B_val = scratchpad[B_index];
                    sum += A_val * B_val;
                    // printf("A_index %d, B_index %d, b_offset %d \n", A_index, B_index, (k * BD_cols + j));
                    if(B_index < 2024*8+10)
                        printf("SP[%d][%d](%d)*SP[%d][%d](%d) = %x \n",A_index/DIM , A_index%DIM,A_val,B_index/DIM, B_index%DIM,B_val  , A_val * B_val);
                }
            }

            // Save result to accumulator_sram
            if (i < ACC_ROWS && j < DIM) {
                accumulator_sram[i][j] += sum;
                //printf("Accumulator[%d][%d] = %f", i, j, accumulator_sram[i][j]);
                //printf("\n");
            }
        }
    }
}


static void conv_2d (
     const int32_t * data_i   ,
     const int32_t * weight   ,
     const int32_t  * bias   ,
     int8_t * data_o         ,
     uint32_t dim_i_r        ,
     uint32_t dim_i_c        ,
     uint32_t dim_i_ch       ,
     uint32_t dim_w_r        ,
     uint32_t dim_w_c        ,
     uint32_t dim_o_r        ,
     uint32_t dim_o_c        ,
     uint32_t dim_o_ch       ,
     int32_t  i_r_t          ,
     int32_t  i_c_t          ,
     uint32_t i_ch_t         ,
     uint32_t o_r_t          ,
     uint32_t o_c_t          ,
     uint32_t o_ch_t         ,
     uint32_t i_r_s          ,
     uint32_t i_c_s          ,
     uint32_t i_ch_s         ,
     uint32_t o_r_s          ,
     uint32_t o_c_s          ,
     uint32_t o_ch_s         ,
     uint32_t stride_c       ,
     uint32_t stride_r       ,
     uint32_t zp_u           ,
     uint32_t zp_l           ,
     uint32_t w_sp_addr_start,
     uint32_t o_sp_addr_start,
     bool     mv_out         ,
     bool     accu           ,
     bool     mv_in_bias     ,
     uint32_t i_ch_blck      ,
     uint32_t o_ch_blck      ,
     uint32_t dim_w_r_c      ,
     bool     full_bias
)
{
    #if (0)
    printf("i_c_t  : %d\n", i_c_t);
    printf("i_r_t  : %d\n", i_r_t);
    printf("o_c_t  : %d\n", o_c_t);
    printf("o_r_t  : %d\n", o_r_t);
    printf("i_ch_t : %d\n", i_ch_t);
    printf("o_ch_t : %d\n", o_ch_t);
    printf("i_c_s  : %d\n", i_c_s);
    printf("i_r_s  : %d\n", i_r_s);
    printf("o_c_s  : %d\n", o_c_s);
    printf("o_r_s  : %d\n", o_r_s);
    printf("i_ch_s : %d\n", i_ch_s);
    printf("o_ch_s : %d\n", o_ch_s);
    #endif

    //mv in bias
    //printf("b\n");
    if (mv_in_bias)
    {
        //uint32_t mv_out_col_blck = o_ch_blck > MAX_BLOCK_LEN ? MAX_BLOCK_LEN : o_ch_blck;
        //mv_out_col_blck *= DIM;
        for (int o_ch = 0; o_ch < o_ch_t; o_ch += DIM)
        {
            for (int o_r = 0; o_r < o_r_t; o_r += 1)
            {
                for (int o_c = 0; o_c < o_c_t; o_c += 1)
                {
                    // printf("mvinb 0x%08X,0x%08X,%d,\n", b_dram_addr, o_sp_addr, o_col);
                    // gemmini_extended_mvin3(b_dram_addr, o_sp_addr, o_col, 1);
                    // +++
                    //gemmini_fence();
                    // ---
                }
            }
        }
    }

    //mv in input
    //printf("in\n");
    // +++
    // if(dim_i_r == 1) {
    //     i_r_s = 0;
    // }
    // ---
    uint32_t i_r = 0;
    uint32_t mvi_sum_r = 0;
    uint32_t mvi_sum_c = 0;
    uint32_t mvi_cnt = 0;
    // 全部要搬進去的 row = i_r_t*i_c_t, col = in_channels
    while (i_r < i_r_t)
    {
        int32_t i_r_abs = i_r_s + i_r;
        uint32_t i_c = 0;
        while (i_c < i_c_t)
        {
            int32_t i_c_abs = i_c_s + i_c;
            uint32_t i_ch = 0;
            uint32_t mvin_once_i_row =  /*in zp_l*/i_c_abs       < 0       ? 0 - i_c_s         :
                              /*in zp_r*/i_c_abs      == dim_i_c ? i_c_t - i_c       :
                              /*to zp_r*/i_c_abs + DIM > dim_i_c ? dim_i_c - i_c_abs :
                              /*others */i_c + DIM     > i_c_t   ? i_c_t - i_c       : DIM;

            /*
                In cases where i_c_t is smaller than dim_i_c, the third condition can give incorrect mvin_once_i_row.
                For instance, if input dimension is 14, stride is 2, kernel dimension is 1, and output dimension is 7,
                then output tiling would be 7, input tiling would be (7 - 1) * 2 + 1 = 13.
                i_c: 0 -> mvin_once_i_row is in "others" and mvin_once_i_row would be 8.
                i_c: 8 -> mvin_once_i_row is in "to zp_r" and mvin_once_i_row would be 14(dim_i_c) - 8(i_c_abs) = 6.
                Yet 8 + 6 = 14 exceeds the input tiling size, which is 13.
                So we add a condition to prevent i_c + mvin_once_i_row will exceeds i_c_t.
            */
            if( i_c + mvin_once_i_row > i_c_t )
            {
                mvin_once_i_row = i_c_t - i_c;
            }

            /* Gemmini will halt if mvin_once_i_row is greater than DIM. */
            if( mvin_once_i_row > DIM )
            {
                mvin_once_i_row = DIM;
            }

            while (i_ch < i_ch_t)
            {
                uint32_t i_col = i_ch + DIM > i_ch_t ? i_ch_t - i_ch : DIM; /* Gemmini will halt if i_col is greater than DIM. */
                uint32_t i_sp_addr = (i_ch / DIM * i_r_t + i_r) * i_c_t + i_c;
                if ((i_c_abs < 0) || (i_r_abs < 0) || (i_c_abs >= dim_i_c) || (i_r_abs >= dim_i_r))
                {
                    //printf("mvin_p i_sp_addr %d,i_col %d,mvin_once_i_row %d,\n", i_sp_addr, i_col, mvin_once_i_row);
                    mvi_sum_r += mvin_once_i_row;
                    mvi_sum_c += i_col;
                    mvi_cnt++;
                    gemmini_extended_mvin(NULL, i_sp_addr, i_col, mvin_once_i_row);
                }
                else
                {
                    const int32_t * i_dram_addr = data_i + (i_r_abs * dim_i_c + i_c_abs) * dim_i_ch + i_ch + i_ch_s;
                    // i_col = in_channels
                    //printf("mvin_i i_dram_addr 0x%08X,i_sp_addr %d,i_col %d,mvin_once_i_row %d,\n", i_dram_addr, i_sp_addr, i_col, mvin_once_i_row);
                    mvi_sum_r += mvin_once_i_row;
                    mvi_sum_c += i_col;
                    mvi_cnt++;
                    gemmini_extended_mvin(i_dram_addr, i_sp_addr, i_col, mvin_once_i_row);
                }
                i_ch += i_col;
                // +++
                //gemmini_fence();
                // ---
            }
            i_c += mvin_once_i_row;
        }
        i_r += 1;
    }
    //printf("mvi_sum data size r %d , c %d, cnt %d\n", mvi_sum_r, mvi_sum_c, mvi_cnt);
    //mv in weight
    //printf("o_ch_t %d\n", o_ch_t);
    mvi_sum_r = 0;
    mvi_sum_c = 0;
    mvi_cnt = 0;
    for (int o_ch = 0; o_ch < o_ch_t; o_ch+=DIM)
    {
        uint32_t w_col = o_ch + DIM > o_ch_t ? o_ch_t - o_ch : DIM;
        for (int i_ch = 0; i_ch < i_ch_t; i_ch+=DIM)
        {
            uint32_t w_row = i_ch + DIM > i_ch_t ? i_ch_t - i_ch : DIM;
            uint32_t i_ch_abs = i_ch + i_ch_s;
            for (int k_r_c = 0; k_r_c < dim_w_r_c; k_r_c++)
            {
                const int32_t* w_dram_addr = weight + (k_r_c * dim_i_ch + i_ch_abs) * dim_o_ch + o_ch_s + o_ch;
                //const uint32_t * w_dram_addr = weight + (i_ch_abs * dim_w_r_c + k_r_c) * dim_o_ch + o_ch_s + o_ch;
                uint32_t w_sp_addr = w_sp_addr_start + (o_ch/DIM * dim_w_r_c + k_r_c) * i_ch_t + i_ch;
                // printf("0x%08X,0x%08X,%d,%d\n", w_dram_addr, w_sp_addr, w_col, w_row);
                gemmini_extended_mvin2(w_dram_addr, w_sp_addr, w_col, w_row, dim_o_ch);
                //printf("*\n");
                // +++
                //gemmini_fence();
                // ---
            }
        }
    }
    //printf("mvi_sum weight size r %d , c %d, cnt %d\n", mvi_sum_r, mvi_sum_c, mvi_cnt);

    //compute
    //printf("compute\n");
    uint16_t  o_cnt=0;
    for (int o_ch_s = 0; o_ch_s < o_ch_t; o_ch_s+=DIM)
    {
        uint32_t o_col = o_ch_s + DIM > o_ch_t ? o_ch_t - o_ch_s : DIM;
        bool accu_psum = accu | false;
        for (int i_ch = 0; i_ch < i_ch_t; i_ch+=DIM)
        {
            uint32_t i_col = i_ch + DIM > i_ch_t ? i_ch_t - i_ch : DIM;
            for (int k_r = 0; k_r < dim_w_r; k_r+=1)
            {
                for (int k_c = 0; k_c < dim_w_c; k_c+=1)
                {
                    bool change_wght = true;
                    uint32_t w_sp_addr = w_sp_addr_start + ((o_ch_s/DIM * dim_w_r + k_r) * dim_w_c + k_c) * i_ch_t + i_ch;

                    for (int o_c = 0; o_c < o_c_t; o_c+=DIM)
                    {
                        //uint32_t o_row = o_c + o_c_t > o_c_t ? o_c_t - o_c : DIM;
                        uint32_t o_row = o_c + DIM > o_c_t ? o_c_t - o_c : DIM;
                        uint32_t i_c = o_c * stride_c + k_c;
                        for (int o_r = 0; o_r < o_r_t; o_r+=1)
                        {
                            uint32_t i_r = o_r * stride_r + k_r;
                            // 假如要加bias，就是把bit30=1
                            uint32_t o_sp_addr = (o_sp_addr_start | ((accu_psum | mv_in_bias) << 30)) + (o_ch_s / DIM * o_r_t + o_r) * o_c_t + o_c;
                            uint32_t i_sp_addr = (i_ch / DIM * i_r_t + i_r) * i_c_t + i_c;
                            printf("i_sp_addr %x,o_sp_addr %X,o_col %d,i_col %d,o_row %d\n", i_sp_addr, o_sp_addr, o_col, i_col, o_row);
                            printf("i_ch %d, i_r_t %d,i_r  %d,i_c_t %d\n", i_ch, i_r_t, i_r, i_c_t);
                            //printf("o_sp_addr : %x\n\n",o_sp_addr);
                            // 先搬weight
                            if(w_sp_addr == GARBAGE_ADDR)
                            {
                                o_cnt++;
                                //printf("preload_in only ,o_sp_addr %X,o_col %d,i_col %d,o_row %d\n", o_sp_addr, o_col, i_col, o_row);
                            }
                            else
                            {
                                // printf("preload_w+in w_sp_addr %d,o_sp_addr %X,o_col %d,i_col %d,o_row %d\n", w_sp_addr, o_sp_addr, o_col, i_col, o_row);
                                // printf("o_cnt %d\n", ++o_cnt);
                                o_cnt = 0;
                            }

                            // gemmini_extended_preload(/*BD     */w_sp_addr ,
                            //                          /*C      */o_sp_addr ,
                            //                          /*BD_cols*/o_col , /*BD_rows*/i_col ,
                            //                          /*C_cols */o_col , /*C_rows */o_row);
                            if (change_wght)
                            {
                                //printf("cal+activate  w_sp_addr %d,o_sp_addr %X,o_col %d,i_col %d,o_row %d\n", w_sp_addr, o_sp_addr, o_col, i_col, o_row);
                                uint32_t BD_sp_offset = (w_sp_addr >= w_sp_addr_start) ? (w_sp_addr - w_sp_addr_start) : w_sp_addr;
                                gemmini_extended_compute_preloaded(/*A      */i_sp_addr, /*BD     */BD_sp_offset,
                                                                   /*A_cols */i_col    , /*A_rows */o_row,
                                                                   /*BD_cols*/o_col    , /*BD_rows*/i_col);
                                change_wght = false;
                                // w_sp_addr = GARBAGE_ADDR;
                            }
                            else
                            {
                                // printf("accumulated i_sp_addr %d,o_sp_addr %X,o_col %d,i_col %d,o_row %d\n", i_sp_addr, o_sp_addr, o_col, i_col, o_row);
                                uint32_t BD_sp_offset = (w_sp_addr >= w_sp_addr_start) ? (w_sp_addr - w_sp_addr_start) : w_sp_addr;
                                gemmini_extended_compute_accumulated(/*A      */i_sp_addr, /*BD    */BD_sp_offset,
                                                                    /*A_cols */i_col    , /*A_rows */o_row,
                                                                    /*BD_cols*/o_col    , /*BD_rows*/i_col, w_sp_addr_start);


                            }
                            // +++
                            //gemmini_fence();
                            // ---
                        }
                    }
                    accu_psum = true;
                }
            }

        }
    }

    //mv out data
    //printf("+mv out\n");
    if(mv_out)
    {
        //uint32_t mv_out_col_blck = o_ch_blck > MAX_BLOCK_LEN ? MAX_BLOCK_LEN : o_ch_blck;
        //mv_out_col_blck *= DIM;
        for (int o_ch_s = 0; o_ch_s < o_ch_t; o_ch_s += DIM)
        {
            for (int o_r = 0; o_r < o_r_t; o_r += 1)
            {
                for (int o_c = 0; o_c < o_c_t; o_c += DIM)
                {
                    // printf("mvout o_dram_addr 0x%X,o_sp_addr %X,o_col %d,o_row %d\n", o_dram_addr, o_sp_addr, o_col, o_row);
                    //gemmini_extended_mvout(o_dram_addr, o_sp_addr, o_col, o_row);
                    //printf("mvout-\n");
                    // +++
                    //gemmini_fence();
                    // ---
                }
            }
        }
        printf("mv out\n");
    }
    //
}

void conv_2d_auto(
     const int32_t *    data_i,
     const int32_t *    weight,
     const int32_t  *    bias,
     int8_t *    data_o     ,
     uint32_t    dim_i_c    ,
     uint32_t    dim_i_r    ,
     uint32_t    dim_i_ch   ,
     uint32_t    dim_o_c    ,
     uint32_t    dim_o_r    ,
     uint32_t    dim_o_ch   ,
     uint32_t    dim_k_c    ,
     uint32_t    dim_k_r    ,
     uint32_t    stride_c   ,
     uint32_t    stride_r   ,
     uint32_t    zp_l       ,
     uint32_t    zp_u       ,
     int         act        ,
     float       i_scale    ,
     float       w_scale    ,
     float       b_scale    ,
     float       o_scale    ,
     bool        full_bias  ,
     size_t      relu6_shift
)
{
    //*****先預留空間給weight */
    uint32_t dim_k_rxc = dim_k_r * dim_k_c;
    uint32_t wght_basic_size = dim_k_rxc * (dim_i_ch > DIM ? DIM : dim_i_ch);
    //printf("wght_basic_size %d\n", wght_basic_size);
    uint32_t w_sp_addr_start = BANK_ROWS * BANK_NUM  - wght_basic_size;
    // scratchpad 的weight存放方式=  dim_k_r * dim_k_c* dim_i_ch * o_ch_s; 這裡先不管o_ch，假設為1
    //*****依造ACC_ROWS的大小，取得最大的 o_c_t*o_r_t */
        //⭣⭣⭣⭣⭣⭣這段再決定o_c_t的大小⭣⭣⭣⭣⭣⭣/
        uint32_t o_c_t = dim_o_c;
        // 根據目前o_c_t的大小，來計算i_c_t的大小
        uint32_t i_c_t = (o_c_t - 1) * stride_c + dim_k_c;
        //printf("cal i_c_t + pad= %d \n", i_c_t);
        //printf("w_sp_addr_start = %d\n", w_sp_addr_start);
        //printf("need acc_row's size = o_c_t*o_r_t\n");
        //printf("make sure ? o_c_t*o_r_t < ACC_ROWS\n");
        while (((o_c_t > ACC_ROWS) || (i_c_t > w_sp_addr_start)) && o_c_t != 1 && i_c_t != 1)
        {
            // 目前的o_c_t放不進gemmini，就除2。
            o_c_t = (o_c_t >> 1) + (o_c_t & 1);
            i_c_t = (o_c_t - 1) * stride_c + dim_k_c;
            printf("down o_c_t cal i_c_t + pad= %d \n", i_c_t);
        }
        // 還不知道這段在幹嘛
        if ((o_c_t != 1) && (o_c_t != dim_o_c) && (o_c_t + (o_c_t >> 1)) <= ACC_ROWS && ((o_c_t + (o_c_t >> 1) - 1) * stride_c + dim_k_c) <= w_sp_addr_start)
        {
            o_c_t += o_c_t >> 1;
            i_c_t = (o_c_t - 1) * stride_c + dim_k_c;
            printf("[2]i_c_t=(%d-1)*%d + %d = %d\n", o_c_t, stride_c, dim_k_c, i_c_t);
        }
        printf("finish o_c_t %d, i_c_t %d\n", o_c_t, i_c_t);
        //⭡⭡⭡⭡⭡⭡⭡⭡⭡這段再決定o_c_t的大小 */


        //⭣⭣⭣⭣⭣⭣這段再決定o_r_t的大小⭣⭣⭣⭣⭣⭣/
        uint32_t rest_acc_row = ACC_ROWS / o_c_t;
        uint32_t rest_spad_row = w_sp_addr_start / i_c_t;
        uint32_t o_r_t = dim_o_r;
        // 根據目前o_r_t的大小，來計算i_r_t的大小
        uint32_t i_r_t = (o_r_t - 1) * stride_r + dim_k_r;
        printf("cal i_r_t + pad= %d \n", i_c_t);
        while ((o_r_t > rest_acc_row || (i_r_t > rest_spad_row)) && o_r_t != 1 && i_r_t != 1)
        {

            o_r_t = (o_r_t >> 1) + (o_r_t & 1);
            i_r_t = (o_r_t - 1) * stride_r + dim_k_r;
            printf("down o_r_t  cal  i_r_t + pad= %d \n", i_r_t);
        }
        if ((o_r_t != 1) && (o_r_t != dim_o_r) && (o_r_t + (o_r_t >> 1)) <= rest_acc_row && ((o_r_t + (o_r_t >> 1) - 1) * stride_r + dim_k_r) <= rest_spad_row)
        {
            o_r_t += o_r_t >> 1;
            i_r_t = (o_r_t - 1) * stride_r + dim_k_r;
            printf("down o_r_t  cal  i_r_t + pad= %d \n", i_r_t);
        }
        printf("final o_r_t %d, i_r_t %d\n", o_r_t, i_r_t);
        //⭡⭡⭡⭡⭡⭡⭡⭡⭡這段再決定o_r_t的大小 */

        //⭣⭣⭣⭣⭣⭣這段再決定o_ch_t的大小⭣⭣⭣⭣⭣⭣/
        // 假如o_c_t*o_r_t很小，代表可以同時多放幾組o_c_t*o_r_t
        rest_acc_row /= o_r_t;
        printf("rest_acc_row %d = I can put how many times o_c_t*o_r_t\n", rest_acc_row);
        uint32_t i_2d_t = i_c_t * i_r_t;
        rest_spad_row = (BANK_ROWS * BANK_NUM - i_2d_t) / wght_basic_size;
        // o_ch_t_blck_ori是用來計算，原本的的dim_o_ch要切成幾塊/DIM
        uint32_t o_ch_t_blck_ori = dim_o_ch / DIM + (dim_o_ch % DIM != 0);
        printf("rest_spad_row %d\n", rest_spad_row);
        printf("o_ch_t_blck_ori %d\n", o_ch_t_blck_ori);


        uint32_t o_ch_t_blck = o_ch_t_blck_ori;
        while(((o_ch_t_blck > rest_acc_row) || (o_ch_t_blck > rest_spad_row)) && (o_ch_t_blck != 1))
        {
            o_ch_t_blck = (o_ch_t_blck >> 1) + (o_ch_t_blck & 0x1);
        }
        if ((o_ch_t_blck != 1) && (o_ch_t_blck != o_ch_t_blck_ori) && (o_ch_t_blck + (o_ch_t_blck >> 1) <= rest_acc_row) && (o_ch_t_blck + (o_ch_t_blck >> 1) <= rest_spad_row))
        {
            o_ch_t_blck += o_ch_t_blck >> 1;
        }
        printf("I can put total times out_channel = final o_ch_t_blck %d\n", o_ch_t_blck);


        uint32_t o_ch_t = o_ch_t_blck * DIM;
        //⭡⭡⭡⭡⭡⭡⭡⭡⭡這段再決定o_ch_t的大小 */

        //⭣⭣⭣⭣⭣⭣這段再決定i_ch_t的大小⭣⭣⭣⭣⭣⭣/
        uint32_t i_ch_t = dim_i_ch;
        printf("i_ch_t %d\n", i_ch_t);
        //uint32_t total_spad_row = i_2d_t * (dim_i_ch / DIM + (dim_i_ch % DIM != 1)) + dim_k_rxc * i_ch_t * o_ch_t_blck;
        uint32_t total_spad_row = i_2d_t * (dim_i_ch / DIM + (dim_i_ch % DIM != 0)) + dim_k_rxc * i_ch_t * o_ch_t_blck;
        printf("total_spad_row first = %d\n", total_spad_row);
        while(total_spad_row > BANK_ROWS * BANK_NUM)
        {
            i_ch_t = (i_ch_t >> 1) + (i_ch_t & 0x1);
            total_spad_row = i_2d_t * (i_ch_t / DIM + (i_ch_t % DIM != 0)) + dim_k_rxc * i_ch_t * o_ch_t_blck;
            printf("total_spad_row in while = %d, i_ch_t %d\n", total_spad_row, i_ch_t);
        }
        //看scratchpad剩下多少空間，可以的話加大i_ch_t
        if (i_2d_t * ((i_ch_t + (i_ch_t >> 1)) / DIM + ((i_ch_t + (i_ch_t >> 1)) % DIM != 0)) + dim_k_rxc * (i_ch_t + (i_ch_t >> 1)) * o_ch_t_blck <= BANK_ROWS * BANK_NUM)
        {
            printf("i_ch_t %d += %d\n", i_ch_t, (i_ch_t >> 1));
            i_ch_t += i_ch_t >> 1;
            printf("total_spad_row add = %d\n", total_spad_row);
        }
        printf("final i_ch_t %d\n", i_ch_t);
        //⭡⭡⭡⭡⭡⭡⭡⭡⭡這段再決定i_ch_t的大小 */

    w_sp_addr_start = BANK_ROWS * BANK_NUM - dim_k_rxc * i_ch_t * o_ch_t_blck;
    printf("final w_sp_addr_start %d\n", w_sp_addr_start);
    uint32_t i_ch_t_blck = i_ch_t / DIM + (i_ch_t % DIM != 0);

    //ex config
    // gemmini_extended_config_ex(/*dataflow   */WS      , /*act        */act         , /*sys_shift*/0       ,
    //                            /*acc_scale  */o_scale , /*relu6_shift*/relu6_shift , /*A_stride */stride_c,
    //                            /*a_transpose*/false   , /*b_transpose*/false);

    // //in load config
    // gemmini_extended3_config_ld(/*stride*/dim_i_ch, /*scale*/i_scale, /*shrunk*/false, 0);

    // //output store config
    // gemmini_config_st(/*stride_C * sizeof_C*/dim_o_ch);
    // //wght load config
    // gemmini_extended3_config_ld(/*stride*/dim_o_ch, /*scale*/w_scale, /*shrunk*/false, 1);

    // //bias load config
    // gemmini_extended3_config_ld(/*stride_D * sizeof_D*/full_bias ? dim_o_ch << 2 : dim_o_ch , b_scale, !full_bias, 2);

    // +++
    // 1d conv zp_u should be 0.
    if(dim_i_r == 1) {
        zp_u = 0;
    }
    // ---
    printf("ecah time conv_2d's size = o_ch_t*o_r_t*o_c_t = %d*%d*%d\n", o_ch_t, o_r_t, o_c_t);
    int xxx_o_ch = 0;
    for (int o_ch_s = 0; o_ch_s < dim_o_ch; o_ch_s+=o_ch_t)
    {
        uint32_t o_ch_num = o_ch_s + o_ch_t > dim_o_ch ? dim_o_ch - o_ch_s : o_ch_t;
        xxx_o_ch += o_ch_num;
        printf("sum of o_ch_num %d\n", xxx_o_ch);
        for (int o_r_s = 0; o_r_s < dim_o_r; o_r_s+=o_r_t)
        {
            uint32_t o_r_num = o_r_s + o_r_t > dim_o_r ? dim_o_r - o_r_s : o_r_t;
            uint32_t i_r_num = (o_r_num - 1) * stride_r + dim_k_r;
            int32_t i_r_s = o_r_s * stride_r - zp_u;

            //printf("i_r_s %d\n", i_r_s);
            // printf("o_r_num %d\n", o_r_num);
            //printf(" %d\n", i_r_num);
            for (int o_c_s = 0; o_c_s < dim_o_c; o_c_s+=o_c_t)
            {
                uint32_t o_c_num = o_c_s + o_c_t > dim_o_c ? dim_o_c - o_c_s : o_c_t;
                uint32_t i_c_num = (o_c_num - 1) * stride_c + dim_k_c;
                int32_t i_c_s = o_c_s * stride_c - zp_l;
                printf("i_r_s %d, i_c_s %d\n",i_r_s ,i_c_s);
                printf("o_ch_num %d, o_r_num %d, o_c_num %d\n", o_ch_num, o_r_num, o_c_num);
                printf("i_r_num %d, i_c_num %d\n", i_r_num, i_c_num);
                bool mv_in_bias = bias != NULL;
                for (int i_ch = 0; i_ch < dim_i_ch; i_ch+=i_ch_t)
                {
                    // 全部的input channel都計算完，就可以mv out
                    bool mv_out = (i_ch + i_ch_t >= dim_i_ch) ;
                    uint32_t i_ch_num = mv_out ? dim_i_ch - i_ch : i_ch_t;
                    //printf("i_ch_num %d\n", i_ch_num);
                    bool accu   = i_ch != 0;
                    //printf("[%d,%d,%d][%d,%d][%d,%d,%d]\n",dim_i_r, dim_i_c, dim_i_ch, dim_k_r, dim_k_c, dim_o_r, dim_o_c, dim_o_ch);
                    conv_2d(
                    /*int8_t * data_i         */data_i ,
                    /*int8_t * weight         */weight ,
                    /*int32_t* bias           */bias ,
                    /*int8_t * data_o         */(int8_t *) data_o ,
                    /*uint32_t dim_i_r        */dim_i_r           ,
                    /*uint32_t dim_i_c        */dim_i_c           ,
                    /*uint32_t dim_i_ch       */dim_i_ch          ,
                    /*uint32_t dim_w_r        */dim_k_r           ,
                    /*uint32_t dim_w_c        */dim_k_c           ,
                    /*uint32_t dim_o_r        */dim_o_r           ,
                    /*uint32_t dim_o_c        */dim_o_c           ,
                    /*uint32_t dim_o_ch       */dim_o_ch          ,
                    /*uint32_t i_r_t          */i_r_num           ,
                    /*uint32_t i_c_t          */i_c_num           ,
                    /*uint32_t i_ch_t         */i_ch_num          ,
                    /*uint32_t o_r_t          */o_r_num           ,
                    /*uint32_t o_c_t          */o_c_num           ,
                    /*uint32_t o_ch_t         */o_ch_num          ,
                    /*int32_t  i_r_s          */i_r_s             ,
                    /*int32_t  i_c_s          */i_c_s             ,
                    /*uint32_t i_ch_s         */i_ch              ,
                    /*uint32_t o_r_s          */o_r_s             ,
                    /*uint32_t o_c_s          */o_c_s             ,
                    /*uint32_t o_ch_s         */o_ch_s            ,
                    /*uint32_t stride_c       */stride_c          ,
                    /*uint32_t stride_r       */stride_r          ,
                    /*uint32_t zp_u           */zp_u              ,
                    /*uint32_t zp_l           */zp_l              ,
                    /*uint32_t w_sp_addr_start*/w_sp_addr_start   ,
                    /*uint32_t o_sp_addr_start*/0x80000000        ,
                    /*bool     mv_out         */mv_out            ,
                    /*bool     accu           */accu              ,
                    /*bool     mv_in_bias     */mv_in_bias        ,
                    /*uint32_t i_ch_blck      */i_ch_t_blck       ,
                    /*uint32_t o_ch_blck      */o_ch_t_blck       ,
                    /*uint32_t dim_w_r_c      */dim_k_rxc          ,
                    /*bool     full_bias      */full_bias
                    );
                    mv_in_bias = false;
                }
            }
        }
    }
    //printf("+ fence\n");
    // gemmini_fence();
    //printf("- fence\n");
}

int main()
{
     // Test parameters
    uint32_t dim_k_c = KERNEL_C;
    uint32_t dim_k_r = KERNEL_R;
    uint32_t dim_i_c = INPUT_C;
    uint32_t dim_i_r = INPUT_R;
    uint32_t dim_i_ch = INPUT_CH;
    uint32_t dim_o_ch = 8;
    uint32_t stride_c = 1;
    uint32_t stride_r = 1;
  //uint32_t dim_o_c = (dim_i_c - dim_k_c) / stride_c + 1; // Calculate output width
  //uint32_t dim_o_r = (dim_i_r - dim_k_r) / stride_r + 1; // Calculate output height
    uint32_t dim_o_c = 8;
    uint32_t dim_o_r = 8;
    uint32_t zp_l = 0;
  //uint32_t zp_u = zp_l;
    uint32_t zp_u = 0;
    int act = 0;
    float i_scale = 1.0;
    float w_scale = 1.0;
    float b_scale = 1.0;
    float o_scale = 1.0;
    bool full_bias = true;
    size_t relu6_shift = 0;

    printf("cal output o_ch_s %d, o_r %d, o_c %d\n", dim_o_ch, dim_o_r , dim_o_c);
    // Input data
    data_i = (int32_t *)malloc(dim_i_r * dim_i_c * dim_i_ch * sizeof(int32_t));
    if (data_i == NULL) {
        // Handle error
        printf("Memory allocation failed!\n");
        return 1;
    }

    for (int i = 0; i < dim_i_r * dim_i_c * dim_i_ch; i++) {
                data_i[i] = i;
    }

    // Weights
    weight = (int32_t *)malloc(dim_k_r * dim_k_c * dim_i_ch * dim_o_ch * sizeof(int32_t));
    for (uint32_t i = 0; i < dim_k_r * dim_k_c * dim_i_ch * dim_o_ch; i++) {
        weight[i] = i | (0x01<<31);
    }

    // Bias
    int32_t bias[dim_o_ch];
    for (int i = 0; i < dim_o_ch; i++) {
        bias[i] = 0; // Initialize to 0 for simplicity
    }

    // Output data
    int8_t *data_o = (int8_t *)malloc(dim_o_r * dim_o_c * dim_o_ch * sizeof(int8_t));
    if (data_o == NULL) {
        // Handle error
        printf("Memory allocation failed!\n");
        return 1;
    }

    for (int i = 0; i < dim_o_r * dim_o_c * dim_o_ch; i++) {
        data_o[i] = 0; // Initialize to 0
    }

    // Run the convolution
    conv_2d_auto(data_i, weight, bias, data_o, dim_i_c, dim_i_r, dim_i_ch, dim_o_c, dim_o_r, dim_o_ch, dim_k_c, dim_k_r, stride_c, stride_r, zp_l, zp_u, act, i_scale, w_scale, b_scale, o_scale, full_bias, relu6_shift);
    free(data_i);
    free(data_o);
    // Print the output
    // printf("Output data:\n");
    // for (int i = 0; i < dim_o_r; i++) {
    //     for (int j = 0; j < dim_o_c; j++) {
    //         for (int k = 0; k < dim_o_ch; k++) {
    //             printf("%d ", data_o[i * dim_o_c * dim_o_ch + j * dim_o_ch + k]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    return 0;
}
