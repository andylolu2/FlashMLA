#pragma once

#include <cuda_runtime_api.h>

struct __align__(4*8) DecodingSchedMeta {
    int begin_req_idx, end_req_idx;     // Both inclusive
    int begin_block_idx, end_block_idx; // Inclusive, exclusive
    int begin_split_idx;
    int is_first_req_splitted, is_last_req_splitted;
    int _pad[1];
};
static constexpr int DecodingSchedMetaSize = sizeof(DecodingSchedMeta);

struct DenseAttnDecodeParams { // TODO Change name to DenseAttnDecodeParams
    using index_t = int64_t;

    int b;              // batch size
    int s_q;
    int q_seq_per_hk;   // The number of q(s) per KV head, = h_q / h_k * s_q
    int d, d_v;         // K/V dimension
    int h_q, h_k;       // The number of Q/K heads
    int num_blocks;     // Number of blocks in total
    int q_head_per_hk;  // The number of q_head(s) per KV head, = h_q / h_k
    bool is_causal;
    float scale_softmax, scale_softmax_log2;
    
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ o_ptr;
    float *__restrict__ softmax_lse_ptr;

    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t o_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t o_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t o_head_stride;

    int *__restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;
    int *__restrict__ seqlens_k_ptr;

    DecodingSchedMeta *__restrict__ tile_scheduler_metadata_ptr;
    int num_sm_parts;
    int *__restrict__ num_splits_ptr;

    int total_num_splits;
    float *__restrict__ softmax_lseaccum_ptr;
    float *__restrict__ oaccum_ptr;

    cudaStream_t stream;
};

struct CombineParams {
    int b, s_q, h_q, d_v;

    float* __restrict__ lse;    // [b, s_q, h_q]
    void* __restrict__ out;   // [b, s_q, h_q, d_v]
    int stride_lse_b, stride_lse_s_q;
    int stride_o_b, stride_o_s_q, stride_o_h_q;

    float* __restrict__ lse_accum;  // [num_splits, s_q, h_q]
    float* __restrict__ o_accum;    // [num_splits, s_q, h_q, d_v]
    int stride_lse_accum_split, stride_lse_accum_s_q;
    int stride_o_accum_split, stride_o_accum_s_q, stride_o_accum_h_q;

    DecodingSchedMeta* __restrict__ tile_scheduler_metadata_ptr; // [num_sm_parts, ], contiguous
    int* __restrict__ num_splits_ptr; // [batch_size+1, ], contiguous
    int num_sm_parts;

    float* attn_sink;  // [h_q], may be nullptr

    cudaStream_t stream;
};

struct GetDecodeSchedMetaParams {
    int b;  // batch size
    int s_q;
    int block_size_n;
    int fixed_overhead_num_blocks;

    int *__restrict__ seqlens_k_ptr;    // Only necessary for dense attention

    DecodingSchedMeta *__restrict__ tile_scheduler_metadata_ptr;
    int *__restrict__ num_splits_ptr;
    int num_sm_parts;

    cudaStream_t stream;
};
