import torch
import flashinfer

class FlashInferBackend():
    def __init__(self, num_qo_head, num_kv_head, head_dim, block_size, dtype, device):
        self.num_qo_head = num_qo_head
        self.num_kv_head = num_kv_head
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.device = device
        self.prefill_workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.decode_workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.prefill_paged_workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(self.prefill_workspace, kv_layout="NHD")
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(self.decode_workspace, kv_layout="NHD",
                                                                       use_tensor_cores=True)
        self.prefill_paged_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.prefill_paged_workspace,
                                                                                   kv_layout="NHD")

    def plan_prefill(self, cu_seqlens) -> flashinfer.BatchPrefillWithRaggedKVCacheWrapper:
        self.prefill_wrapper.plan(cu_seqlens, cu_seqlens, self.num_qo_head, self.num_kv_head,
                             self.head_dim, q_data_type=self.dtype, kv_data_type=self.dtype, causal=True,
                             sm_scale=1.0)   # q 已在 Attention 里预乘 1/sqrt(D)，见 Attention.fi_q_scale
        return self.prefill_wrapper

    def plan_decode(self, kv_indptr, kv_indices, kv_last_page_len) -> flashinfer.BatchDecodeWithPagedKVCacheWrapper:
        self.decode_wrapper.plan(kv_indptr, kv_indices, kv_last_page_len, self.num_qo_head, self.num_kv_head,
                                 self.head_dim, self.block_size, pos_encoding_mode="NONE",
                                 q_data_type=self.dtype, kv_data_type=self.dtype, sm_scale=1.0)
        return self.decode_wrapper

    def plan_prefill_paged(self, qo_indptr, kv_indptr, kv_indices, kv_last_page_len):
        self.prefill_paged_wrapper.plan(qo_indptr, kv_indptr, kv_indices, kv_last_page_len, self.num_qo_head,
                                        self.num_kv_head, self.head_dim, self.block_size, pos_encoding_mode="NONE",
                                        q_data_type=self.dtype, kv_data_type=self.dtype, causal=True, sm_scale=1.0)
        return self.prefill_paged_wrapper