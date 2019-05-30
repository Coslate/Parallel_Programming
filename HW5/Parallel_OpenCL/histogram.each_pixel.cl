__kernel void histogram(__global uint8_t* orig_img_d, __global uint32_t *hist_calc_d){
    int thread_id       = get_global_id(0);
    int color_offset    = (thread_id % 4) << 8;
    atomic_inc(&hist_calc_d[color_offset + orig_img_d[thread_id]]);
}
