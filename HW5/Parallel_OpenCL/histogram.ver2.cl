__kernel void histogram(__global unsigned char *orig_img_d, __global unsigned int *hist_calc_d, const int height, const int width){
    int thread_id_x       = get_global_id(0);
    int thread_id_y       = get_global_id(1);
    int thread_id_z       = get_global_id(2);

    if(thread_id_x < width && thread_id_y < height){
        atomic_inc(&hist_calc_d[orig_img_d[(thread_id_y * width + thread_id_x)*4]]);
        atomic_inc(&hist_calc_d[256 + orig_img_d[(thread_id_y * width + thread_id_x)*4] + 1]);
        atomic_inc(&hist_calc_d[512 + orig_img_d[(thread_id_y * width + thread_id_x)*4] + 2]);
    }
}

__kernel void histogram_tile(__global unsigned char *orig_img_d, __global unsigned int *hist_calc_tile_d, const int height, const int width){
    int thread_id_x       = get_global_id(0);
    int thread_id_y       = get_global_id(1);
    int thread_id_z       = get_global_id(2);

    int local_size        = (int)get_local_size(0) * (int)get_local_size(1);
    int group_indx        = (get_group_id(1) * get_num_groups(0) + get_group_id(0)) * 768;

    int tid               = get_local_id(1) * get_local_size(0) + get_local_id(0);
    int j                 = 768;
    int indx              = 0;

    __local unsigned int tmp_histogram[768];

    while( j > 0 ){
        if(tid < j){
            tmp_histogram[indx+tid] = 0;
        }
        
        j -= local_size;
        indx += local_size;
    }

    barrier( CLK_LOCAL_MEM_FENCE );

    if(thread_id_x < width && thread_id_y < height){
        atomic_inc(&tmp_histogram[orig_img_d[(thread_id_y * width + thread_id_x)*4]]);
        atomic_inc(&tmp_histogram[256 + orig_img_d[(thread_id_y * width + thread_id_x)*4 + 1]]);
        atomic_inc(&tmp_histogram[512 + orig_img_d[(thread_id_y * width + thread_id_x)*4 + 2]]);
    }

    barrier( CLK_LOCAL_MEM_FENCE );

    if( tid < 768 ){
        hist_calc_tile_d[group_indx + tid] = tmp_histogram[tid];
    }

}

__kernel void histogram_merge(__global unsigned int *hist_calc_tile_d, __global unsigned int *hist_calc_d, const int total_groups_tile){
    int tid = (int)get_global_id(0);
    int group_indx = 768;
    unsigned int tmp_histo_reg = hist_calc_tile_d[tid];

    for( int i=1; i < total_groups_tile; ++i){
             tmp_histo_reg += hist_calc_tile_d[group_indx+tid];
             group_indx += 768;
    }
    
    hist_calc_d[tid] = tmp_histo_reg;
}
