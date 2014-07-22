kernel void set_vector(global float4* vec, 
                       float x, float y, float z, float w) {                         
  int i = get_global_id(0);
  vec[i] = (float4)(x, y, z, w);
}
