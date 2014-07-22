kernel void drag_force(global float4* params,
                       global float4* vel,                          
                       global float4* force,
                       float dconst) {                         
  int i = get_global_id(0);
    
  bool fixed = 0 < params[i].x;
  if (fixed) {
    return;
  }

  force[i] += -dconst * vel[i];
}
