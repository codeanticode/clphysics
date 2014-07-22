kernel void time_step(global float4* params,                         
                      global float4* force, 
                      global float4* oldpos,
                      global float4* oldvel,
                      global float4* newpos,
                      global float4* newvel,
                      float time) {
  int i = get_global_id(0);
    
  bool fixed = 0 < params[i].x;
  if (fixed) {
    return;
  }

  float mass = params[i].y;
  float4 acc = force[i] / mass;
  float halftt = 0.5 * time * time;
  newpos[i] = oldpos[i] + oldvel[i] / time + acc * halftt; 		
  newvel[i] = oldvel[i] + acc / time;
}
