kernel void copy(global float4* srcpos,
                 global float4* srcvel,
                 global float4* destpos,
                 global float4* destvel) {
  int i = get_global_id(0);
  destpos[i] = srcpos[i];
  destvel[i] = srcvel[i];
}

kernel void update(global float4* params,   
                   global float4* pos0,
                   global float4* vel0,
                   global float4* force1,
                   global float4* vel1,
                   global float4* pos2,
                   global float4* vel2,
                   float dt) {
  int i = get_global_id(0);
    
  bool fixed = 0 < params[i].x;
  if (fixed) {
    return;
  }
  
  float mass = params[i].y;                   
  pos2[i] = pos0[i] + vel1[i] * dt;  
  vel2[i] = vel0[i] + force1[i] * dt / mass;
}

kernel void time_step(global float4* params,
                      global float4* oldpos,
                      global float4* oldvel,
                      global float4* newpos,
                      global float4* newvel,
                      global float4* k1vel,
                      global float4* k2vel,
                      global float4* k3vel,
                      global float4* k4vel,
                      global float4* k1force,
                      global float4* k2force,
                      global float4* k3force,
                      global float4* k4force,
                      float dt) {
  int i = get_global_id(0);
    
  bool fixed = 0 < params[i].x;
  if (fixed) {
    return;
  }
  
  float mass = params[i].y; 
  
  // Update position:
  newpos[i] = oldpos[i] + dt / 6.0f * (k1vel[i] + 2.0f * k2vel[i] + 2.0f * k3vel[i] + k4vel[i]);
  
  // Update velocity:
  newvel[i] = oldvel[i] + dt / (6.0f * mass) * (k1force[i] + 2.0f * k2force[i] + 2.0f * k3force[i] + k4force[i]);
}
