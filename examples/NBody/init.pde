int NBODY_CONFIG_RANDOM = 0;
int NBODY_CONFIG_SHELL = 1;
int NBODY_CONFIG_EXPAND = 2;

void randomizeBodies(int config, ParticleSystem system, float clusterScale, float velocityScale, int numBodies) {
  if (config == NBODY_CONFIG_RANDOM) {
    float pscale = clusterScale * max(1, numBodies / (1024.0));
    float vscale = velocityScale * pscale;
 
    for (int i = 0; i < system.getSize(); i++) {
      float px, py, pz;
      float d = 2;
      px = py = pz = 0;
      while (1 < d) {
        px = random(-1, 1);
        py = random(-1, 1);
        pz = random(-1, 1);
        d = dist(0, 0, 0, px, py, pz);
      }
      system.setParticle(i, 1, pscale * px, pscale * py, pscale * pz);
    }
    system.updatePositions();
  
    for (int i = 0; i < system.getSize(); i++) {
      float vx, vy, vz;
      float d = 2;
      vx = vy = vz = 0;
      while (1 < d) {
        vx = random(-1, 1);
        vy = random(-1, 1);
        vz = random(-1, 1);
        d = dist(0, 0, 0, vx, vy, vz);
      }      
      system.setVelocity(i, vscale * vx, vscale * vy, vscale * vz);
    }
    system.updateVelocities();
    
  } else if (config == NBODY_CONFIG_SHELL) {
    float pscale = clusterScale;
    float vscale = pscale * velocityScale;
    float inner = 2.5 * pscale;
    float outer = 4.0 * pscale;  
  
    float vel[] = new float[system.getSize() * 3];
    for (int i = 0; i < system.getSize(); i++) {
      float px, py, pz;
      float d = 2;
      px = py = pz = 0;
      while (1 < d) {
        px = random(-1, 1);
        py = random(-1, 1);
        pz = random(-1, 1);
        d = dist(0, 0, 0, px, py, pz);
      }
      system.setParticle(i, 1, px * (inner + (outer - inner) * random(0, 1)), 
                               py * (inner + (outer - inner) * random(0, 1)), 
                               pz * (inner + (outer - inner) * random(0, 1)));
                               
      PVector pos = new PVector(px, py, pz);
      PVector axis = new PVector(0, 0, 1);
      //PVector axis = new PVector(random(-1, 1), random(-1, 1), random(-1, 1));
      if (abs(1 - pos.dot(axis)) < 0.0001) {
        axis.x = pos.y;
        axis.y = pos.x;
        axis.normalize();
      }
      PVector vv = pos.cross(axis);                
      vel[3 * i + 0] = vv.x * vscale;
      vel[3 * i + 1] = vv.y * vscale;
      vel[3 * i + 2] = vv.z * vscale;                
    }
    system.updatePositions();
  
    for (int i = 0; i < system.getSize(); i++) {
      system.setVelocity(i, vel[3 * i + 0], vel[3 * i + 1], vel[3 * i + 2]);
    }
    system.updateVelocities();    
  } else if (config == NBODY_CONFIG_EXPAND) {
    float pscale = clusterScale * max(1, numBodies / (1024.0));
    float vscale = velocityScale * pscale;    
    
    float vel[] = new float[system.getSize() * 3];
    for (int i = 0; i < system.getSize(); i++) {
      float px, py, pz;
      float d = 2;
      px = py = pz = 0;
      while (1 < d) {
        px = random(-1, 1);
        py = random(-1, 1);
        pz = random(-1, 1);
        d = dist(0, 0, 0, px, py, pz);
      }
      system.setParticle(i, 1, pscale * px, pscale * py, pscale * pz);

      vel[3 * i + 0] = px * vscale;
      vel[3 * i + 1] = py * vscale;
      vel[3 * i + 2] = pz * vscale;         
    }
    
    for (int i = 0; i < system.getSize(); i++) {
      system.setVelocity(i, vel[3 * i + 0], vel[3 * i + 1], vel[3 * i + 2]);
    }
    system.updateVelocities();    
  }
}

float evalHermite(float pA, float pB, float vA, float vB, float u) {
  float u2 = u * u; 
  float u3 = u2 * u;
  float B0 = 2 * u3 - 3 * u2 + 1;
  float B1 = -2 * u3 + 3 * u2;
  float B2 = u3 - 2 * u2 + u;
  float B3 = u3 - u;
  return (B0 * pA + B1 * pB + B2 * vA + B3 * vB);
}

void createGaussianMap(int pix[], int N) {
  float M[] = new float[2 * N * N];
    
  float X, Y, Y2, D;
  float inc = 2.0 / N;
  int i = 0;  
  int j = 0;
  Y = -1.0f;
  //float mmax = 0;
  for (int y = 0; y < N; y++, Y += inc) {
    Y2 = Y * Y;
    X = -1.0f;
    for (int x = 0; x < N; x++, X += inc, i +=2, j += 1) {
      D = sqrt(X * X + Y2);
      if (D > 1) D = 1;
      M[i + 1] = M[i] = evalHermite(1.0f, 0, 0, 0, D);
      int val = int(M[i] * 255);
      pix[j] = val << 24 | 255 << 16 | 255 << 8 | 255;
    }
  }
}    

PImage createStarSprite(int N) {
  PImage sprite = createImage(N, N, ARGB);
  sprite.loadPixels();
  createGaussianMap(sprite.pixels, N);  
  sprite.updatePixels();
  return sprite;
}
