// OpenCL N-Body Physics Simulation:
// http://developer.download.nvidia.com/compute/cuda/3_0/sdk/website/OpenCL/website/samples.html

// Adapted from N-Body Simulation tutorial by Brown Deer Technology:
// http://www.browndeertechnology.com/docs/BDT_OpenCL_Tutorial_NBody-rev3.html
// The Chapter 31 from GPU Gems 3 (Fast N-Body Simulation with CUDA) is also
// useful. Available online:
// http://http.developer.nvidia.com/GPUGems3/gpugems3_ch31.html

//import processing.opengl2.*;
import codeanticode.clphysics.*;
//import codeanticode.syphon.*;

ParticleSystem system;
PImage sprite;
PGraphics canvas;
//SyphonServer syphon;

int numBodies = 5000;

void setup() {
  size(720, 360, P3D);
  canvas = createGraphics(width, height, P3D);
  frameRate(120);
  
  system = new ParticleSystem(this, ParticleSystem.MODIFIED_EULER, numBodies);
  system.addGravity();
  //PShape3D particles = (PShape3D)system.getParticles();
  //sprite = loadImage("particle.png");  
  sprite = createStarSprite(20);
  
//  particles.setTexture(sprite);
//  particles.setSpriteSize(10, 400, QUADRATIC);
//  particles.autoBounds(false);
//  particles.setColor(color(255));
  
  randomizeBodies(NBODY_CONFIG_RANDOM, system, 100, 0.0, numBodies);
  
//  syphon = new SyphonServer(this);
}

void draw() {  
  system.tick(1);
 
  canvas.beginDraw();
  canvas.background(180);  
  canvas.translate(width/2, height/2);
  canvas.hint(DISABLE_DEPTH_MASK);
  canvas.shape(system.getParticles());
  canvas.hint(ENABLE_DEPTH_MASK);
  canvas.endDraw();
  
  image(canvas, 0, 0);
//  syphon.sendImage(canvas);
  
  println(frameRate);
}