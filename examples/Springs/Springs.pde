import codeanticode.clphysics.*;

final float NODE_SIZE = 10;
final float EDGE_LENGTH = 20;
final float EDGE_STRENGTH = 0.4;
final float DAMPING_STRENGHT = 0.2;
final float SPACER_STRENGTH = 100;

ParticleSystem system;
float scale = 1;
float centroidX = 0;
float centroidY = 0;

float centerX = 0;
float centerY = 0;
float zoom = 500;
float newZoom = 500;

int numBodies = 2500;
boolean start = false;

PGraphics canvas;

void setup() {
  size(720, 360, P3D);
  canvas = createGraphics(width, height, P3D);
  frameRate(120);    
  
  system = new ParticleSystem(this, ParticleSystem.RUNGE_KUTTA, numBodies);
  system.addSpring(numBodies - 1);  
  
  system.addDrag();
  system.setDragConstant(0.05);
  
  system.addGravity();
  system.setGravityConstant(-SPACER_STRENGTH);
  system.setGravityMinDist(EDGE_LENGTH);  
  
  float x, y, z;  
  float pos[] = new float[3 * numBodies];
  int indices[] = new int[2 * (numBodies - 1)];
  for (int i = 0; i < numBodies; i++) {   
    if (0 < i) { 
      int j = int(random(0, i));
      pos[3 * i + 0] = x = pos[3 * j + 0] + random(-2, 2);
      pos[3 * i + 1] = y = pos[3 * j + 1] + random(-1, 1);
      pos[3 * i + 2] = z = 0;
      system.setParticle(i, 1, x, y, z); 
      system.setSpring(i - 1, i, j, EDGE_STRENGTH, DAMPING_STRENGHT, EDGE_LENGTH);
      indices[2 * (i - 1) + 0] = i;
      indices[2 * (i - 1) + 1] = j;
      println("Making spring between : " + i + " " + j);
    } else {
      pos[3 * i + 0] = x = width/2;
      pos[3 * i + 1] = y = height/2;      
      pos[3 * i + 2] = z = 0;
     system.setParticle(i, 1, x, y, z); 
    }
  }
  system.updatePositions();
  system.updateSpring();
  
//  PShape3D particles = (PShape3D)system.getParticles();
//  particles.initIndices(2 * (numBodies - 1));
//  particles.setIndices(indices);
//  particles.useIndices(false);  
//  particles.setColor(color(0));
  
  println("Done creating random network.");   
}

void draw() {
  canvas.beginDraw();
  canvas.background(255);
    
  if (start) {
    system.tick(1);
    
    zoom += 0.2 * (newZoom - zoom);
    canvas.translate(centerX, centerY, -zoom);
  
//    PShape3D particles = (PShape3D)system.getParticles();
//    canvas.shape(particles);
//    particles.setDrawMode(LINES);
//    particles.useIndices(true);
    canvas.shape(system.getParticles());
//    particles.useIndices(false);
//    particles.setDrawMode(POINTS);  
  }
  canvas.endDraw();    
  image(canvas, 0, 0);
  println(frameRate);
}

void mouseDragged() {  
  centerX += mouseX - pmouseX;
  centerY += mouseY - pmouseY;
}

void keyPressed() {  
  if (key == CODED) {
    if (keyCode == UP) newZoom = zoom + 100;
    else if (keyCode == DOWN) newZoom = zoom - 100;
  } else if (key == ' ') {
    start = true;
  }  
}