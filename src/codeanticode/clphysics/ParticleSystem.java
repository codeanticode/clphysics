/*
  Copyright (c) 2011-14 Andres Colubri

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General
  Public License along with this library; if not, write to the
  Free Software Foundation, Inc., 59 Temple Place, Suite 330,
  Boston, MA  02111-1307  USA
*/

package codeanticode.clphysics;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLPlatform;
import com.jogamp.opencl.CLProgram;
import com.jogamp.opencl.gl.CLGLBuffer;
import com.jogamp.opencl.gl.CLGLContext;

import static com.jogamp.opencl.CLMemory.Mem.READ_ONLY;
import static com.jogamp.opencl.CLMemory.Mem.READ_WRITE;
import static com.jogamp.opencl.CLMemory.Mem.USE_BUFFER;
import static java.lang.Math.min;
import static java.lang.System.out;

import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;

import processing.core.PApplet;
import processing.core.PConstants;
import processing.core.PShape;
import processing.opengl.PGL;
import processing.opengl.PGraphicsOpenGL;
import processing.opengl.PShapeOpenGL;

public class ParticleSystem {
  public static final int RUNGE_KUTTA = 0;
  public static final int MODIFIED_EULER = 1;

  protected static final float DEFAULT_GRAVITY = 0;
  protected static final float DEFAULT_DRAG = 0.001f;

  DragForce drag = null;
  Gravity gravity = null;
  Spring spring = null;

  int type = PConstants.POINTS;
  Integrator integrator;

  //Vector3D gravity;
  //float drag;

  boolean hasDeadParticles = false;

  PApplet p;
  PGraphicsOpenGL pg;
  PShapeOpenGL particles;
  
  // Context for CL-GL sharing. Not working yet :-(
  CLGLContext contextCLGL;
  CLGLBuffer<?> clVertices;
  
  CLContext context;
  CLDevice device;
  CLCommandQueue queue;
  
  CLBuffer<FloatBuffer> positions[];
  CLBuffer<FloatBuffer> velocities[];
  CLBuffer<FloatBuffer> parameters;
  CLBuffer<FloatBuffer> forces;
  
  int currRead;
  int currWrite; 

  int numVertPerPt;
  int numParticles = 10000;  
  int globalWorkSize;
  int localWorkSize;  

  CLProgram program;
  CLKernel kernel;  
  
  boolean clglInterop;
  
  public ParticleSystem(PApplet parent, int integr, int npart) {
    this.p = parent;
    
    numParticles = npart;

    initCL();
    
    if (integr == RUNGE_KUTTA) {
      integrator = new RungeKuttaIntegrator(this);
    } else if (integr == MODIFIED_EULER) {
      integrator = new ModifiedEulerIntegrator(this);
    } else {
      throw new RuntimeException("Unrecognized integrator");      
    }    
  }

  public void dispose() {
    finishCL();
  }

  public void addDrag() {
    drag = new DragForce(this);
    drag.turnOn();
  }
  
  public void addGravity() {
    gravity = new Gravity(this);    
    gravity.turnOn();
  }

  public void addSpring(int n) {
    spring = new Spring(this, n);    
    spring.turnOn();
  }  
  
  public int getSize() {
    return numParticles;
  }
  
  @SuppressWarnings("unchecked")
  protected void initCL() {
    pg = (PGraphicsOpenGL)p.g;
    //PApplet.println(ogl2.getContext());

    // Find GL compatible device:
    CLDevice[] devices = CLPlatform.getDefault().listCLDevices(); 
    CLDevice dev = null;    
    int mx = 0;
    for (CLDevice d : devices) {
      PApplet.println(d.getName() + " " + d.isGLMemorySharingSupported() + " " + d.getMaxComputeUnits());
      if (mx < d.getMaxComputeUnits()) {
        mx = d.getMaxComputeUnits();
        dev = d;
      } 
    }    
    if (null == dev) { 
      throw new RuntimeException("couldn't find any CL/GL memory sharing devices .."); 
    }
    
    PApplet.println(dev);

    context = CLContext.create(dev);
    PApplet.println(context);
    clglInterop = false;
    
    /*
    // Create OpenCL context before creating any OpenGL objects 
    // you want to share with OpenCL (AMD driver requirement):
    PApplet.println(ogl2.getContext());
    contextCLGL = CLGLContext.create(ogl2.getContext(), dev);    
    out.println("created " + contextCLGL);
    clglInterop = true;
*/
    
    // Select fastest device:
    device = context.getMaxFlopsDevice();
    out.println("using " + device);
    
    // Create command queue on device:
    queue = device.createCommandQueue();    
    
    // Length of arrays to process
    
    // 4 components per particle, for memory alignment reasons.
    int n = numParticles;
    int maxCU = device.getMaxComputeUnits();
    int maxGS = device.getMaxWorkGroupSize();
    int[] maxItemSiz = device.getMaxWorkItemSizes();
    
    PApplet.println("Max compute units: " + maxCU);
    PApplet.println("Max work group size: " + maxGS);
    PApplet.println("Max work group sizes: " + maxItemSiz[0] + "x" + maxItemSiz[1] + "x" + maxItemSiz[2]);
    
    localWorkSize = min(maxGS, 256); // Local work size dimensions      
    globalWorkSize = roundUp(localWorkSize, n);             // rounded up to the nearest multiple of the localWorkSize     

    PApplet.println("local work size: " + localWorkSize);
    PApplet.println("global work size: " + globalWorkSize);
    
    if (pg.strokeCap == PConstants.ROUND) {
      int   MIN_POINT_ACCURACY    = 20;
      int   MAX_POINT_ACCURACY    = 200;
      float POINT_ACCURACY_FACTOR = 10.0f;    
      numVertPerPt = PApplet.min(MAX_POINT_ACCURACY, PApplet.max(MIN_POINT_ACCURACY,
                                 (int) (PConstants.TWO_PI * pg.strokeWeight /
                                 POINT_ACCURACY_FACTOR))) + 1; 
    } else {
      numVertPerPt = 5;
    }
    
    //particles = new PShape3D(p, numParticles, PShape3D.newParameters(type, PShape3D.STREAM));
    particles = (PShapeOpenGL)pg.createShape();
    particles.glUsage = PGL.STREAM_DRAW;
    particles.beginShape(PConstants.POINTS);
    for (int i = 0; i < numParticles; i++) {
      particles.vertex(0, 0);
    }
    particles.endShape();
    
    //clVertices = context.createFromGLBuffer(particles2.glVertexBufferID, CLGLBuffer.Mem.READ_WRITE);
    
    positions = new CLBuffer[2];
    velocities = new CLBuffer[2]; 
    for (int i = 0; i < 2; i++) {
      positions[i] = context.createFloatBuffer(globalWorkSize * 4, READ_WRITE);
      velocities[i] = context.createFloatBuffer(globalWorkSize * 4, READ_WRITE);          
    }    
    forces = context.createFloatBuffer(globalWorkSize * 4, READ_WRITE); 
    
    // The compute device should use the host buffer as the storage bits for
    // this float buffer. In this way, modifications in the host side are 
    // automatically transferred to the device.
    parameters = context.createFloatBuffer(globalWorkSize * 4, READ_ONLY, USE_BUFFER);
    
    InputStream stream = this.getClass().getResourceAsStream("system.cl");    
    try {
      program = context.createProgram(stream).build();
    } catch (IOException e) {
      e.printStackTrace();
    }
    kernel = program.createCLKernel("set_vector");    
    
    initSystem();
    
    currRead = 0;
    currWrite = 1;
  }

  protected CLBuffer<FloatBuffer> getReadPositions() {
    return positions[currRead];
  }

  protected CLBuffer<FloatBuffer> getReadVelocities() {
    return velocities[currRead];
  }
    
  protected void initSystem() {
    float x = 0;
    float y = 0;
    float z = 0;
    float w = 0;
    
    kernel.setArg(0, positions[0]);
    kernel.setArg(1, x);
    kernel.setArg(2, y);
    kernel.setArg(3, z);
    kernel.setArg(4, w);
    queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();

    kernel.setArg(0, positions[1]);
    kernel.setArg(1, x);
    kernel.setArg(2, y);
    kernel.setArg(3, z);
    kernel.setArg(4, w);
    queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();

    kernel.setArg(0, velocities[0]);
    kernel.setArg(1, x);
    kernel.setArg(2, y);
    kernel.setArg(3, z);
    kernel.setArg(4, w);
    queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();
    
    kernel.setArg(0, velocities[1]);
    kernel.setArg(1, x);
    kernel.setArg(2, y);
    kernel.setArg(3, z);
    kernel.setArg(4, w);
    queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();

    FloatBuffer buf = parameters.getBuffer();
    for (int i = 0; i < globalWorkSize; i++) {
      buf.put(0); buf.put(1); buf.put(0); buf.put(0);     
    }
    buf.rewind();
  }
  
  protected void swapReadWrite() {
    int temp = currRead;
    currRead =  currWrite;
    currWrite = temp;    
  }
  
  protected void finishCL() {
    context.release();
  }

  protected int roundUp(int groupSize, int globalSize) {
    int r = globalSize % groupSize;
    if (r == 0) {
      return globalSize;
    } else {
      return globalSize + groupSize - r;
    }
  }
  
  public final void setIntegrator(int integrator) {
    /*
    switch (integrator) {
    case RUNGE_KUTTA:
      this.integrator = new RungeKuttaIntegrator(this);
      break;
    case MODIFIED_EULER:
      this.integrator = new ModifiedEulerIntegrator(this);
      break;
    }
    */
  }

  /*
  public final void setGravity(float x, float y, float z) {
    gravity.set(x, y, z);
  }

  // default down gravity
  public final void setGravity(float g) {
    gravity.set(0, g, 0);
  }

  public final void setDrag(float d) {
    drag = d;
  }
*/
  
  public final void tick() {
    tick(1);
  }

  public final void tick(float t) {
    integrator.step(t);
        
    if (!clglInterop) {
      // If CL-GL inter-operability is not available, then we need to 
      // copy the positions back to the host memory, and then transfer
      // them to the VBO in the PShape3D object.
      
      queue.putReadBuffer(positions[currWrite], true);
      FloatBuffer buf = positions[currWrite].getBuffer();
      float pos[] = {0, 0, 0};
      float[] vertices = particles.getTessellation(PConstants.POINTS, PShapeOpenGL.POSITION);
      for (int i = 0; i < numParticles; i++) {
        buf.position(4 * i);
        buf.get(pos, 0, 3);        
        for (int j = 0; j < numVertPerPt; j++) {
          int n = i * numVertPerPt + j;          
          vertices[4 * n + 0] = pos[0];
          vertices[4 * n + 1] = pos[1];
          vertices[4 * n + 2] = pos[2];          
        }
      }
      buf.rewind();
    }
    
    swapReadWrite();
  }

  public void draw() {
    p.shape(particles);
  }
  
  public PShape getParticles() {
    return particles;
  }
  
  public void setParticle(int i, float mass, float x, float y, float z) {
    FloatBuffer buf; 
    
    float data[] = {x, y, z, 0};    
    buf = positions[currRead].getBuffer();  
    buf.position(4 * i);
    buf.put(data, 0, 4);    
    buf.rewind();
    
    buf = parameters.getBuffer();
    buf.put(4 * i + 1, mass);
    buf.rewind();
  }
  
  public void setVelocity(int i, float vx, float vy, float vz) {
    FloatBuffer buf; 
    
    float data[] = {vx, vy, vz, 0};    
    buf = velocities[currRead].getBuffer();  
    buf.position(4 * i);
    buf.put(data, 0, 4);    
    buf.rewind();
  }
  
  public void updateVelocities() {
    queue.putWriteBuffer(velocities[currRead], true);          
  }  
  
  public void setParticle(int i) {
    setParticle(i, 1.0f, 0f, 0f, 0f);
  } 
  
  public void fixParticle(int i) {
    FloatBuffer buf = parameters.getBuffer();
    buf.put(4 * i + 0, 1);  
  }

  public void freeParticle(int i) {
    FloatBuffer buf = parameters.getBuffer();
    buf.put(4 * i + 0, 0);      
  }
  
  public boolean isParticleFixed(int i) {
    FloatBuffer buf = parameters.getBuffer();
    return 0 < buf.get(4 * i + 0);    
  }

  public boolean isParticleFree(int i) {
    return !isParticleFree(i); 
  }  
  
  public void updatePositions() {
    queue.putWriteBuffer(positions[currRead], true);          
  }

  public void setSpring(int n, int i, int j, float ks, float d, float r) {
    spring.setPair(n, i, j, ks, d, r);
  }
  
  public void updateSpring() {
    spring.update();
  }
  
  public void setGravityConstant(float c) {
    gravity.setGravitationalConstant(c);
  }

  public void setGravityMinDist(float d) {
    gravity.setMinPairDistance(d);
  }
  
  public void setDragConstant(float d) {
    drag.setDragConstant(d);
  }
    
  public final void clear() {
    /*
    particles2.clear();
    
    pairForces.clear();
    
    
    particles.clear();
    springs.clear();
    attractions.clear();
    */
  }
  
  
  
  /*
  public final Attraction makeAttraction(Particle a, Particle b, float k,
      float minDistance) {
    Attraction m = new Attraction(a, b, k, minDistance);
    attractions.add(m);
    return m;
  }  
  public final Particle makeParticle(float mass, float x, float y, float z) {
    Particle p = new Particle(mass);
    p.position().set(x, y, z);
    particles.add(p);
    return p;
  }

  public final Particle makeParticle() {
    return makeParticle(1.0f, 0f, 0f, 0f);
  }
 */
  
  
  protected void clearForces() {
    clearForces(forces);
    
    /*
    // Read back forces into host memory for checking purposes:
    queue.putReadBuffer(forces, true);
    for(int i = 0; i < 10; i++)
      out.print(forces.getBuffer().get() + ", ");
    out.println("...; " + forces.getBuffer().remaining() + " more");
    forces.getBuffer().rewind();
    */
  }
  
  protected void clearForces(CLBuffer<FloatBuffer> force) {
    float x = 0;
    float y = 0;
    float z = 0;
    float w = 0;
    
    kernel.setArg(0, force);
    kernel.setArg(1, x);
    kernel.setArg(2, y);
    kernel.setArg(3, z);
    kernel.setArg(4, w);
    queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize).finish();        
  }
  
  protected void applyForces() {
    if (spring != null) spring.apply(positions[currRead], velocities[currRead], forces);
    if (gravity != null) gravity.apply(positions[currRead], velocities[currRead], forces);
    if (drag != null) drag.apply(positions[currRead], velocities[currRead], forces);
    
    
    /*
    if (!gravity.isZero()) {
      for (int i = 0; i < particles.size(); ++i) {
        Particle p = (Particle) particles.get(i);
        p.force.add(gravity);
      }
    }

    for (int i = 0; i < particles.size(); ++i) {
      Particle p = (Particle) particles.get(i);
      p.force.add(p.velocity.x() * -drag, p.velocity.y() * -drag,
          p.velocity.z() * -drag);
    }

    for (int i = 0; i < springs.size(); i++) {
      Spring f = (Spring) springs.get(i);
      f.apply();
    }

    for (int i = 0; i < attractions.size(); i++) {
      Attraction f = (Attraction) attractions.get(i);
      f.apply();
    }

    for (int i = 0; i < customForces.size(); i++) {
      PairForce f = (PairForce) customForces.get(i);
      f.apply();
    }
    */
  }

  protected void applyForces(CLBuffer<FloatBuffer> pos,
                             CLBuffer<FloatBuffer> vel,
                             CLBuffer<FloatBuffer> force) {
    if (spring != null) spring.apply(pos, vel, force);
    if (gravity != null) gravity.apply(pos, vel, force);
    if (drag != null) drag.apply(pos, vel, force);
  }
  
  /*
  protected final void clearForces() {
    Iterator i = particles.iterator();
    while (i.hasNext()) {
      Particle p = (Particle) i.next();
      p.force.clear();
    }
  }

  public final int numberOfParticles() {
    return particles.size();
  }

  public final int numberOfSprings() {
    return springs.size();
  }

  public final int numberOfAttractions() {
    return attractions.size();
  }

  public final Particle getParticle(int i) {
    return (Particle) particles.get(i);
  }

  public final Spring getSpring(int i) {
    return (Spring) springs.get(i);
  }

  public final Attraction getAttraction(int i) {
    return (Attraction) attractions.get(i);
  }

  public final void addCustomForce(PairForce f) {
    customForces.add(f);
  }

  public final int numberOfCustomForces() {
    return customForces.size();
  }

  public final PairForce getCustomForce(int i) {
    return (PairForce) customForces.get(i);
  }

  public final PairForce removeCustomForce(int i) {
    return (PairForce) customForces.remove(i);
  }

  public final void removeParticle(Particle p) {
    particles.remove(p);
  }

  public final Spring removeSpring(int i) {
    return (Spring) springs.remove(i);
  }

  public final Attraction removeAttraction(int i) {
    return (Attraction) attractions.remove(i);
  }

  public final void removeAttraction(Attraction s) {
    attractions.remove(s);
  }

  public final void removeSpring(Spring a) {
    springs.remove(a);
  }

  public final void removeCustomForce(PairForce f) {
    customForces.remove(f);
  }
*/
}