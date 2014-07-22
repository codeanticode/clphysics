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

public class EulerIntegrator implements Integrator {
  public void step(float t) {
    
  }  
  /*
  ParticleSystem s;

  public EulerIntegrator(ParticleSystem s) {
    this.s = s;
  }

  public void step(float t) {
    s.clearForces();
    s.applyForces();

    for (int i = 0; i < s.numberOfParticles(); i++) {
      Particle p = (Particle) s.getParticle(i);
      if (p.isFree()) {
        p.velocity().add(p.force().x() / (p.mass() * t),
            p.force().y() / (p.mass() * t), p.force().z() / (p.mass() * t));
        p.position().add(p.velocity().x() / t, p.velocity().y() / t,
            p.velocity().z() / t);
      }
    }

  }
*/
}
