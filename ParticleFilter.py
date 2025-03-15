import numpy as np

class ParticleFilter:
    def __init__(self, num_particles, x_init, process_noise, measurement_noise):
        self.num_particles = num_particles 
        self.particles = np.random.normal(x_init, 1, num_particles) 
        self.weights = np.ones(num_particles) / num_particles  
        self.process_noise = process_noise 
        self.measurement_noise = measurement_noise 

    def predict(self):
        self.particles += np.random.normal(0, self.process_noise, self.num_particles)

    def update(self, measurement):
        self.weights *= self.gaussian(measurement, self.measurement_noise)
        self.weights += 1e-300 
        self.weights /= np.sum(self.weights)  

    def gaussian(self, measurement, noise):
        return np.exp(-0.5 * ((self.particles - measurement) / noise) ** 2) / (noise * np.sqrt(2 * np.pi))

    def resample(self):
        indices = np.random.choice(range(self.num_particles), size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

    def estimate(self):
        return np.sum(self.particles * self.weights)

