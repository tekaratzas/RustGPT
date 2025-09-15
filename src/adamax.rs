use ndarray::Array2;

pub struct AdaMax {
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    timestep: usize,
    pub m: Array2<f32>,
    pub u: Array2<f32>,
}

impl AdaMax {
    pub fn new(shape: (usize, usize)) -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            timestep: 0,
            m: Array2::zeros(shape),
            u: Array2::zeros(shape),
        }
    }

    pub fn step(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>, lr: f32) {
        self.timestep += 1;

        // Update biased first moment estimate
        self.m = &self.m * self.beta1 + &(grads * (1.0 - self.beta1));

        // Update the exponentially weighted infinity norm
        self.u = self.u.zip_with(&grads.mapv(f32::abs), |u_val, grad_val| (self.beta2 * u_val).max(*grad_val));

        // Bias-corrected first moment estimate
        let m_hat = &self.m / (1.0 - self.beta1.powi(self.timestep as i32));
      
        let update = &m_hat / (&self.u + self.epsilon);
        *params -= &(update * lr);
    }
}
