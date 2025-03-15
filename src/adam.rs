use ndarray::Array2;

pub struct Adam {
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    m: Array2<f32>, // Momentum term
    v: Array2<f32>, // Adaptive learning rate term
    t: usize, // Time step
}

impl Adam {
    /// Initializes Adam with zero momentum and variance
    pub fn new(shape: (usize, usize)) -> Self {
        Adam {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m: Array2::zeros(shape),
            v: Array2::zeros(shape),
            t: 0,
        }
    }

    /// Updates weights using Adam optimization
    pub fn update(&mut self, weights: &mut Array2<f32>, grads: &Array2<f32>, lr: f32) {
        self.t += 1; // Increase time step

        // Compute biased first moment estimate (momentum)
        self.m = &self.m * self.beta1 + grads * (1.0 - self.beta1);

        // Compute biased second moment estimate (adaptive learning rate)
        self.v = &self.v * self.beta2 + grads.mapv(|g| g * g) * (1.0 - self.beta2);

        // Bias correction
        let m_hat = &self.m / (1.0 - self.beta1.powi(self.t as i32));
        let v_hat = &self.v / (1.0 - self.beta2.powi(self.t as i32));

        // Apply weight update
        *weights -= &(m_hat * lr / (v_hat.mapv(f32::sqrt) + self.epsilon));
    }
}