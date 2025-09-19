use ndarray::Array2;

pub struct Adam {
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    timestep: usize,
    pub m: Array2<f32>,
    pub v: Array2<f32>,
}

impl Adam {
    pub fn new(shape: (usize, usize)) -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            timestep: 0,
            m: Array2::zeros(shape),
            v: Array2::zeros(shape),
        }
    }

    pub fn step(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>, lr: f32) {
        self.timestep += 1;
        self.m = &self.m * self.beta1 + &(grads * (1.0 - self.beta1));
        self.v = &self.v * self.beta2 + &(grads.mapv(|x| x * x) * (1.0 - self.beta2));

        let m_hat = &self.m / (1.0 - self.beta1.powi(self.timestep as i32));
        let v_hat = &self.v / (1.0 - self.beta2.powi(self.timestep as i32));

        let update = m_hat / (v_hat.mapv(|x| x.sqrt()) + self.epsilon); // Removed unnecessary clone

        *params -= &(update * lr);
    }
}
