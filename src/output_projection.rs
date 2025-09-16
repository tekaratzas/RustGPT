use ndarray::{Array2, Axis};
use rand_distr::{Normal, Distribution};

use crate::optimizer::{new_optimizer, Optimizer, OptimizerType};
use crate::llm::Layer;

pub struct OutputProjection {
   pub w_out: Array2<f32>, // Weight matrix
   pub b_out: Array2<f32>, // Bias vector
   pub cached_input: Option<Array2<f32>>,
   pub optimizer_w: Box<dyn Optimizer>,
   pub optimizer_b: Box<dyn Optimizer>,
}


impl OutputProjection {
    // The 'new' function now accepts the optimizer type to be used.
    pub fn new(embedding_dim: usize, vocab_size: usize, optimizer_type: &OptimizerType) -> Self {
        let mut rng = rand::rng();
        let std = (2.0 / embedding_dim as f32).sqrt();
        let normal = Normal::new(0.0, std).unwrap();
        
        OutputProjection {
            w_out: Array2::from_shape_fn((embedding_dim, vocab_size), |_| normal.sample(&mut rng)),
            b_out: Array2::zeros((1, vocab_size)),
            cached_input: None,
            // Use the factory to create the chosen optimizer for both weights and biases.
            optimizer_w: new_optimizer(optimizer_type, (embedding_dim, vocab_size)),
            optimizer_b: new_optimizer(optimizer_type, (1, vocab_size)),
        }
    }
}

impl Layer for OutputProjection {
    fn layer_type(&self) -> &str {
        "OutputProjection"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.cached_input = Some(input.clone());
        input.dot(&self.w_out) + &self.b_out
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let input = self.cached_input.as_ref().unwrap();
        let grad_w_out = input.t().dot(grads);
        // Note: sum_axis is often used for bias gradients instead of mean_axis.
        let grad_b_out = grads.sum_axis(Axis(0)).insert_axis(Axis(0));

        let grad_input = grads.dot(&self.w_out.t());

        // Use the dedicated optimizers to update both weights and biases.
        // This replaces the manual update for b_out, making the code more consistent.
        self.optimizer_w.step(&mut self.w_out, &grad_w_out, lr);
        self.optimizer_b.step(&mut self.b_out, &grad_b_out, lr);

        grad_input
    }
}