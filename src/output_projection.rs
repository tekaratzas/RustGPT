use ndarray::{Array2, Axis};
use rand_distr::{Distribution, Normal};

use crate::{adam::Adam, llm::Layer};

pub struct OutputProjection {
    pub w_out: Array2<f32>, // Weight matrix
    pub b_out: Array2<f32>, // Bias vector
    pub optimizer: Adam,
    pub cached_input: Option<Array2<f32>>,
}

impl OutputProjection {
    /// Initialize output layer with random weights and zero bias
    pub fn new(embedding_dim: usize, vocab_size: usize) -> Self {
        let mut rng = rand::rng();
        // Xavier/He initialization: std = sqrt(2 / fan_in)
        let std = (2.0 / embedding_dim as f32).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        OutputProjection {
            w_out: Array2::from_shape_fn((embedding_dim, vocab_size), |_| normal.sample(&mut rng)),
            b_out: Array2::zeros((1, vocab_size)),
            optimizer: Adam::new((embedding_dim, vocab_size)),
            cached_input: None,
        }
    }
}

impl Layer for OutputProjection {
    fn layer_type(&self) -> &str {
        "OutputProjection"
    }

    /// Forward pass: project embeddings to vocab logits
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // input shape is [sequence_length, embedding_dim]
        self.cached_input = Some(input.clone());
        input.dot(&self.w_out) + &self.b_out // shape is [sequence_length, vocab_size]
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        // grads shape is [sequence_length, vocab_size]
        let input = self.cached_input.as_ref().unwrap();
        let grad_w_out = input.t().dot(grads);
        let grad_b_out = grads.mean_axis(Axis(0)).unwrap();

        let grad_input = grads.dot(&self.w_out.t());

        self.optimizer.step(&mut self.w_out, &grad_w_out, lr);
        self.b_out -= &(lr * &grad_b_out);

        grad_input
    }

    fn parameters(&self) -> usize {
        self.w_out.len() + self.b_out.len()
    }
}
