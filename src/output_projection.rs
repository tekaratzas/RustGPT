use ndarray::Array2;
use rand::prelude::*;

pub struct OutputProjection {
    w_out: Array2<f32>, // Weight matrix
    b_out: Array2<f32>, // Bias vector
}

impl OutputProjection {
    /// Initialize output layer with random weights and zero bias
    pub fn new(embedding_dim: usize, vocab_size: usize) -> Self {
        let mut rng = rand::rng();
        OutputProjection {
            w_out: Array2::from_shape_fn((embedding_dim, vocab_size), |_| rng.random_range(-0.1..0.1)),
            b_out: Array2::zeros((1, vocab_size)),
        }
    }

    /// Forward pass: project embeddings to vocab logits
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        input.dot(&self.w_out) + &self.b_out
    }
}