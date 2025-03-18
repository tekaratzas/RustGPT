use ndarray::Array2;
use rand::prelude::*;

use crate::{adam::Adam, llm::Layer};

pub struct OutputProjection {
   pub w_out: Array2<f32>, // Weight matrix
   pub b_out: Array2<f32>, // Bias vector
   pub optimizer: Adam,
}

impl OutputProjection {
    /// Initialize output layer with random weights and zero bias
    pub fn new(embedding_dim: usize, vocab_size: usize) -> Self {
        let mut rng = rand::rng();
        OutputProjection {
            w_out: Array2::from_shape_fn((embedding_dim, vocab_size), |_| rng.random_range(-0.1..0.1)),
            b_out: Array2::zeros((1, vocab_size)),
            optimizer: Adam::new((embedding_dim, vocab_size), 0.001),
        }
    }
}

impl Layer for OutputProjection {
    /// Forward pass: project embeddings to vocab logits
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        input.dot(&self.w_out) + &self.b_out
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {

        Array2::zeros(self.w_out.dim())
    }
}