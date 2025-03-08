use ndarray::Array2;
use rand::prelude::*;
use crate::llm::Layer;

pub struct FeedForward {
    w1: Array2<f32>,
    b1: Array2<f32>,
    w2: Array2<f32>,
    b2: Array2<f32>,
}

impl FeedForward {
    /// Initialize a feedforward layer with random weights
    pub fn new(embedding_dim: usize, hidden_dim: usize) -> Self {
        let mut rng = rand::rng();
        FeedForward {
            w1: Array2::from_shape_fn((embedding_dim, hidden_dim), |_| rng.random_range(-0.1..0.1)),
            b1: Array2::zeros((1, hidden_dim)), // Bias initialized to 0
            w2: Array2::from_shape_fn((hidden_dim, embedding_dim), |_| rng.random_range(-0.1..0.1)),
            b2: Array2::zeros((1, embedding_dim)), // Bias initialized to 0
        }
    }

    /// ReLU activation function
    fn relu(x: Array2<f32>) -> Array2<f32> {
        x.mapv(|v| v.max(0.0))
    }
}

impl Layer for FeedForward {
    fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let hidden = Self::relu(input.dot(&self.w1) + &self.b1);
        hidden.dot(&self.w2) + &self.b2
    }
}