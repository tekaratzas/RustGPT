use ndarray::Array2;
use ndarray::Axis;
use rand::prelude::*;
use crate::layer_norm::LayerNorm;
use crate::{adam::Adam, llm::Layer};

pub struct FeedForward {
    w1: Array2<f32>,
    b1: Array2<f32>,
    w2: Array2<f32>,
    b2: Array2<f32>,
    norm: LayerNorm,

    // Cached values for backward pass
    input: Option<Array2<f32>>,
    hidden_pre_activation: Option<Array2<f32>>,
    hidden_post_activation: Option<Array2<f32>>,

    optimizer_w1: Adam,
    optimizer_b1: Adam,
    optimizer_w2: Adam,
    optimizer_b2: Adam,
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
            norm: LayerNorm::new(embedding_dim),
            input: None,
            hidden_pre_activation: None,
            hidden_post_activation: None,
            optimizer_w1: Adam::new((embedding_dim, hidden_dim)),
            optimizer_b1: Adam::new((1, hidden_dim)),
            optimizer_w2: Adam::new((hidden_dim, embedding_dim)),
            optimizer_b2: Adam::new((1, embedding_dim)),
        }
    }
}

impl Layer for FeedForward {
    fn layer_type(&self) -> &str {
        "FeedForward"
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        // Unwrap cached values
        let input = self.input.as_ref().expect("forward must be run first");
        let hidden_pre_activation = self.hidden_pre_activation.as_ref().unwrap();
        let hidden_post_activation = self.hidden_post_activation.as_ref().unwrap();

        let batch_size = input.shape()[0] as f32;

        let grad_w2 = hidden_post_activation.t().dot(grads) / batch_size;
        let grad_b2 = grads.mean_axis(Axis(0)).unwrap();

        // Gradient w.r.t. hidden_post_activation
        let grad_hidden_post_activation = grads.dot(&self.w2.t());

        // Gradient through ReLU
        let relu_grad = hidden_pre_activation.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
        let grad_hidden_pre_activation = grad_hidden_post_activation * relu_grad;

        // Gradient w.r.t. W1 and b1
        let grad_w1 = input.t().dot(&grad_hidden_pre_activation) / batch_size;
        let grad_b1 = grad_hidden_pre_activation.mean_axis(Axis(0)).unwrap();

        // Gradient w.r.t. input (for previous layer)
        let grad_input = grad_hidden_pre_activation.dot(&self.w1.t());

        // Update parameters via Adam optimizer
        self.optimizer_w2.step(&mut self.w2, &grad_w2, lr); // lr is 0.01
        self.optimizer_b2.step(&mut self.b2, &grad_b2.insert_axis(Axis(0)), lr);
        self.optimizer_w1.step(&mut self.w1, &grad_w1, lr);
        self.optimizer_b1.step(&mut self.b1, &grad_b1.insert_axis(Axis(0)), lr);

        grad_input
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {

        let hidden_pre_activation = input.dot(&self.w1) + &self.b1;
        let hidden_post_activation = hidden_pre_activation.mapv(|x| x.max(0.0)); // ReLU
        
        let output = hidden_post_activation.dot(&self.w2) + &self.b2;

        // Cache values
        self.input = Some(input.clone());
        self.hidden_pre_activation = Some(hidden_pre_activation);
        self.hidden_post_activation = Some(hidden_post_activation);

        let residual = output + input; // residual connection
        self.norm.normalize(&residual)
    }
}