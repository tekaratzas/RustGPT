use crate::adam::Adam;
use crate::llm::Layer;
use ndarray::{Array2, Axis};

pub struct LayerNorm {
    epsilon: f32,       // Small constant for stability
    gamma: Array2<f32>, // Learnable scaling parameter
    beta: Array2<f32>,  // Learnable bias parameter

    cached_input: Option<Array2<f32>>,
    cached_mean: Option<Array2<f32>>,
    cached_std: Option<Array2<f32>>,

    optimizer_gamma: Adam,
    optimizer_beta: Adam,
}

impl LayerNorm {
    /// Initialize LayerNorm with learnable parameters
    pub fn new(embedding_dim: usize) -> Self {
        LayerNorm {
            epsilon: 1e-5,
            gamma: Array2::ones((1, embedding_dim)), // Initialize gamma to 1
            beta: Array2::zeros((1, embedding_dim)), // Initialize beta to 0
            cached_input: None,
            cached_mean: None,
            cached_std: None,
            optimizer_gamma: Adam::new((1, embedding_dim)),
            optimizer_beta: Adam::new((1, embedding_dim)),
        }
    }

    pub fn normalize(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let mean = input.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1)); // Mean per token
        let std = input.std_axis(Axis(1), 0.0).insert_axis(Axis(1)); // Std per token

        // Cache values for backward pass
        self.cached_input = Some(input.clone());
        self.cached_mean = Some(mean.clone());
        self.cached_std = Some(std.clone());

        let normalized = (input - &mean) / (&std + self.epsilon);
        &self.gamma * &normalized + &self.beta
    }
}

impl Layer for LayerNorm {
    fn layer_type(&self) -> &str {
        "LayerNorm"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.normalize(input)
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let input = self.cached_input.as_ref().unwrap();
        let mean = self.cached_mean.as_ref().unwrap();
        let std = self.cached_std.as_ref().unwrap();

        let normalized = (input - mean) / (std + self.epsilon);
        let n_features = input.shape()[1] as f32; // Number of features per token

        // Gradients w.r.t. gamma and beta
        let grad_gamma = (&normalized * grads).sum_axis(Axis(0)).insert_axis(Axis(0));
        let grad_beta = grads.sum_axis(Axis(0)).insert_axis(Axis(0));

        // Gradient w.r.t. normalized values
        let grad_normalized = &self.gamma * grads;

        // LayerNorm backward pass with full chain rule
        let grad_input = {
            let variance = std * std + self.epsilon;
            let grad_var = (&grad_normalized * &normalized)
                .sum_axis(Axis(1))
                .insert_axis(Axis(1))
                * (-0.5)
                / variance.mapv(|x| x * x.sqrt());
            let grad_mean = grad_normalized.sum_axis(Axis(1)).insert_axis(Axis(1)) * (-1.0)
                / (std + self.epsilon)
                + &grad_var * (input - mean).sum_axis(Axis(1)).insert_axis(Axis(1)) * (-2.0)
                    / n_features;

            &grad_normalized / (std + self.epsilon)
                + &grad_var * 2.0 * (input - mean) / n_features
                + &grad_mean / n_features
        };

        // Update learnable parameters
        self.optimizer_gamma.step(&mut self.gamma, &grad_gamma, lr);
        self.optimizer_beta.step(&mut self.beta, &grad_beta, lr);

        grad_input
    }

    fn parameters(&self) -> usize {
        self.gamma.len() + self.beta.len()
    }
}
