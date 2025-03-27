use crate::adam::Adam;
use ndarray::Array2;
use ndarray::Axis;
use crate::llm::Layer;

pub struct LayerNorm {
    epsilon: f32,   // Small constant for stability
    // gamma: Array2<f32>, // Learnable scaling parameter
    // beta: Array2<f32>,  // Learnable bias parameter

    // cached_input: Option<Array2<f32>>,
    // cached_mean: Option<Array2<f32>>,
    // cached_std: Option<Array2<f32>>,

    // optimizer_gamma: Adam,
    // optimizer_beta: Adam,
}

impl LayerNorm {
    /// Initialize LayerNorm with learnable parameters
    pub fn new(embedding_dim: usize) -> Self {
        LayerNorm {
            epsilon: 1e-5,
            // gamma: Array2::ones((1, embedding_dim)), // Initialize gamma to 1
            // beta: Array2::zeros((1, embedding_dim)), // Initialize beta to 0
            // cached_input: None,
            // cached_mean: None,
            // cached_std: None,
            // optimizer_gamma: Adam::new((1, embedding_dim)),
            // optimizer_beta: Adam::new((1, embedding_dim)),
        }
    }

    pub fn normalize(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let mean = input.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1)); // Mean per token
        let std = input.std_axis(Axis(1), 0.0).insert_axis(Axis(1)); // Std per token

        (input - &mean) / (&std + self.epsilon)
    }
}

impl Layer for LayerNorm {
    fn layer_type(&self) -> &str {
        "LayerNorm"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // self.cached_input = Some(input.clone());
        self.normalize(input)
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        grads.clone() // nothing to do here. Non-learnable parameters.
    }
}
