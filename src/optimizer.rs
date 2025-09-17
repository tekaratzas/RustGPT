use ndarray::Array2;
use crate::optimizers::{adamw::AdamW};

/// Enum to easily select an optimizer type.
#[derive(Clone)]
pub enum OptimizerType {
    AdamW { weight_decay: f32 }, // AdamW includes weight decay
}

/// A trait for optimizers that update model parameters.
pub trait Optimizer {
    /// Performs a single optimization step.
    fn step(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>, lr: f32);
}

/// Factory function to create a new optimizer based on its type.
pub fn new_optimizer(optimizer_type: &OptimizerType, shape: (usize, usize)) -> Box<dyn Optimizer> {
    match optimizer_type {
        OptimizerType::AdamW { weight_decay } => Box::new(AdamW::new(shape, *weight_decay)),
    }
}