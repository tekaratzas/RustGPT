use ndarray::{Array3};
use crate::{Vocab};
pub trait Layer {
    fn layer_type(&self) -> &str;

    fn forward(&mut self, input: &Array3<f32>) -> Array3<f32>;

    fn backward(&mut self, grads: &Array3<f32>, lr: f32) -> Array3<f32>;

    fn parameters(&self) -> usize;
    
}

#[allow(clippy::upper_case_acronyms)]
pub struct LLM {
    pub vocab: Vocab,
    pub network: Vec<Box<dyn Layer>>,
}