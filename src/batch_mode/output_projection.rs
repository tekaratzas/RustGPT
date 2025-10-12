use ndarray::{Array2, Array3, Axis};
use rand_distr::{Distribution, Normal};

use crate::adam::Adam;
use crate::BatchLayer;
pub struct OutputProjection {
    pub w_out: Array2<f32>, // Weight matrix
    pub b_out: Array2<f32>, // Bias vector
    pub optimizer: Adam,
    pub cached_input: Option<Array3<f32>>,
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

impl BatchLayer for OutputProjection {
    fn layer_type(&self) -> &str {
        "OutputProjection"
    }

    /// Forward pass for batched input: [batch_size, seq_len, embedding_dim]
    fn forward(&mut self, input: &Array3<f32>) -> Array3<f32> {
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];

        self.cached_input = Some(input.clone());

        // Allocate output: [batch_size, seq_len, vocab_size]
        let mut output = Array3::<f32>::zeros((batch_size, seq_len, self.b_out.shape()[1]));

        for (mut out_slice, in_slice) in output.outer_iter_mut().zip(input.outer_iter()) {
            // in_slice shape: [seq_len, embedding_dim]
            // out_slice shape: [seq_len, vocab_size]
            out_slice.assign(&(in_slice.dot(&self.w_out) + &self.b_out));
        }

        output
    }

    /// Backward pass for batched input
    fn backward(&mut self, grads: &Array3<f32>, lr: f32) -> Array3<f32> {
        let input = self.cached_input.as_ref().unwrap();
        let batch_size = input.shape()[0];

        let mut grad_input = Array3::<f32>::zeros(input.raw_dim());
        let mut grad_w_out = Array2::<f32>::zeros(self.w_out.raw_dim());
        let mut grad_b_out = Array2::<f32>::zeros(self.b_out.raw_dim());

        for (i, (in_slice, grad_slice)) in input.outer_iter().zip(grads.outer_iter()).enumerate() {
            // Compute gradients for weights and bias
            grad_w_out += &in_slice.t().dot(&grad_slice);
            grad_b_out += &grad_slice.mean_axis(Axis(0)).unwrap();
        
            // Compute gradient wrt input slice and assign to grad_input
            grad_input.slice_mut(ndarray::s![i, .., ..]).assign(&grad_slice.dot(&self.w_out.t()));
        }

        grad_w_out /= batch_size as f32;
        grad_b_out /= batch_size as f32;

        self.optimizer.step(&mut self.w_out, &grad_w_out, lr);
        self.b_out -= &(lr * &grad_b_out);

        grad_input
    }

    fn parameters(&self) -> usize {
        self.w_out.len() + self.b_out.len()
    }
}
