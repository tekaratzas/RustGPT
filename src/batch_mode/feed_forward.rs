use ndarray::{Array2, Array3, Axis};
use rand_distr::{Distribution, Normal};
use itertools::izip;
use crate::{adam::Adam};
use crate::BatchLayer;

pub struct FeedForward {
    w1: Array2<f32>,
    b1: Array2<f32>,
    w2: Array2<f32>,
    b2: Array2<f32>,

    // Cached values for backward pass
    cached_input: Option<Array3<f32>>,
    hidden_pre_activation: Option<Array3<f32>>,
    hidden_post_activation: Option<Array3<f32>>,

    optimizer_w1: Adam,
    optimizer_b1: Adam,
    optimizer_w2: Adam,
    optimizer_b2: Adam,
}

impl FeedForward {
    /// Initialize a feedforward layer with random weights
    pub fn new(embedding_dim: usize, hidden_dim: usize) -> Self {
        let mut rng = rand::rng();

        // Xavier/He initialization for w1: std = sqrt(2 / fan_in)
        let std_w1 = (2.0 / embedding_dim as f32).sqrt();
        let normal_w1 = Normal::new(0.0, std_w1).unwrap();

        // Xavier/He initialization for w2: std = sqrt(2 / fan_in)
        let std_w2 = (2.0 / hidden_dim as f32).sqrt();
        let normal_w2 = Normal::new(0.0, std_w2).unwrap();

        FeedForward {
            w1: Array2::from_shape_fn((embedding_dim, hidden_dim), |_| normal_w1.sample(&mut rng)),
            b1: Array2::zeros((1, hidden_dim)), // Bias initialized to 0
            w2: Array2::from_shape_fn((hidden_dim, embedding_dim), |_| normal_w2.sample(&mut rng)),
            b2: Array2::zeros((1, embedding_dim)), // Bias initialized to 0
            cached_input: None,
            hidden_pre_activation: None,
            hidden_post_activation: None,
            optimizer_w1: Adam::new((embedding_dim, hidden_dim)),
            optimizer_b1: Adam::new((1, hidden_dim)),
            optimizer_w2: Adam::new((hidden_dim, embedding_dim)),
            optimizer_b2: Adam::new((1, embedding_dim)),
        }
    }
}

impl BatchLayer for FeedForward {
    fn layer_type(&self) -> &str {
        "FeedForward"
    }

    fn forward(&mut self, input: &Array3<f32>) -> Array3<f32> {
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];

        // Allocate output, hidden_pre, hidden_post
        let mut hidden_pre_activation = Array3::<f32>::zeros((batch_size, seq_len, self.b1.shape()[1]));
        let mut hidden_post_activation = Array3::<f32>::zeros((batch_size, seq_len, self.b1.shape()[1]));
        let mut output = Array3::<f32>::zeros((batch_size, seq_len, self.b2.shape()[1]));


        for (((mut out_slice, mut pre_slice), mut post_slice), in_slice) in output
        .outer_iter_mut()
        .zip(hidden_pre_activation.outer_iter_mut())
        .zip(hidden_post_activation.outer_iter_mut())
        .zip(input.outer_iter())
        {
            let hidden_pre_slice = in_slice.dot(&self.w1) + &self.b1;
            let hidden_post_slice = hidden_pre_slice.mapv(|x| x.max(0.0)); // ReLU
            pre_slice.assign(&hidden_pre_slice);
            post_slice.assign(&hidden_post_slice);
            out_slice.assign(&(hidden_post_slice.dot(&self.w2) + &self.b2));
        }

        // Cache values
        self.cached_input = Some(input.clone());
        self.hidden_pre_activation = Some(hidden_pre_activation);
        self.hidden_post_activation = Some(hidden_post_activation);

        output + input // residual connection (no LayerNorm here)
    }


    fn backward(&mut self, grads: &Array3<f32>, lr: f32) -> Array3<f32> {
        // Unwrap cached values
        let input = self.cached_input.as_ref().expect("forward must be run first");
        let batch_size = input.shape()[0];
        let hidden_pre_activation = self.hidden_pre_activation.as_ref().unwrap();
        let hidden_post_activation = self.hidden_post_activation.as_ref().unwrap();

        // Setup gradient accumulators
        let mut grad_input = Array3::<f32>::zeros(input.raw_dim()); // [batch, seq_len, input_dim]
        let mut grad_w1 = Array2::<f32>::zeros(self.w1.raw_dim());
        let mut grad_w2 = Array2::<f32>::zeros(self.w2.raw_dim());
        let mut grad_b1 = Array2::<f32>::zeros(self.b1.raw_dim());
        let mut grad_b2 = Array2::<f32>::zeros(self.b2.raw_dim());

        // now, we compute the gradients for w1, w2, b1, b2, and update parameters via Adam.
        for (i, (in_slice, grad_slice, hidden_pre_slice, hidden_post_slice)) in
        izip!(
            input.outer_iter(),
            grads.outer_iter(),
            hidden_pre_activation.outer_iter(),
            hidden_post_activation.outer_iter()
        )
        .enumerate() {
            grad_w2 += &hidden_post_slice.t().dot(&grad_slice);
            grad_b2 += &grad_slice.sum_axis(Axis(0)).insert_axis(Axis(0)); // Shape: [1, embedding_dim]
            
            // Gradient w.r.t. hidden_post_activation
            let grad_hidden_post_activation = &grad_slice.dot(&self.w2.t());

            // Gradient through ReLU
            let relu_grad = &hidden_pre_slice.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
            let grad_hidden_pre_activation = grad_hidden_post_activation * relu_grad;

            // Gradient w.r.t. W1 and b1
            grad_w1 += &in_slice.t().dot(&grad_hidden_pre_activation);
            grad_b1 += &grad_hidden_pre_activation
                .sum_axis(Axis(0))
                .insert_axis(Axis(0)); // Shape: [1, hidden_dim]
            
            // Gradient w.r.t. input (through feed-forward computation)
            let grad_input_feedforward = grad_hidden_pre_activation.dot(&self.w1.t());

            // Add gradient from residual connection (for each tensor)
            // Forward: output = W2(ReLU(W1*input + b1)) + b2 + input
            // Backward: grad_input = grad_feedforward + grad_residual
            grad_input.slice_mut(ndarray::s![i, .., ..]).assign(&(&grad_input_feedforward + &grad_slice));
        }

        grad_input /= batch_size as f32;
        grad_w1 /= batch_size as f32;
        grad_w2 /= batch_size as f32;
        grad_b1 /= batch_size as f32;
        grad_b2 /= batch_size as f32;

        // Update parameters via Adam optimizer
        self.optimizer_w2.step(&mut self.w2, &grad_w2, lr);
        self.optimizer_b2.step(&mut self.b2, &grad_b2, lr);
        self.optimizer_w1.step(&mut self.w1, &grad_w1, lr);
        self.optimizer_b1.step(&mut self.b1, &grad_b1, lr);

        grad_input
    }

    fn parameters(&self) -> usize {
        self.b1.len() + self.b2.len() + self.w1.len() + self.w2.len()
    }
}
