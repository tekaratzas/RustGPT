use crate::adam::Adam;
use crate::layer_norm::LayerNorm;
use crate::EMBEDDING_DIM;
use ndarray::Array2;
use rand::prelude::*;
use rand::thread_rng;
use crate::llm::Layer;
use std::f32;

pub struct SelfAttention {
    pub embedding_dim: usize,
    //pub num_heads: usize,
    w_q: Array2<f32>, // Weight matrices for Q, K, V
    w_k: Array2<f32>,
    w_v: Array2<f32>,
    norm: LayerNorm,

    cached_input: Option<Array2<f32>>,

    optimizer_w_q: Adam,
    optimizer_w_k: Adam,
    optimizer_w_v: Adam,
}

impl Default for SelfAttention {
    fn default() -> Self {
        SelfAttention::new(EMBEDDING_DIM)
    }
}
    

impl SelfAttention {
    /// Initializes a Transformer with random Q, K, V weights
    pub fn new(embedding_dim: usize) -> Self {
        let mut rng = rand::rng();
        SelfAttention {
            embedding_dim,
            w_q: Array2::from_shape_fn((embedding_dim, embedding_dim), |_| rng.random_range(-0.1..0.1)),
            w_k: Array2::from_shape_fn((embedding_dim, embedding_dim), |_| rng.random_range(-0.1..0.1)),
            w_v: Array2::from_shape_fn((embedding_dim, embedding_dim), |_| rng.random_range(-0.1..0.1)),
            norm: LayerNorm::new(embedding_dim),
            cached_input: None,
            optimizer_w_q: Adam::new((embedding_dim, embedding_dim)),
            optimizer_w_k: Adam::new((embedding_dim, embedding_dim)),
            optimizer_w_v: Adam::new((embedding_dim, embedding_dim)),
        }
    }

    fn compute_qkv(&self, input: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let q = input.dot(&self.w_q); // Q = X * W_Q
        let k = input.dot(&self.w_k); // K = X * W_K
        let v = input.dot(&self.w_v); // V = X * W_V
        (q, k, v)
    }

    fn attention(&self, q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>) -> Array2<f32> {
        let dk = (self.embedding_dim as f32).sqrt();

        let k_t = k.t();
        let scores = q.dot(&k_t) / dk;

        let weights = self.softmax(&scores);
        weights.dot(v)
    }

    fn softmax(&self, scores: &Array2<f32>) -> Array2<f32> {
        let mut result = scores.clone();
        
        // Apply softmax row-wise
        for mut row in result.rows_mut() {
            let max_val = row.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            // Calculate exp for each element
            let exp_values: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
            let sum_exp: f32 = exp_values.iter().sum();
            
            // Normalize by sum
            for (i, &exp_val) in exp_values.iter().enumerate() {
                row[i] = exp_val / sum_exp;
            }
        }
        
        result
    }

    fn softmax_backward(
        softmax_output: &Array2<f32>,  // shape: [seq_len, vocab_size]
        grad_output: &Array2<f32>,     // shape: [seq_len, vocab_size]
    ) -> Array2<f32> {
        let mut grad_input = softmax_output.clone(); // to hold the result
    
        for ((mut grad_row, softmax_row), grad_out_row) in
            grad_input
                .outer_iter_mut()
                .zip(softmax_output.outer_iter())
                .zip(grad_output.outer_iter())
        {
            // dot product: y ⊙ dL/dy
            let dot = softmax_row
                .iter()
                .zip(grad_out_row.iter())
                .map(|(&y_i, &dy_i)| y_i * dy_i)
                .sum::<f32>();
    
            for ((g, &y_i), &dy_i) in grad_row
                .iter_mut()
                .zip(softmax_row.iter())
                .zip(grad_out_row.iter())
            {
                *g = y_i * (dy_i - dot);
            }
        }
    
        grad_input
    }
}

impl Layer for SelfAttention {
    fn layer_type(&self) -> &str {
        "SelfAttention"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.cached_input = Some(input.clone());
        let qkv = self.compute_qkv(input);
        let attention = self.attention(&qkv.0, &qkv.1, &qkv.2);
        let residual = attention + input; // residual connection
        self.norm.normalize(&residual)
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let input = self.cached_input.as_ref().unwrap();
        let q = input.dot(&self.w_q);
        let k = input.dot(&self.w_k);
        let v = input.dot(&self.w_v);
        let dk = self.w_q.shape()[1] as f32;
        let scale = dk.sqrt();
    
        let scores = q.dot(&k.t()) / scale;
        let attn_weights = self.softmax(&scores); // also cached
        let attn_output = attn_weights.dot(&v); // also cached
    
        // Step 1: grads = ∂L/∂attn_output
        let grad_attn_weights = grads.dot(&v.t());
        let grad_v = attn_weights.t().dot(grads);
    
        // Step 2: softmax backward
        let grad_scores = SelfAttention::softmax_backward(&attn_weights, &grad_attn_weights); // [seq_len, seq_len]
    
        // Step 3: ∂L/∂Q and ∂L/∂K
        let grad_q = grad_scores.dot(&k);
        let grad_k = grad_scores.t().dot(&q);
    
        // Step 4: ∂L/∂W_q/W_k/W_v
        let grad_w_q = input.t().dot(&grad_q);
        let grad_w_k = input.t().dot(&grad_k);
        let grad_w_v = input.t().dot(&grad_v);
    
        // Step 5: ∂L/∂input
        let grad_input =
            grad_q.dot(&self.w_q.t()) +
            grad_k.dot(&self.w_k.t()) +
            grad_v.dot(&self.w_v.t());
    
        // Step 6: update weights
        self.optimizer_w_q.step(&mut self.w_q, &grad_w_q, lr);
        self.optimizer_w_k.step(&mut self.w_k, &grad_w_k, lr);
        self.optimizer_w_v.step(&mut self.w_v, &grad_w_v, lr);
    
        grad_input        
    }
}