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
            // Calculate exp for each element
            let exp_values: Vec<f32> = row.iter().map(|&x| x.exp()).collect();
            let sum_exp: f32 = exp_values.iter().sum();
            
            // Normalize by sum
            for (i, &exp_val) in exp_values.iter().enumerate() {
                row[i] = exp_val / sum_exp;
            }
        }
        
        result
    }
}

impl Layer for SelfAttention {
    fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let qkv = self.compute_qkv(input);
        self.attention(&qkv.0, &qkv.1, &qkv.2)
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        Array2::zeros(self.w_q.dim())
    }
}