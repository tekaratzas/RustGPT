use ndarray::{Array2, Axis};
use crate::transformer::TransformerBlock;
use crate::Embeddings;
use crate::Vocab;
use crate::output_projection::OutputProjection;
use crate::EMBEDDING_DIM;
use crate::HIDDEN_DIM;

pub trait Layer {
    fn forward(&self, input: &Array2<f32>) -> Array2<f32>;

    fn forward_with_residual(&self, input: &Array2<f32>, layer_norm: &LayerNorm) -> Array2<f32> {
        let output = self.forward(input);
        let residual = &output + input;
        layer_norm.normalize(&residual) 
    }
}

pub struct LLM {
    pub embeddings: Embeddings,
    pub vocab: Vocab,

    output_projection: OutputProjection,
    transformer_block: TransformerBlock,
}

impl Default for LLM {
    fn default() -> Self {
        let transformer_block = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
        let output_projection = OutputProjection::new(EMBEDDING_DIM, Vocab::words().len());
        Self {
            embeddings: Embeddings::default(),
            vocab: Vocab::default(),
            output_projection,
            transformer_block,
        }
    }
}
impl LLM {
    pub fn predict(&self, text: &str) -> String {
        // Generated Embeddings - Learned - seequence x embedding_size
        let token_embeddings = self.embed(text);

        // Transformer Block - Learned - sequence x hidden_size
        let output = self.transformer_block.forward(&token_embeddings);

        // Output Projection - Learned - sequence x vocab_size
        let logits  = self.output_projection.forward(&output);

        // Softmax - convert activiations of each token to a probability distribution over the vocabulary
        let probs = Self::softmax(&logits);

        // Greedy Decode - Choose the highest probability token for each position
        let tokens = Self::greedy_decode(&probs);

        // Convert token_ids to strings
        let token_strs = tokens.iter().map(|t| self.vocab.decode[t].clone()).collect::<Vec<String>>();

        token_strs.join(" ")
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace().map(|s| s.to_string()).collect()
    }

    pub fn embed(&self, text: &str) -> Array2<f32> {
        let tokenized = self.tokenize(text);
        println!("tokenized: {:?}", tokenized);
        
        let tokens = tokenized
            .iter()
            .filter_map(|c| self.vocab.encode(&c.to_string()))
            .collect::<Vec<usize>>();
    
        println!("tokens: {:?}", tokens);
        self.embeddings.embed_tokens(&tokens)
    }

    fn softmax(logits: &Array2<f32>) -> Array2<f32> {
        let mut result = logits.clone();
        
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

    fn greedy_decode(probs: &Array2<f32>) -> Vec<usize> {
        probs.map_axis(Axis(1), |row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index)
                .unwrap()
        }).to_vec()
    }
}

pub struct LayerNorm {
    epsilon: f32,   // Small constant for stability
    gamma: Array2<f32>, // Learnable scaling parameter
    beta: Array2<f32>,  // Learnable bias parameter
}

impl LayerNorm {
    /// Initialize LayerNorm with learnable parameters
    pub fn new(embedding_dim: usize) -> Self {
        LayerNorm {
            epsilon: 1e-5,
            gamma: Array2::ones((1, embedding_dim)), // Initialize gamma to 1
            beta: Array2::zeros((1, embedding_dim)), // Initialize beta to 0
        }
    }

    pub fn normalize(&self, input: &Array2<f32>) -> Array2<f32> {
        let mean = input.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1)); // Mean per token
        let std = input.std_axis(Axis(1), 0.0).insert_axis(Axis(1)); // Std per token

        // Normalize: (X - mean) / (std + epsilon)
        let normalized = (input - &mean) / (&std + self.epsilon);
        normalized * &self.gamma + &self.beta // Apply gamma & beta
    }
}