use ndarray::{Array2, Axis};
use crate::transformer::TransformerBlock;
use crate::Embeddings;
use crate::Vocab;
use crate::output_projection::OutputProjection;
use crate::EMBEDDING_DIM;
use crate::HIDDEN_DIM;
use crate::MAX_SEQ_LEN;

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
        let output_projection = OutputProjection::new(EMBEDDING_DIM, Vocab::default_words().len());
        Self {
            embeddings: Embeddings::default(),
            vocab: Vocab::default(),
            output_projection,
            transformer_block,
        }
    }
}

impl LLM {
    pub fn new(vocab: Vocab) -> Self {
        let transformer_block = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
        let output_projection = OutputProjection::new(EMBEDDING_DIM, vocab.words.len());
        Self {
            embeddings: Embeddings::new(vocab.clone()),
            vocab,
            output_projection,
            transformer_block,
        }
    }
}
impl LLM {
    pub fn predict(&self, text: &str) -> String {
        // Tokenize the input text
        let mut tokenized = self.tokenize(text);
        let mut output_tokens: Vec<usize> = Vec::new();

        let input_len = tokenized.len();
                
        for _ in 0..MAX_SEQ_LEN-input_len {
            // Check if we're approaching the maximum sequence length
            if output_tokens.len() >= MAX_SEQ_LEN - 1 {
                break;
            }
            
            println!("tokenized length: {:?}", tokenized.len());
            // Generated Input Embeddings - Learned - seequence x embedding_size
            let token_embeddings =  self.embeddings.embed_tokens(&tokenized);

            // Transformer Block - Learned - sequence x hidden_size
            let output = self.transformer_block.forward(&token_embeddings);

            // Output Projection - Learned - sequence x vocab_size
            let logits  = self.output_projection.forward(&output);

            // Softmax - convert activiations of each token to a probability distribution over the vocabulary
            let probs = Self::softmax(&logits); // sequence x vocab_size

            // Greedy Decode - Choose the highest probability token for each position
            let tokens = Self::greedy_decode(&probs);

            let next_token = tokens[tokens.len() - 1];

            output_tokens.push(next_token);
            tokenized.push(next_token);

            if next_token == self.vocab.encode("</s>").unwrap() { break; }
        }

        // Convert token_ids to strings
        let token_strs = output_tokens.iter().map(|t| self.vocab.decode[t].clone()).collect::<Vec<String>>();

        token_strs.join(" ")
    }

    fn train(data: Vec<(Vec<usize>, Vec<usize>)>, epochs: usize, lr: f32) {

    }

    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        // Split by whitespace first
        let mut tokens = Vec::new();
        
        for word in text.split_whitespace() {
            let mut current_word = String::new();
            
            for c in word.chars() {
                if c.is_ascii_punctuation() {
                    // If we have a word before the punctuation, add it
                    if !current_word.is_empty() {
                        if let Some(token_id) = self.vocab.encode(&current_word) {
                            tokens.push(token_id);
                        }
                        current_word.clear();
                    }
                    
                    // Add the punctuation as its own token
                    if let Some(token_id) = self.vocab.encode(&c.to_string()) {
                        tokens.push(token_id);
                    }
                } else {
                    current_word.push(c);
                }
            }
            
            // Add any remaining word
            if !current_word.is_empty() {
                if let Some(token_id) = self.vocab.encode(&current_word) {
                    tokens.push(token_id);
                }
            }
        }
        
        tokens
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