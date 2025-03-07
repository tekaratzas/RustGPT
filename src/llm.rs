use ndarray::{s, Array2, Axis};
use crate::Embeddings;
use crate::Vocab;

pub struct LLM {
    pub embeddings: Embeddings,
    pub vocab: Vocab,
}

impl Default for LLM {
    fn default() -> Self {
        Self {
            embeddings: Embeddings::default(),
            vocab: Vocab::default(),
        }
    }
}
impl LLM {
    pub fn predict(&self, text: &str) -> String {
        let token_embeddings = self.embed(text);

        println!("token_embeddings: {:?}", token_embeddings);
        String::new()
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace().map(|s| s.to_string()).collect()
    }

    pub fn embed(&self, text: &str) -> Array2<f32> {
        let tokenized = self.tokenize(text);
        println!("tokenized: {:?}", tokenized);
        
        let tokens = tokenized
            .iter()
            .filter_map(|c| self.vocab.vocab.get(&c.to_string()).copied())
            .collect::<Vec<usize>>();
    
        println!("tokens: {:?}", tokens);
        self.embeddings.embed_tokens(&tokens)
    }
}



