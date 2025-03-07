use std::collections::HashMap;
pub struct Vocab {
    pub vocab: HashMap<String, usize>,
}

impl Default for Vocab {
    fn default() -> Self {
        Self {
            vocab: Self::create_vocab(),
        }
    }
}

impl Vocab {
    pub fn create_vocab() -> HashMap<String, usize> {
        let mut vocab = HashMap::new();
        vocab.insert("Hello".to_string(), 1);
        vocab.insert("world".to_string(), 2);
        vocab
    }
    
    pub fn vocab_size() -> usize {
        100
    }
}