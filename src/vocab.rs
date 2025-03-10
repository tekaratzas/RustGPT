use std::collections::HashMap;
pub struct Vocab {
    pub encode: HashMap<String, usize>,
    pub decode: HashMap<usize, String>,
}

impl Default for Vocab {
    fn default() -> Self {
        Self::new(Self::words())
    }
}

impl Vocab {
    pub fn new(words: Vec<&str>) -> Self {
        let mut encode = HashMap::new();
        let mut decode = HashMap::new();

        for (i, &word) in words.iter().enumerate() {
            encode.insert(word.to_string(), i);
            decode.insert(i, word.to_string());
        }

        Vocab { encode, decode }
    }

    /// Convert a word to its token index
    pub fn encode(&self, word: &str) -> Option<usize> {
        self.encode.get(word).copied()
    }

    /// Convert a token index back to a word
    pub fn decode(&self, token_id: usize) -> Option<&String> {
        self.decode.get(&token_id)
    }

    pub fn words() -> Vec<&'static str> {
        vec!["hello", "world", "this", "is", "rust", "</s>"]
    }
}