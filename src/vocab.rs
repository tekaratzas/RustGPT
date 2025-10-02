use bincode::Encode;
use std::collections::HashMap;
use std::collections::HashSet;

#[derive(Clone, Encode)]
pub struct Vocab {
    pub encode: HashMap<String, usize>,
    pub decode: HashMap<usize, String>,
    pub words: Vec<String>,
}

impl Default for Vocab {
    fn default() -> Self {
        Self::new(Self::default_words())
    }
}

impl Vocab {
    pub fn new(words: Vec<&str>) -> Self {
        let mut encode = HashMap::new();
        let mut decode = HashMap::new();

        for (i, &word) in words.iter().enumerate() {
            println!("Adding word: {word} to encoding: {i}");
            encode.insert(word.to_string(), i);
            decode.insert(i, word.to_string());
        }

        Vocab {
            encode,
            decode,
            words: words.iter().map(|w| w.to_string()).collect(),
        }
    }

    /// Convert a word to its token index
    pub fn encode(&self, word: &str) -> Option<usize> {
        self.encode.get(word).copied()
    }

    /// Convert a token index back to a word
    #[allow(dead_code)]
    pub fn decode(&self, token_id: usize) -> Option<&String> {
        self.decode.get(&token_id)
    }

    pub fn default_words() -> Vec<&'static str> {
        vec!["hello", "world", "this", "is", "rust", "</s>"]
    }
    
    /// Process text data to extract vocabulary words and add them to the vocabulary set
    pub fn process_text_for_vocab(texts: &[String], vocab_set: &mut HashSet<String>) {
        // Add end of sequence token
        vocab_set.insert("</s>".to_string());

        // Process all training examples for vocabulary
        for text in texts {
            for word in text.split_whitespace() {
                // Handle punctuation by splitting it from words
                let mut current = String::new();
                for c in word.chars() {
                    if c.is_ascii_punctuation() {
                        if !current.is_empty() {
                            vocab_set.insert(current.clone());
                            current.clear();
                        }
                        vocab_set.insert(c.to_string());
                    } else {
                        current.push(c);
                    }
                }
                if !current.is_empty() {
                    vocab_set.insert(current);
                }
            }
        }
    }
}

impl Into<String> for Vocab {
    fn into(self) -> String {
        String::from_iter(
            self.words
                .iter()
                .enumerate()
                .map(|(i, str)| format!("({i},{str}),")),
        )
    }
}
