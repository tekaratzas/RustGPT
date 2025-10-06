use std::io::Write;

use ::llm::{EMBEDDING_DIM, HIDDEN_DIM, MAX_SEQ_LEN};
use dataset_loader::{Dataset, DatasetType};

// Import local modules that make up different parts of the model
mod adam;
mod dataset_loader;
mod embeddings;
mod feed_forward;
mod layer_norm;
mod llm;
mod output_projection;
mod self_attention;
mod transformer;
mod vocab;

fn main() {
    // Mock input used to test the model before and after training
    let string = String::from("User: How do mountains form?");

    // Create a hash set to collect all unique tokens (words and punctuation)
    let mut vocab_set = std::collections::HashSet::new();

    // Add special end-of-sequence token to vocabulary
    vocab_set.insert("</s>".to_string());

    // === Load dataset ===
    let dataset = Dataset::new(
        String::from("data/pretraining_data.json"),
        String::from("data/chat_training_data.json"),
        DatasetType::JSON,
    ); // This loads pretraining and chat fine-tuning data

    // === Build vocabulary from pre-training data ===
    for text in &dataset.pretraining_data {
        for word in text.split_whitespace() {
            // Split punctuation from words (e.g., "hello," → "hello" and ",")
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

    // === Build vocabulary from chat (instruction-tuning) data ===
    for row in &dataset.chat_training_data {
        for word in row.split_whitespace() {
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

    // Convert vocabulary set into a sorted vector for deterministic order
    let mut vocab_words: Vec<String> = vocab_set.into_iter().collect();
    vocab_words.sort();

    // Convert Vec<String> to Vec<&str> because Vocab expects string slices
    let vocab_words_refs: Vec<&str> = vocab_words.iter().map(|s: &String| s.as_str()).collect();

    // Build the vocabulary structure
    let vocab = Vocab::new(vocab_words_refs);

    // === Build Transformer-based model ===
    // These represent multiple stacked Transformer layers
    let transformer_block_1 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let transformer_block_2 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let transformer_block_3 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);

    // Projection layer: maps hidden state to vocabulary logits
    let output_projection = OutputProjection::new(EMBEDDING_DIM, vocab.words.len());

    // Embedding layer: turns tokens into dense vectors
    let embeddings = Embeddings::new(vocab.clone());

    // Create the full LLM by stacking components in order
    let mut llm = LLM::new(
        vocab,
        vec![
            Box::new(embeddings),
            Box::new(transformer_block_1),
            Box::new(transformer_block_2),
            Box::new(transformer_block_3),
            Box::new(output_projection),
        ],
    );

    // === Print model information ===
    println!("\n=== MODEL INFORMATION ===");
    println!("Network architecture: {}", llm.network_description());
    println!(
        "Model configuration -> max_seq_len: {}, embedding_dim: {}, hidden_dim: {}",
        MAX_SEQ_LEN, EMBEDDING_DIM, HIDDEN_DIM
    );
    println!("Total parameters: {}", llm.total_parameters());

    // === Test before any training ===
    println!("\n=== BEFORE TRAINING ===");
    println!("Input: {}", string);
    println!("Output: {}", llm.predict(&string));

    // === Pre-training phase ===
    println!("\n=== PRE-TRAINING MODEL ===");
    println!(
        "Pre-training on {} examples for {} epochs with learning rate {}",
        dataset.pretraining_data.len(),
        100,
        0.0005
    );

    // Collect pre-training examples as slices
    let pretraining_examples: Vec<&str> = dataset
        .pretraining_data
        .iter()
        .map(|s| s.as_str())
        .collect();

    // Train the model on pretraining data
    llm.train(pretraining_examples, 100, 0.0005);

    // === Instruction tuning (fine-tuning for chat) ===
    println!("\n=== INSTRUCTION TUNING ===");
    println!(
        "Instruction tuning on {} examples for {} epochs with learning rate {}",
        dataset.chat_training_data.len(),
        100,
        0.0001
    );

    // Collect chat training examples as slices
    let chat_training_examples: Vec<&str> = dataset
        .chat_training_data
        .iter()
        .map(|s| s.as_str())
        .collect();

    // Train again on chat data with a smaller learning rate for stability
    llm.train(chat_training_examples, 100, 0.0001);

    // === Test after training ===
    println!("\n=== AFTER TRAINING ===");
    println!("Input: {}", string);
    let result = llm.predict(&string);
    println!("Output: {}", result);
    println!("======================\n");

    // === Interactive loop ===
    println!("\n--- Interactive Mode ---");
    println!("Type a prompt and press Enter to generate text.");
    println!("Type 'exit' to quit.");

    let mut input = String::new();
    loop {
        // Clear input buffer
        input.clear();

        // Print prompt without newline
        print!("\nEnter prompt: ");
        std::io::stdout().flush().unwrap();

        // Read line from stdin
        std::io::stdin()
            .read_line(&mut input)
            .expect("Failed to read input");

        // Remove surrounding whitespace
        let trimmed_input = input.trim();

        // Check for exit command
        if trimmed_input.eq_ignore_ascii_case("exit") {
            println!("Exiting interactive mode.");
            break;
        }

        // Add "User:" prefix so model understands the format
        // Generate response from model
        let formatted_input = format!("User: {}", trimmed_input);
        let prediction = llm.predict(&formatted_input);
        println!("Model output: {}", prediction);
    }
}
