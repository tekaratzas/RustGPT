use std::io::Write;

use embeddings::Embeddings;
use output_projection::OutputProjection;
use transformer::TransformerBlock;
use llm::LLM;
use vocab::Vocab;

mod llm;
mod embeddings;
mod vocab;
mod transformer;
mod feed_forward;
mod self_attention;
mod output_projection;
mod adam;
mod layer_norm;

// Use the constants from lib.rs
const MAX_SEQ_LEN: usize = 40;
const EMBEDDING_DIM: usize = 50;
const HIDDEN_DIM: usize = 50;

fn main() {
    // Mock input
    let string = String::from("mountains are formed");

    // Extract all unique words from training data to create vocabulary
    let mut vocab_set = std::collections::HashSet::new();
    
    // Add end of sequence token
    vocab_set.insert("</s>".to_string());
    
    let training_data = vec![
        ("the sky is often blue during the day due to the scattering of sunlight by the atmosphere </s>"),
        ("mountains are formed through tectonic forces or volcanism over long geological time periods </s>"),
        ("the amazon rainforest is one of the most biodiverse places on earth </s>"),
        ("water boils at 100 degrees celsius at standard atmospheric pressure </s>"),
        ("the moon orbits the earth approximately every 27.3 days </s>"),
        ("photosynthesis is the process by which green plants use sunlight to synthesize food </s>"),
        ("gravity is a force that attracts two bodies toward each other based on their mass </s>"),
        ("the human brain contains about 86 billion neurons that transmit information </s>"),
        ("electricity is the flow of electrons through a conductor, often used to power devices </s>"),
        ("climate change refers to long-term shifts in temperatures and weather patterns </s>"),

        ("oak trees can live for hundreds of years and produce acorns as their fruit </s>"),
        ("pluto was reclassified from a planet to a dwarf planet in 2006 </s>"),
        ("glass is made by heating sand, soda ash, and limestone to very high temperatures </s>"),
        ("volcanoes can erupt with lava, ash, and gases, altering landscapes and ecosystems </s>"),
        ("the great wall of china was built to protect ancient china from invasions </s>"),
        ("penguins are flightless birds that are well adapted to life in cold environments </s>"),
        ("deserts receive less than 250 millimeters of precipitation each year </s>"),
        ("jupiter is the largest planet in our solar system and has dozens of moons </s>"),
        ("light travels at approximately 299,792 kilometers per second in a vacuum </s>"),
        ("gold is a dense, soft metal often used in jewelry and electronics due to its conductivity </s>"),

        ("most of the earth's surface is covered by water, primarily in oceans </s>"),
        ("bicycles are an efficient mode of transport that convert human energy into motion </s>"),
        ("chocolate is made from roasted and ground cacao seeds, often sweetened and flavored </s>"),
        ("the internet is a global network that allows for digital communication and information sharing </s>"),
        ("wind energy is harnessed using turbines and converted into electricity </s>"),
        ("cats are domesticated mammals known for their independence and hunting instincts </s>"),
        ("languages evolve over time through cultural, social, and technological influences </s>"),
        ("the printing press revolutionized the spread of information in the 15th century </s>"),
        ("sound is a vibration that travels through air, water, or solid materials </s>"),
        ("carbon is an essential element in organic chemistry, forming the basis of life </s>"),

        ("the library of alexandria was one of the most significant libraries of the ancient world </s>"),
        ("honeybees play a vital role in pollination, which supports ecosystems and agriculture </s>"),
        ("electric vehicles produce less air pollution than traditional gasoline-powered cars </s>"),
        ("bread is typically made from flour, water, yeast, and salt through a baking process </s>"),
        ("the sahara desert is the largest hot desert in the world, spanning multiple countries </s>"),
        ("renewable resources replenish naturally and include sunlight, wind, and water </s>"),
        ("eclipses occur when one celestial body moves into the shadow of another </s>"),
        ("language models are trained using vast amounts of text to learn patterns in language </s>"),
        ("compasses work by aligning a magnetic needle with the earth's magnetic field </s>"),
        ("vaccines help the immune system recognize and fight off specific pathogens </s>"),
    ];
    
    // Process all training examples
    for row in &training_data {
        // Add words from outputs
        for word in row.split_whitespace() {
            // Handle punctuation by splitting it from words
            let mut current = String::new();
            for c in word.chars() {
                if c.is_ascii_punctuation() {
                    if !current.is_empty() {
                        vocab_set.insert(current);
                        current = String::new();
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
    
    let vocab_words: Vec<String> = vocab_set.into_iter().collect();
    let vocab_words_refs: Vec<&str> = vocab_words.iter().map(|s| s.as_str()).collect();
    let vocab = Vocab::new(vocab_words_refs);

    let transformer_block_1 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let transformer_block_2 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let transformer_block_3 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let output_projection = OutputProjection::new(EMBEDDING_DIM, vocab.words.len());
    let embeddings = Embeddings::new(vocab.clone());
    let mut llm = LLM::new(vocab, vec![
        Box::new(embeddings),
        Box::new(transformer_block_1),
        Box::new(transformer_block_2),
        Box::new(transformer_block_3),
        Box::new(output_projection),
    ]);

    println!("\n=== MODEL INFORMATION ===");
    println!("Network architecture: {}", llm.network_description());
    
    println!("\n=== BEFORE TRAINING ===");
    println!("Input: {}", string);
    println!("Output: {}", llm.predict(&string));
    
    println!("\n=== TRAINING MODEL ===");
    println!("Training on {} examples for {} epochs with learning rate {}", 
             training_data.len(), 100, 0.001);
    llm.train(training_data, 100, 0.001);
    
    println!("\n=== AFTER TRAINING ===");
    println!("Input: {}", string);
    let result = llm.predict(&string);
    println!("Output: {}", result);
    println!("======================\n");

    // Interactive mode for user input
    println!("\n--- Interactive Mode ---");
    println!("Type a prompt and press Enter to generate text.");
    println!("Type 'exit' to quit.");
    
    let mut input = String::new();
    loop {
        // Clear the input string
        input.clear();
        
        // Prompt for user input
        print!("\nEnter prompt: ");
        std::io::stdout().flush().unwrap();
        
        // Read user input
        std::io::stdin().read_line(&mut input).expect("Failed to read input");
        
        // Trim whitespace and check for exit command
        let trimmed_input = input.trim();
        if trimmed_input.eq_ignore_ascii_case("exit") {
            println!("Exiting interactive mode.");
            break;
        }
        
        // Generate prediction based on user input
        let prediction = llm.predict(trimmed_input);
        println!("Model output: {}", prediction);
    }
}
