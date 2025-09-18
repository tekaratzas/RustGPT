use std::io::Write;

use embeddings::Embeddings;
use llm::LLM;
use output_projection::OutputProjection;
use transformer::TransformerBlock;
use vocab::Vocab;

mod adam;
mod embeddings;
mod feed_forward;
mod layer_norm;
mod llm;
mod output_projection;
mod self_attention;
mod transformer;
mod vocab;

// Use the constants from lib.rs
const MAX_SEQ_LEN: usize = 80;
const EMBEDDING_DIM: usize = 128;
const HIDDEN_DIM: usize = 256;

fn main() {
    // Mock input - test conversational format
    let string = String::from("User: How do mountains form?");

    // Extract all unique words from training data to create vocabulary
    let mut vocab_set = std::collections::HashSet::new();

    // Add end of sequence token
    vocab_set.insert("</s>".to_string());

    // Pre-training data - simple text completion patterns
    let pretraining_data = vec![
        "The sun rises in the east and sets in the west </s>",
        "Water flows downhill due to gravity </s>",
        "Birds fly through the air using their wings </s>",
        "Fish swim in rivers, lakes, and oceans </s>",
        "Trees grow tall and produce leaves </s>",
        "Rain falls from clouds in the sky </s>",
        "Fire is hot and produces light </s>",
        "Ice is frozen water that melts when heated </s>",
        "Mountains are tall and rocky formations </s>",
        "The moon orbits around planet Earth </s>",
        "Flowers bloom in spring and summer </s>",
        "Snow is cold and white </s>",
        "Wind moves air from place to place </s>",
        "Rivers flow into larger bodies of water </s>",
        "Sand is found on beaches and in deserts </s>",
        "Grass grows in fields and yards </s>",
        "Rocks are hard and can be different colors </s>",
        "Stars shine bright in the night sky </s>",
        "Waves move across the surface of water </s>",
        "Clouds form when water vapor rises </s>",
        "Lightning is bright and makes thunder </s>",
        "Storms bring rain and strong winds </s>",
        "Seasons change throughout the year </s>",
        "Animals eat food to survive </s>",
        "Plants need sunlight and water to grow </s>",
    ];

    let chat_training_data = vec![
        // Conversational instruction-following data
        ("User: What causes rain? Assistant: Rain is caused by water vapor in clouds condensing into droplets that become too heavy to remain airborne </s>"),
        ("User: How do mountains form? Assistant: Mountains are formed through tectonic forces or volcanism over long geological time periods </s>"),
        ("User: What is the Amazon rainforest? Assistant: The Amazon rainforest is one of the most biodiverse places on earth, home to countless species </s>"),
        ("User: At what temperature does water boil? Assistant: Water boils at 100 degrees celsius at standard atmospheric pressure </s>"),
        ("User: How long does it take the moon to orbit Earth? Assistant: The moon orbits the earth approximately every 27.3 days </s>"),
        ("User: What is photosynthesis? Assistant: Photosynthesis is the process by which green plants use sunlight to synthesize food from carbon dioxide </s>"),
        ("User: How does gravity work? Assistant: Gravity is a force that attracts two bodies toward each other based on their mass </s>"),
        ("User: How many neurons are in the human brain? Assistant: The human brain contains about 86 billion neurons that transmit information throughout the body </s>"),
        ("User: What is electricity? Assistant: Electricity is the flow of electrons through a conductor, often used to power devices </s>"),
        ("User: What is climate change? Assistant: Climate change refers to long-term shifts in temperatures and weather patterns on Earth </s>"),
        ("User: How long do oak trees live? Assistant: Oak trees can live for hundreds of years and produce acorns as their fruit </s>"),
        ("User: What happened to Pluto? Assistant: Pluto was reclassified from a planet to a dwarf planet in 2006 by astronomers </s>"),
        ("User: How is glass made? Assistant: Glass is made by heating sand, soda ash, and limestone to very high temperatures until they melt </s>"),
        ("User: What do volcanoes do? Assistant: Volcanoes can erupt with lava, ash, and gases, altering landscapes and ecosystems </s>"),
        ("User: Why was the Great Wall built? Assistant: The great wall of china was built to protect ancient china from invasions </s>"),
        ("User: What are penguins like? Assistant: Penguins are flightless birds that are well adapted to life in cold environments </s>"),
        ("User: How much rain do deserts get? Assistant: Deserts receive less than 250 millimeters of precipitation each year </s>"),
        ("User: What is Jupiter like? Assistant: Jupiter is the largest planet in our solar system and has dozens of moons </s>"),
        ("User: How fast does light travel? Assistant: Light travels at approximately 299,792 kilometers per second in a vacuum </s>"),
        ("User: What is gold used for? Assistant: Gold is a dense, soft metal often used in jewelry and electronics due to its conductivity </s>"),
        ("User: What covers most of Earth? Assistant: Most of the earth's surface is covered by water, primarily in oceans </s>"),
        ("User: How do bicycles work? Assistant: Bicycles are an efficient mode of transport that convert human energy into motion through gears </s>"),
        ("User: How is chocolate made? Assistant: Chocolate is made from roasted and ground cacao seeds, often sweetened and flavored </s>"),
        ("User: What is the internet? Assistant: The internet is a global network that allows for digital communication and information sharing </s>"),
        ("User: How do wind turbines work? Assistant: Wind energy is harnessed using turbines that convert wind motion into electricity </s>"),
        ("User: What are cats like? Assistant: Cats are domesticated mammals known for their independence and hunting instincts </s>"),
        ("User: How do languages change? Assistant: Languages evolve over time through cultural, social, and technological influences </s>"),
        ("User: What did the printing press do? Assistant: The printing press revolutionized the spread of information in the 15th century </s>"),
        ("User: What is sound? Assistant: Sound is a vibration that travels through air, water, or solid materials to reach our ears </s>"),
        ("User: Why is carbon important? Assistant: Carbon is an essential element in organic chemistry, forming the basis of all life on Earth </s>"),
        ("User: What was the Library of Alexandria? Assistant: The library of alexandria was one of the most significant libraries of the ancient world </s>"),
        ("User: What do honeybees do? Assistant: Honeybees play a vital role in pollination, which supports ecosystems and agriculture </s>"),
        ("User: Are electric cars better? Assistant: Electric vehicles produce less air pollution than traditional gasoline-powered cars </s>"),
        ("User: How do you make bread? Assistant: Bread is typically made from flour, water, yeast, and salt through a baking process </s>"),
        ("User: What is the Sahara desert? Assistant: The sahara desert is the largest hot desert in the world, spanning multiple countries </s>"),
        ("User: What are renewable resources? Assistant: Renewable resources replenish naturally and include sunlight, wind, and water </s>"),
        ("User: What are eclipses? Assistant: Eclipses occur when one celestial body moves into the shadow of another </s>"),
        ("User: How are language models trained? Assistant: Language models are trained using vast amounts of text to learn patterns in language </s>"),
        ("User: How do compasses work? Assistant: Compasses work by aligning a magnetic needle with the earth's magnetic field </s>"),
        ("User: What do vaccines do? Assistant: Vaccines help the immune system recognize and fight off specific pathogens </s>"),
        // Greeting and conversational patterns
        ("User: Hello! Assistant: Hello! How can I help you today? </s>"),
        ("User: Hi! Assistant: Hi there! What can I do for you? </s>"),
        ("User: Hey! Assistant: Hey! How's it going? What would you like to know? </s>"),
        ("User: Good morning! Assistant: Good morning! Hope you're having a great day. How can I assist you? </s>"),
        ("User: Good afternoon! Assistant: Good afternoon! What can I help you with today? </s>"),
        ("User: Good evening! Assistant: Good evening! How may I help you tonight? </s>"),
        ("User: How are you? Assistant: I'm doing well, thank you for asking! How can I help you today? </s>"),
        ("User: What's up? Assistant: Not much, just here to help! What would you like to know? </s>"),
        ("User: Greetings! Assistant: Greetings! It's nice to meet you. How can I be of assistance? </s>"),
        ("User: Thank you. Assistant: You're welcome! I'm happy to help </s>"),
        ("User: Can you explain that? Assistant: Of course! I'd be happy to explain in more detail </s>"),
        ("User: I don't understand. Assistant: No problem! Let me try to explain it differently </s>"),
        ("User: What do you think? Assistant: Based on the information available, I think it's important to consider multiple perspectives </s>"),
    ];

    // Process all training examples for vocabulary
    // First process pre-training data
    for text in &pretraining_data {
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

    // Then process chat training data
    for row in &chat_training_data {
        // Add words from outputs
        for word in row.split_whitespace() {
            // Handle punctuation by splitting it from words
            let mut current = String::new();
            for c in word.chars() {
                if c.is_ascii_punctuation() {
                    if !current.is_empty() {
                        vocab_set.insert(current.clone()); // Clone to avoid moving
                        current.clear(); // Use clear() instead of String::new()
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

    let mut vocab_words: Vec<String> = vocab_set.into_iter().collect();
    vocab_words.sort(); // Sort for deterministic ordering
    let vocab_words_refs: Vec<&str> = vocab_words.iter().map(|s| s.as_str()).collect();
    let vocab = Vocab::new(vocab_words_refs);

    let transformer_block_1 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let transformer_block_2 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let transformer_block_3 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let output_projection = OutputProjection::new(EMBEDDING_DIM, vocab.words.len());
    let embeddings = Embeddings::new(vocab.clone());
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

    println!("\n=== MODEL INFORMATION ===");
    println!("Network architecture: {}", llm.network_description());

    println!("\n=== BEFORE TRAINING ===");
    println!("Input: {}", string);
    println!("Output: {}", llm.predict(&string));

    println!("\n=== PRE-TRAINING MODEL ===");
    println!(
        "Pre-training on {} examples for {} epochs with learning rate {}",
        pretraining_data.len(),
        100,
        0.0005
    );
    llm.train(pretraining_data, 100, 0.0005);

    println!("\n=== INSTRUCTION TUNING ===");
    println!(
        "Instruction tuning on {} examples for {} epochs with learning rate {}",
        chat_training_data.len(),
        100,
        0.0001
    );
    llm.train(chat_training_data, 100, 0.0001); // Much lower learning rate for stability

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
        std::io::stdin()
            .read_line(&mut input)
            .expect("Failed to read input");

        // Trim whitespace and check for exit command
        let trimmed_input = input.trim();
        if trimmed_input.eq_ignore_ascii_case("exit") {
            println!("Exiting interactive mode.");
            break;
        }

        // Generate prediction based on user input with "User:" prefix
        let formatted_input = format!("User: {}", trimmed_input);
        let prediction = llm.predict(&formatted_input);
        println!("Model output: {}", prediction);
    }
}
