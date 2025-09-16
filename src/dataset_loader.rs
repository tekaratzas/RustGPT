pub struct Dataset {
    pub pretraining_data: Vec<String>,
    pub chat_training_data: Vec<String>,
}

impl Dataset {
    pub fn new(pretraining_data_path: String, chat_training_data_path: String) -> Self {
        use std::fs;
        use serde_json;

        // Load pretraining data from JSON
        let pretraining_json = fs::read_to_string(pretraining_data_path)
            .expect("Failed to read pretraining_data.json");
        let pretraining_data: Vec<String> = serde_json::from_str(&pretraining_json)
            .expect("Failed to parse pretraining_data.json");

        // Load chat training data from JSON
        let chat_json = fs::read_to_string("data/chat_training_data.json")
            .expect("Failed to read chat_training_data.json");
        let chat_training_data: Vec<String> = serde_json::from_str(&chat_json)
            .expect("Failed to parse chat_training_data.json");

        Dataset {
            pretraining_data: pretraining_data.clone(),
            chat_training_data: chat_training_data.clone(),
        }
    }
}