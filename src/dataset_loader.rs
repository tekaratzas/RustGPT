use csv::ReaderBuilder;
use serde_json;
use std::fs;

pub struct Dataset {
    pub pretraining_data: Vec<String>,
    pub chat_training_data: Vec<String>,
}

#[allow(dead_code)]
pub enum DatasetType {
    JSON,
    CSV,
}

impl Dataset {
    pub fn new(
        pretraining_data_path: String,
        chat_training_data_path: String,
        type_of_data: DatasetType,
    ) -> Self {
        let pretraining_data: Vec<String>;
        let chat_training_data: Vec<String>;

        match type_of_data {
            DatasetType::CSV => {
                pretraining_data = get_data_from_csv(pretraining_data_path);
                chat_training_data = get_data_from_csv(chat_training_data_path);
            }
            DatasetType::JSON => {
                pretraining_data = get_data_from_json(pretraining_data_path);
                chat_training_data = get_data_from_json(chat_training_data_path);
            }
        }

        Dataset {
            pretraining_data: pretraining_data.clone(),
            chat_training_data: chat_training_data.clone(),
        }
    }
}

fn get_data_from_json(path: String) -> Vec<String> {
    // convert json file to Vec<String>
    let data_json = fs::read_to_string(path).expect("Failed to read data file");
    let data: Vec<String> = serde_json::from_str(&data_json).expect("Failed to parse data file");
    data
}

fn get_data_from_csv(path: String) -> Vec<String> {
    // convert csv file to Vec<String>
    let file = fs::File::open(path).expect("Failed to open CSV file");
    let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);
    let mut data = Vec::new();

    for result in rdr.records() {
        let record = result.expect("Failed to read CSV record");
        // Each record is a row, join all columns into a single string
        data.push(record.iter().collect::<Vec<_>>().join(","));
    }
    data
}
