use embeddings::Embeddings;
use llm::LLM;
use vocab::Vocab;

mod llm;
mod embeddings;
mod vocab;
fn main() {
    // Mock input
    let string = String::from("Hello world");

    let llm = LLM::default();

    let result = llm.predict(&string);
    println!("output of LLM: {}", result);
}
