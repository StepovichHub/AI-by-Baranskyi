use my_rust_ai::NeuralNetwork;

fn main() {
    println!("Власний ШІ на Rust ініціалізовано.");
    println!("Запустіть приклад XOR командою: cargo run --example xor");

    let model = NeuralNetwork::new(2, 4, 1, 42);
    let sample_out = model.predict(&[0.0, 1.0]);
    println!("Тестовий прогноз до навчання [0.0, 1.0] -> {:.4}", sample_out[0]);
}