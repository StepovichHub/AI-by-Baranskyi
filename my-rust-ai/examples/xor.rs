use my_rust_ai::NeuralNetwork;

fn main() {
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
    ];

    let mut nn = NeuralNetwork::new(2, 4, 1, 12345);

    println!("До навчання:");
    for (x, y) in inputs.iter().zip(targets.iter()) {
        let pred = nn.predict(x)[0];
        println!("  вхід: {:?} | ціль: {:.0} | прогноз: {:.4}", x, y[0], pred);
    }

    nn.fit(&inputs, &targets, 15_000, 1.5);

    println!("\nПісля навчання:");
    for (x, y) in inputs.iter().zip(targets.iter()) {
        let pred = nn.predict(x)[0];
        println!(
            "  вхід: {:?} | ціль: {:.0} | прогноз: {:.4} (округлено: {:.0})",
            x,
            y[0],
            pred,
            pred.round()
        );
    }
}