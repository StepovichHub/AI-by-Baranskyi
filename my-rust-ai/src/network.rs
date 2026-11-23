use crate::tensor::Tensor;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative_from_output(y: f64) -> f64 {
    y * (1.0 - y)
}

/// Простий генератор псевдовипадкових чисел у діапазоні [-1.0, 1.0] без зовнішніх залежностей.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let normalized = ((self.state >> 33) as f64) / ((1u64 << 31) as f64);
        normalized * 2.0 - 1.0
    }
}

pub struct NeuralNetwork {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
}

impl NeuralNetwork {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, seed: u64) -> Self {
        let mut rng = SimpleRng::new(seed);

        let w1_data: Vec<f64> = (0..input_size * hidden_size)
            .map(|_| rng.next_f64())
            .collect();
        let b1_data: Vec<f64> = (0..hidden_size).map(|_| rng.next_f64()).collect();

        let w2_data: Vec<f64> = (0..hidden_size * output_size)
            .map(|_| rng.next_f64())
            .collect();
        let b2_data: Vec<f64> = (0..output_size).map(|_| rng.next_f64()).collect();

        Self {
            input_size,
            hidden_size,
            output_size,
            w1: Tensor::new(input_size, hidden_size, w1_data),
            b1: Tensor::new(1, hidden_size, b1_data),
            w2: Tensor::new(hidden_size, output_size, w2_data),
            b2: Tensor::new(1, output_size, b2_data),
        }
    }

    fn forward_with_cache(&self, x: &Tensor) -> (Tensor, Tensor, Tensor) {
        assert_eq!(x.cols, self.input_size);
        let z1 = x.matmul(&self.w1).add(&self.b1);
        let a1 = z1.map(sigmoid);

        let z2 = a1.matmul(&self.w2).add(&self.b2);
        let a2 = z2.map(sigmoid);

        (x.clone(), a1, a2)
    }

    pub fn predict(&self, input: &[f64]) -> Vec<f64> {
        let x = Tensor::from_row_vector(input);
        let (_, _, out) = self.forward_with_cache(&x);
        out.data
    }

    pub fn train_sample(&mut self, input: &[f64], target: &[f64], learning_rate: f64) -> f64 {
        assert_eq!(input.len(), self.input_size);
        assert_eq!(target.len(), self.output_size);

        let x = Tensor::from_row_vector(input);
        let y_true = Tensor::from_row_vector(target);

        let (x_cache, a1, a2) = self.forward_with_cache(&x);

        // Помилка та градієнт вихідного шару (MSE похідна * похідна сигмоїди)
        let mut err2_data = Vec::with_capacity(self.output_size);
        let mut loss = 0.0;
        for i in 0..self.output_size {
            let diff = a2.data[i] - y_true.data[i];
            loss += 0.5 * diff * diff;
            let delta = diff * sigmoid_derivative_from_output(a2.data[i]);
            err2_data.push(delta);
        }
        let delta2 = Tensor::new(1, self.output_size, err2_data);

        // Градієнти для W2 та b2
        let grad_w2 = a1.transpose().matmul(&delta2);
        let grad_b2 = delta2.clone();

        // Помилка та градієнт прихованого шару
        let hidden_err = delta2.matmul(&self.w2.transpose());
        let mut err1_data = Vec::with_capacity(self.hidden_size);
        for j in 0..self.hidden_size {
            let delta = hidden_err.data[j] * sigmoid_derivative_from_output(a1.data[j]);
            err1_data.push(delta);
        }
        let delta1 = Tensor::new(1, self.hidden_size, err1_data);

        // Градієнти для W1 та b1
        let grad_w1 = x_cache.transpose().matmul(&delta1);
        let grad_b1 = delta1;

        // Оновлення ваг і зсувів
        self.w2 = self.w2.sub(&grad_w2.map(|g| learning_rate * g));
        self.b2 = self.b2.sub(&grad_b2.map(|g| learning_rate * g));
        self.w1 = self.w1.sub(&grad_w1.map(|g| learning_rate * g));
        self.b1 = self.b1.sub(&grad_b1.map(|g| learning_rate * g));

        loss
    }

    pub fn fit(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
        epochs: usize,
        learning_rate: f64,
    ) {
        assert_eq!(inputs.len(), targets.len());
        for _epoch in 0..epochs {
            for (x, y) in inputs.iter().zip(targets.iter()) {
                self.train_sample(x, y, learning_rate);
            }
        }
    }
}
