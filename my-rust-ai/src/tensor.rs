#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Tensor {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(
            rows * cols,
            data.len(),
            "Розмір даних ({}) не відповідає розмірності {}x{}",
            data.len(),
            rows,
            cols
        );
        Self { rows, cols, data }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn from_row_vector(values: &[f64]) -> Self {
        Self {
            rows: 1,
            cols: values.len(),
            data: values.to_vec(),
        }
    }

    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        self.data[r * self.cols + c] = val;
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols));
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Tensor::new(self.rows, self.cols, data)
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols));
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Tensor::new(self.rows, self.cols, data)
    }

    pub fn matmul(&self, rhs: &Tensor) -> Tensor {
        assert_eq!(
            self.cols, rhs.rows,
            "Несумісні розмірності для множення матриць: {}x{} та {}x{}",
            self.rows, self.cols, rhs.rows, rhs.cols
        );
        let mut out = vec![0.0; self.rows * rhs.cols];
        for r in 0..self.rows {
            for k in 0..self.cols {
                let a_val = self.get(r, k);
                for c in 0..rhs.cols {
                    out[r * rhs.cols + c] += a_val * rhs.get(k, c);
                }
            }
        }
        Tensor::new(self.rows, rhs.cols, out)
    }

    pub fn transpose(&self) -> Tensor {
        let mut out = vec![0.0; self.rows * self.cols];
        for r in 0..self.rows {
            for c in 0..self.cols {
                out[c * self.rows + r] = self.get(r, c);
            }
        }
        Tensor::new(self.cols, self.rows, out)
    }

    pub fn map<F>(&self, f: F) -> Tensor
    where
        F: Fn(f64) -> f64,
    {
        let data = self.data.iter().copied().map(f).collect();
        Tensor::new(self.rows, self.cols, data)
    }
}
