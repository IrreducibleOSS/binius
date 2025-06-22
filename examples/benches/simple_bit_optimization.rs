// Copyright 2024-2025 Irreducible Inc.

//! 1-bit布尔AND门零检查协议优化基准测试
//! 对比标准字段方法与位向量优化方法的性能差异，与 binary_zerocheck.rs 对比

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use binius_field::{BinaryField1b, Field};
use rand::{SeedableRng, RngCore, rngs::StdRng};

/// 简单位向量实现，用于1-bit值存储和操作
#[derive(Debug, Clone)]
struct SimpleBitVec {
    bits: Vec<bool>,
}

impl SimpleBitVec {
    fn new(bits: Vec<bool>) -> Self {
        Self { bits }
    }
    
    fn len(&self) -> usize {
        self.bits.len()
    }
    
    /// 验证AND约束: c = a & b (批量验证)
    fn verify_and_constraints_batch(&self, a: &Self, b: &Self) -> bool {
        if self.len() != a.len() || a.len() != b.len() {
            return false;
        }
        
        // 向量化验证，模拟实际零检查协议的批量处理
        for i in 0..self.len() {
            if self.bits[i] != (a.bits[i] && b.bits[i]) {
                return false;
            }
        }
        true
    }
    
    /// 按位AND操作
    fn bitwise_and(&self, other: &Self) -> Self {
        let bits = self.bits.iter()
            .zip(other.bits.iter())
            .map(|(&a, &b)| a && b)
            .collect();
        Self::new(bits)
    }
}

/// 生成与 binary_zerocheck.rs 相同规模的测试数据
fn generate_zerocheck_data(n_vars: usize, seed: u64) -> (Vec<BinaryField1b>, Vec<BinaryField1b>, Vec<BinaryField1b>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let size = 1 << n_vars;
    
    // 生成随机的F₂元素
    let a_field: Vec<BinaryField1b> = (0..size)
        .map(|_| if rng.next_u32() & 1 != 0 { 
            BinaryField1b::ONE 
        } else { 
            BinaryField1b::ZERO 
        })
        .collect();
        
    let b_field: Vec<BinaryField1b> = (0..size)
        .map(|_| if rng.next_u32() & 1 != 0 { 
            BinaryField1b::ONE 
        } else { 
            BinaryField1b::ZERO 
        })
        .collect();
    
    // c = a * b (对于F₂，乘法等同于AND)
    let c_field: Vec<BinaryField1b> = a_field.iter()
        .zip(b_field.iter())
        .map(|(&a, &b)| a * b)
        .collect();
    
    (a_field, b_field, c_field)
}

/// 将字段元素转换为位向量
fn field_to_bits(field_vec: &[BinaryField1b]) -> Vec<bool> {
    field_vec.iter()
        .map(|&f| f == BinaryField1b::ONE)
        .collect()
}

/// 标准字段方法基准测试 (对应 binary_zerocheck.rs)
fn bench_standard_field_zerocheck(c: &mut Criterion) {
    let n_vars = 24usize; // 16M elements - 与 binary_zerocheck.rs 相同
    
    let mut group = c.benchmark_group("zerocheck_comparison");
    
    // 预先生成数据，避免在测试中重复生成
    let (a_field, b_field, c_field) = generate_zerocheck_data(n_vars, 0);
    
    group.throughput(Throughput::Elements((1 << n_vars) as u64));
    group.bench_function("standard_field_method", |bench| {
        bench.iter(|| {
            // 只测试核心约束验证逻辑
            let mut valid = true;
            for i in 0..a_field.len() {
                let expected = a_field[i] * b_field[i];
                if c_field[i] != expected {
                    valid = false;
                    break; // 早期退出，避免不必要的计算
                }
            }
            valid
        });
    });
    group.finish()
}

/// 位向量优化方法基准测试
fn bench_bit_optimized_zerocheck(c: &mut Criterion) {
    let n_vars = 24usize; // 16M elements - 与标准方法相同规模
    
    let mut group = c.benchmark_group("zerocheck_comparison");
    
    // 预先生成并转换数据
    let (a_field, b_field, c_field) = generate_zerocheck_data(n_vars, 0);
    let a_bits = field_to_bits(&a_field);
    let b_bits = field_to_bits(&b_field);
    let c_bits = field_to_bits(&c_field);
    
    let a_vec = SimpleBitVec::new(a_bits);
    let b_vec = SimpleBitVec::new(b_bits);
    let c_vec = SimpleBitVec::new(c_bits);
    
    group.throughput(Throughput::Elements((1 << n_vars) as u64));
    group.bench_function("bit_optimized_method", |bench| {
        bench.iter(|| {
            // 只测试核心验证逻辑
            c_vec.verify_and_constraints_batch(&a_vec, &b_vec)
        });
    });
    group.finish()
}

/// 简化的内存对比测试
fn bench_memory_usage_comparison(c: &mut Criterion) {
    let n_vars = 20usize; // 使用较小规模避免内存压力
    
    let mut group = c.benchmark_group("memory_efficiency");
    
    group.throughput(Throughput::Elements((1 << n_vars) as u64));
    
    // 标准字段内存使用
    group.bench_function("field_memory_usage", |bench| {
        bench.iter(|| {
            let (a, b, c) = generate_zerocheck_data(n_vars, 0);
            std::hint::black_box((a.len(), b.len(), c.len()))
        });
    });
    
    // 位向量内存使用
    group.bench_function("bit_memory_usage", |bench| {
        bench.iter(|| {
            let (a_field, b_field, c_field) = generate_zerocheck_data(n_vars, 0);
            let a_bits = field_to_bits(&a_field);
            let b_bits = field_to_bits(&b_field);
            let c_bits = field_to_bits(&c_field);
            std::hint::black_box((a_bits.len(), b_bits.len(), c_bits.len()))
        });
    });
    
    group.finish()
}

criterion_group! {
    name = bit_optimization_zerocheck;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(60));
    targets = bench_standard_field_zerocheck, bench_bit_optimized_zerocheck, bench_memory_usage_comparison
}
criterion_main!(bit_optimization_zerocheck);
