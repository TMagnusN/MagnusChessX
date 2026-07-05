use crate::config::OUTPUT_BUCKETS;

#[derive(Clone, Debug, Default)]
pub struct CountStats {
    samples: Vec<usize>,
}

impl CountStats {
    pub fn push(&mut self, value: usize) {
        self.samples.push(value);
    }
    pub fn mean(&self) -> f64 {
        if self.samples.is_empty() {
            0.0
        } else {
            self.samples.iter().sum::<usize>() as f64 / self.samples.len() as f64
        }
    }
    pub fn percentile(&self, percentile: usize) -> usize {
        if self.samples.is_empty() {
            return 0;
        }
        let mut values = self.samples.clone();
        values.sort_unstable();
        let idx = ((values.len() - 1) * percentile).div_ceil(100);
        values[idx.min(values.len() - 1)]
    }
    pub fn max(&self) -> usize {
        self.samples.iter().copied().max().unwrap_or(0)
    }
}

#[derive(Clone, Debug)]
pub struct DatasetStats {
    pub accepted: u64,
    pub rejected: u64,
    pub position: CountStats,
    pub attack: CountStats,
    pub structure: CountStats,
    pub buckets: [u64; OUTPUT_BUCKETS],
}

impl Default for DatasetStats {
    fn default() -> Self {
        Self {
            accepted: 0,
            rejected: 0,
            position: CountStats::default(),
            attack: CountStats::default(),
            structure: CountStats::default(),
            buckets: [0; OUTPUT_BUCKETS],
        }
    }
}

impl DatasetStats {
    pub fn print_branch(name: &str, vocabulary: usize, stats: &CountStats) {
        println!("{name} features:");
        println!("  vocabulary: {vocabulary}");
        println!("  mean active: {:.2}", stats.mean());
        println!("  p50 active: {}", stats.percentile(50));
        println!("  p95 active: {}", stats.percentile(95));
        println!("  p99 active: {}", stats.percentile(99));
        println!("  maximum active: {}", stats.max());
    }

    pub fn print(&self, vocabularies: (usize, usize, usize)) {
        Self::print_branch("Position", vocabularies.0, &self.position);
        Self::print_branch("Attack", vocabularies.1, &self.attack);
        Self::print_branch("Structure", vocabularies.2, &self.structure);
        println!("Material buckets: {:?}", self.buckets);
        let rare = self.accepted / 1000;
        for (bucket, &count) in self.buckets.iter().enumerate() {
            if count <= rare.max(1) {
                eprintln!("warning: material bucket {bucket} has only {count} samples");
            }
        }
        println!("Accepted positions processed: {}", self.accepted);
        println!("Rejected positions: {}", self.rejected);
    }
}
