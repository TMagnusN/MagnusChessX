use std::{fs::File, sync::mpsc, thread};

use bullet_lib::value::loader::DataLoader;
use bulletformat::ChessBoard;
use sfbinpack::{
    CompressedTrainingDataEntryReader, TrainingDataEntry,
    chess::{color::Color, piecetype::PieceType},
};

#[derive(Clone)]
pub struct ResamplingSfBinpackLoader<T: Fn(&TrainingDataEntry) -> bool> {
    chunk_paths: Vec<String>,
    buffer_size: usize,
    threads: usize,
    filter: T,
    seed: u64,
    virtual_epochs: usize,
    chunk_sample: Option<usize>,
}

impl<T: Fn(&TrainingDataEntry) -> bool> ResamplingSfBinpackLoader<T> {
    pub fn new(
        chunk_paths: Vec<String>,
        buffer_size_mb: usize,
        threads: usize,
        filter: T,
        seed: u64,
        virtual_epochs: usize,
        chunk_sample: Option<usize>,
    ) -> Self {
        assert!(!chunk_paths.is_empty());
        assert!(threads > 0);
        assert!(virtual_epochs > 0);
        if let Some(sample) = chunk_sample {
            assert!(sample > 0 && sample <= chunk_paths.len());
        }

        Self {
            chunk_paths,
            buffer_size: buffer_size_mb * 1024 * 1024 / std::mem::size_of::<ChessBoard>() / 2,
            threads,
            filter,
            seed,
            virtual_epochs,
            chunk_sample,
        }
    }
}

impl<T> DataLoader<ChessBoard> for ResamplingSfBinpackLoader<T>
where
    T: Fn(&TrainingDataEntry) -> bool + Clone + Send + Sync + 'static,
{
    fn data_file_paths(&self) -> &[String] {
        &self.chunk_paths
    }

    fn count_positions(&self) -> Option<u64> {
        None
    }

    fn map_chunks<F: FnMut(&[ChessBoard]) -> bool>(&self, _: usize, mut f: F) {
        let chunk_paths = self.chunk_paths.clone();
        let buffer_size = self.buffer_size;
        let threads = self.threads;
        let filter = self.filter.clone();
        let seed = self.seed;
        let virtual_epochs = self.virtual_epochs;
        let chunk_sample = self.chunk_sample;

        let reader_buffer_size = 16384 * threads;
        let (reader_sender, reader_receiver) = mpsc::sync_channel::<Vec<TrainingDataEntry>>(8);
        let (reader_msg_sender, reader_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            let mut buffer = Vec::with_capacity(reader_buffer_size);
            let mut schedule =
                ResamplingChunkSchedule::new(chunk_paths, seed, virtual_epochs, chunk_sample);

            'dataloading: loop {
                let pool = schedule.next_pool();
                for file in pool {
                    let file = File::open(&file).unwrap();
                    let mut reader = CompressedTrainingDataEntryReader::new(file).unwrap();

                    while reader.has_next() {
                        buffer.push(reader.next());

                        if buffer.len() == reader_buffer_size || !reader.has_next() {
                            if reader_msg_receiver.try_recv().unwrap_or(false)
                                || reader_sender.send(buffer).is_err()
                            {
                                break 'dataloading;
                            }

                            buffer = Vec::with_capacity(reader_buffer_size);
                        }
                    }
                }
            }
        });

        let (converted_sender, converted_receiver) =
            mpsc::sync_channel::<Vec<ChessBoard>>(4 * threads);
        let (converted_msg_sender, converted_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            let filter = &filter;
            let mut should_break = false;
            'dataloading: while let Ok(unfiltered) = reader_receiver.recv() {
                if should_break || converted_msg_receiver.try_recv().unwrap_or(false) {
                    reader_msg_sender.send(true).unwrap();
                    break 'dataloading;
                }

                thread::scope(|s| {
                    let chunk_size = unfiltered.len().div_ceil(threads);
                    let mut handles = Vec::new();

                    for chunk in unfiltered.chunks(chunk_size) {
                        let this_sender = converted_sender.clone();
                        let handle = s.spawn(move || {
                            let mut buffer = Vec::with_capacity(chunk_size);

                            for entry in chunk {
                                if filter(entry) {
                                    buffer.push(convert_to_bulletformat(entry));
                                }
                            }

                            this_sender.send(buffer).is_err()
                        });

                        handles.push(handle);
                    }

                    for handle in handles {
                        if handle.join().unwrap() {
                            should_break = true;
                        }
                    }
                });

                if should_break {
                    reader_msg_sender.send(true).unwrap();
                    break 'dataloading;
                }
            }
        });

        let (buffer_sender, buffer_receiver) = mpsc::sync_channel::<Vec<ChessBoard>>(0);
        let (buffer_msg_sender, buffer_msg_receiver) = mpsc::sync_channel::<bool>(1);

        std::thread::spawn(move || {
            let mut shuffle_buffer = Vec::with_capacity(buffer_size);

            'dataloading: while let Ok(converted) = converted_receiver.recv() {
                for entry in converted {
                    shuffle_buffer.push(entry);

                    if shuffle_buffer.len() == buffer_size {
                        shuffle(&mut shuffle_buffer);

                        if buffer_msg_receiver.try_recv().unwrap_or(false)
                            || buffer_sender.send(shuffle_buffer).is_err()
                        {
                            converted_msg_sender.send(true).unwrap();
                            break 'dataloading;
                        }

                        shuffle_buffer = Vec::with_capacity(buffer_size);
                    }
                }
            }
        });

        'dataloading: while let Ok(shuffle_buffer) = buffer_receiver.recv() {
            if f(&shuffle_buffer) {
                buffer_msg_sender.send(true).unwrap();
                break 'dataloading;
            }
        }
    }
}

struct ResamplingChunkSchedule {
    chunks: Vec<String>,
    rng: SplitMix64,
    virtual_epochs: usize,
    chunk_sample: Option<usize>,
}

impl ResamplingChunkSchedule {
    fn new(
        chunks: Vec<String>,
        seed: u64,
        virtual_epochs: usize,
        chunk_sample: Option<usize>,
    ) -> Self {
        Self {
            chunks,
            rng: SplitMix64::new(seed),
            virtual_epochs,
            chunk_sample,
        }
    }

    fn next_pool(&mut self) -> Vec<String> {
        let per_epoch = self.chunk_sample.unwrap_or(self.chunks.len());
        let mut pool = Vec::with_capacity(per_epoch * self.virtual_epochs);
        for _ in 0..self.virtual_epochs {
            let mut epoch = self.chunks.clone();
            fisher_yates_shuffle(&mut epoch, &mut self.rng);
            pool.extend(epoch.into_iter().take(per_epoch));
        }
        pool
    }
}

fn convert_to_bulletformat(entry: &TrainingDataEntry) -> ChessBoard {
    let mut bbs = [0; 8];

    let stm = usize::from(entry.pos.side_to_move().ordinal());
    let pc_bb = |pt| {
        entry.pos.pieces_bb_color(Color::Black, pt).bits()
            | entry.pos.pieces_bb_color(Color::White, pt).bits()
    };

    bbs[0] = entry.pos.pieces_bb(Color::White).bits();
    bbs[1] = entry.pos.pieces_bb(Color::Black).bits();
    bbs[2] = pc_bb(PieceType::Pawn);
    bbs[3] = pc_bb(PieceType::Knight);
    bbs[4] = pc_bb(PieceType::Bishop);
    bbs[5] = pc_bb(PieceType::Rook);
    bbs[6] = pc_bb(PieceType::Queen);
    bbs[7] = pc_bb(PieceType::King);

    let mut score = entry.score;
    let mut result = f32::from(1 + entry.result) / 2.0;

    if stm > 0 {
        score = -score;
        result = 1.0 - result;
    }

    ChessBoard::from_raw(bbs, stm, score, result).expect("Binpack must be malformed!")
}

fn shuffle(data: &mut [ChessBoard]) {
    let mut rng = SplitMix64::new(0xB17B_1A5E_DA7A_5EED);

    for i in (1..data.len()).rev() {
        let idx = rng.next_bounded(i + 1);
        data.swap(idx, i);
    }
}

fn fisher_yates_shuffle<T>(items: &mut [T], rng: &mut SplitMix64) {
    for i in (1..items.len()).rev() {
        let j = rng.next_bounded(i + 1);
        items.swap(i, j);
    }
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_bounded(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0);
        (self.next_u64() % upper as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampling_pools_are_not_repeated() {
        let mut schedule = ResamplingChunkSchedule::new(
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ],
            1,
            3,
            None,
        );
        let first = schedule.next_pool();
        let second = schedule.next_pool();
        assert_ne!(first, second);
    }

    #[test]
    fn sample_mode_keeps_distinct_chunks_per_epoch() {
        let mut schedule = ResamplingChunkSchedule::new(
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            7,
            5,
            Some(2),
        );
        let pool = schedule.next_pool();
        assert_eq!(pool.len(), 10);
        for epoch in pool.chunks(2) {
            assert_eq!(epoch.len(), 2);
            assert_ne!(epoch[0], epoch[1]);
        }
    }
}
