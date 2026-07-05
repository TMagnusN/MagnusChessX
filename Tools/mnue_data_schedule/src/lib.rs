use std::{
    collections::BTreeMap,
    error::Error,
    fmt, fs, io,
    path::{Path, PathBuf},
};

pub const DEFAULT_CHUNK_SHUFFLE_SEED: u64 = 1;
pub const DEFAULT_CHUNK_VIRTUAL_EPOCHS: usize = 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InputDataKind {
    Direct,
    SfBinpack,
}

#[derive(Clone, Debug)]
pub struct DataScheduleOptions {
    pub explicit_paths: Vec<PathBuf>,
    pub data_dirs: Vec<PathBuf>,
    pub chunk_shuffle_seed: u64,
    pub chunk_virtual_epochs: usize,
    pub chunk_sample: Option<usize>,
}

impl Default for DataScheduleOptions {
    fn default() -> Self {
        Self {
            explicit_paths: Vec::new(),
            data_dirs: Vec::new(),
            chunk_shuffle_seed: DEFAULT_CHUNK_SHUFFLE_SEED,
            chunk_virtual_epochs: DEFAULT_CHUNK_VIRTUAL_EPOCHS,
            chunk_sample: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DataSchedule {
    pub kind: InputDataKind,
    pub scheduled_paths: Vec<PathBuf>,
    pub explicit_path_count: usize,
    pub discovered_chunk_count: usize,
    pub unique_chunk_count: usize,
    pub seed: u64,
    pub virtual_epochs: usize,
    pub chunk_sample: Option<usize>,
}

impl DataSchedule {
    pub fn path_strings(&self) -> Vec<String> {
        self.scheduled_paths
            .iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect()
    }
}

#[derive(Debug)]
pub enum DataScheduleError {
    EmptyInputs,
    MissingExplicitPath(PathBuf),
    DataDirNotFound(PathBuf),
    DataDirIsNotDirectory(PathBuf),
    DataDirHasNoBinpacks(PathBuf),
    MixedInputKinds,
    InvalidVirtualEpochs,
    InvalidChunkSample,
    ChunkSampleTooLarge { sample: usize, chunks: usize },
    Io { path: PathBuf, source: io::Error },
}

impl fmt::Display for DataScheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInputs => write!(f, "--data or --data-dir must be supplied at least once"),
            Self::MissingExplicitPath(path) => {
                write!(f, "training data file not found: {}", path.display())
            }
            Self::DataDirNotFound(path) => {
                write!(f, "--data-dir does not exist: {}", path.display())
            }
            Self::DataDirIsNotDirectory(path) => {
                write!(f, "--data-dir is not a directory: {}", path.display())
            }
            Self::DataDirHasNoBinpacks(path) => {
                write!(
                    f,
                    "--data-dir contains no .binpack files: {}",
                    path.display()
                )
            }
            Self::MixedInputKinds => write!(
                f,
                "mixed .binpack and direct .data files are not supported in one run"
            ),
            Self::InvalidVirtualEpochs => {
                write!(f, "--chunk-virtual-epochs must be at least 1")
            }
            Self::InvalidChunkSample => write!(f, "--chunk-sample must be at least 1"),
            Self::ChunkSampleTooLarge { sample, chunks } => write!(
                f,
                "--chunk-sample ({sample}) cannot exceed unique chunk count ({chunks})"
            ),
            Self::Io { path, source } => {
                write!(f, "failed to read {}: {source}", path.display())
            }
        }
    }
}

impl Error for DataScheduleError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

pub fn build_data_schedule(
    options: DataScheduleOptions,
) -> Result<DataSchedule, DataScheduleError> {
    if options.chunk_virtual_epochs == 0 {
        return Err(DataScheduleError::InvalidVirtualEpochs);
    }
    if options.chunk_sample == Some(0) {
        return Err(DataScheduleError::InvalidChunkSample);
    }
    if options.explicit_paths.is_empty() && options.data_dirs.is_empty() {
        return Err(DataScheduleError::EmptyInputs);
    }

    let explicit_path_count = options.explicit_paths.len();
    let mut direct_paths = Vec::new();
    let mut binpack_paths = Vec::new();

    for path in &options.explicit_paths {
        if !path.exists() {
            return Err(DataScheduleError::MissingExplicitPath(path.clone()));
        }
        let path = canonicalize(path)?;
        if is_binpack_path(&path) {
            binpack_paths.push(path);
        } else {
            direct_paths.push(path);
        }
    }

    let mut discovered_chunk_count = 0;
    for data_dir in &options.data_dirs {
        if !data_dir.exists() {
            return Err(DataScheduleError::DataDirNotFound(data_dir.clone()));
        }
        if !data_dir.is_dir() {
            return Err(DataScheduleError::DataDirIsNotDirectory(data_dir.clone()));
        }

        let mut discovered = Vec::new();
        for entry in fs::read_dir(data_dir).map_err(|source| DataScheduleError::Io {
            path: data_dir.clone(),
            source,
        })? {
            let entry = entry.map_err(|source| DataScheduleError::Io {
                path: data_dir.clone(),
                source,
            })?;
            let path = entry.path();
            if is_binpack_path(&path) {
                discovered.push(canonicalize(&path)?);
            }
        }

        if discovered.is_empty() {
            return Err(DataScheduleError::DataDirHasNoBinpacks(data_dir.clone()));
        }

        discovered_chunk_count += discovered.len();
        binpack_paths.extend(discovered);
    }

    if !direct_paths.is_empty() && !binpack_paths.is_empty() {
        return Err(DataScheduleError::MixedInputKinds);
    }

    if binpack_paths.is_empty() {
        return Ok(DataSchedule {
            kind: InputDataKind::Direct,
            scheduled_paths: direct_paths,
            explicit_path_count,
            discovered_chunk_count,
            unique_chunk_count: 0,
            seed: options.chunk_shuffle_seed,
            virtual_epochs: options.chunk_virtual_epochs,
            chunk_sample: options.chunk_sample,
        });
    }

    let chunks = unique_sorted_paths(binpack_paths);
    let unique_chunk_count = chunks.len();
    if let Some(sample) = options.chunk_sample {
        if sample > unique_chunk_count {
            return Err(DataScheduleError::ChunkSampleTooLarge {
                sample,
                chunks: unique_chunk_count,
            });
        }
    }

    let per_epoch = options.chunk_sample.unwrap_or(unique_chunk_count);
    let mut rng = SplitMix64::new(options.chunk_shuffle_seed);
    let mut scheduled_paths = Vec::with_capacity(per_epoch * options.chunk_virtual_epochs);

    for _ in 0..options.chunk_virtual_epochs {
        let mut epoch = chunks.clone();
        fisher_yates_shuffle(&mut epoch, &mut rng);
        scheduled_paths.extend(epoch.into_iter().take(per_epoch));
    }

    Ok(DataSchedule {
        kind: InputDataKind::SfBinpack,
        scheduled_paths,
        explicit_path_count,
        discovered_chunk_count,
        unique_chunk_count,
        seed: options.chunk_shuffle_seed,
        virtual_epochs: options.chunk_virtual_epochs,
        chunk_sample: options.chunk_sample,
    })
}

pub fn print_startup_log(schedule: &DataSchedule) {
    println!("Data explicit paths: {}", schedule.explicit_path_count);
    println!(
        "Data discovered chunk files: {}",
        schedule.discovered_chunk_count
    );
    println!("Data unique chunks: {}", schedule.unique_chunk_count);
    println!(
        "Data scheduled path entries: {}",
        schedule.scheduled_paths.len()
    );
    println!("Chunk shuffle seed: {}", schedule.seed);
    println!("Chunk virtual epochs: {}", schedule.virtual_epochs);
    match schedule.chunk_sample {
        Some(sample) => println!("Chunk sample: {sample}"),
        None => println!("Chunk sample: all"),
    }
    println!("First scheduled chunks:");
    for (idx, path) in schedule.scheduled_paths.iter().take(8).enumerate() {
        println!("  {}: {}", idx + 1, path.display());
    }
    if schedule.scheduled_paths.is_empty() {
        println!("  none");
    }
}

pub fn input_data_kind_name(kind: InputDataKind) -> &'static str {
    match kind {
        InputDataKind::Direct => "direct bulletformat .data",
        InputDataKind::SfBinpack => "Stockfish .binpack",
    }
}

fn canonicalize(path: &Path) -> Result<PathBuf, DataScheduleError> {
    fs::canonicalize(path).map_err(|source| DataScheduleError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn is_binpack_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("binpack"))
}

fn unique_sorted_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut map = BTreeMap::new();
    for path in paths {
        map.entry(path_key(&path)).or_insert(path);
    }
    map.into_values().collect()
}

fn path_key(path: &Path) -> String {
    path.to_string_lossy()
        .replace('\\', "/")
        .to_ascii_lowercase()
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
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_ID: AtomicU64 = AtomicU64::new(0);

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(name: &str) -> Self {
            let id = TEST_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "mnue-data-schedule-{name}-{}-{id}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self { path }
        }

        fn file(&self, name: &str) -> PathBuf {
            let path = self.path.join(name);
            fs::write(&path, []).unwrap();
            path
        }

        fn dir(&self, name: &str) -> PathBuf {
            let path = self.path.join(name);
            fs::create_dir_all(&path).unwrap();
            path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn options() -> DataScheduleOptions {
        DataScheduleOptions {
            chunk_virtual_epochs: 1,
            ..Default::default()
        }
    }

    fn path_names(paths: &[PathBuf]) -> Vec<String> {
        paths
            .iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn data_dir_only_includes_binpack_files() {
        let temp = TempDir::new("includes-binpack");
        temp.file("a.binpack");
        temp.file("b.BINPACK");
        temp.file("ignore.data");
        temp.file("ignore.txt");

        let schedule = build_data_schedule(DataScheduleOptions {
            data_dirs: vec![temp.path.clone()],
            ..options()
        })
        .unwrap();

        let names = path_names(&schedule.scheduled_paths);
        assert_eq!(schedule.discovered_chunk_count, 2);
        assert!(
            names
                .iter()
                .all(|name| name.ends_with("binpack") || name.ends_with("BINPACK"))
        );
    }

    #[test]
    fn stable_sort_makes_schedule_independent_of_input_order() {
        let temp = TempDir::new("stable-sort");
        let a = temp.file("a.binpack");
        let b = temp.file("b.binpack");
        let c = temp.file("c.binpack");

        let first = build_data_schedule(DataScheduleOptions {
            explicit_paths: vec![c.clone(), a.clone(), b.clone()],
            chunk_shuffle_seed: 77,
            chunk_virtual_epochs: 5,
            ..Default::default()
        })
        .unwrap();
        let second = build_data_schedule(DataScheduleOptions {
            explicit_paths: vec![a, b, c],
            chunk_shuffle_seed: 77,
            chunk_virtual_epochs: 5,
            ..Default::default()
        })
        .unwrap();

        assert_eq!(first.scheduled_paths, second.scheduled_paths);
    }

    #[test]
    fn same_seed_gives_identical_schedule() {
        let temp = TempDir::new("same-seed");
        let files = (0..5)
            .map(|idx| temp.file(&format!("{idx}.binpack")))
            .collect::<Vec<_>>();

        let a = build_data_schedule(DataScheduleOptions {
            explicit_paths: files.clone(),
            chunk_shuffle_seed: 123,
            chunk_virtual_epochs: 8,
            ..Default::default()
        })
        .unwrap();
        let b = build_data_schedule(DataScheduleOptions {
            explicit_paths: files,
            chunk_shuffle_seed: 123,
            chunk_virtual_epochs: 8,
            ..Default::default()
        })
        .unwrap();

        assert_eq!(a.scheduled_paths, b.scheduled_paths);
    }

    #[test]
    fn different_seed_gives_different_schedule() {
        let temp = TempDir::new("different-seed");
        let files = (0..7)
            .map(|idx| temp.file(&format!("{idx}.binpack")))
            .collect::<Vec<_>>();

        let a = build_data_schedule(DataScheduleOptions {
            explicit_paths: files.clone(),
            chunk_shuffle_seed: 1,
            chunk_virtual_epochs: 4,
            ..Default::default()
        })
        .unwrap();
        let b = build_data_schedule(DataScheduleOptions {
            explicit_paths: files,
            chunk_shuffle_seed: 2,
            chunk_virtual_epochs: 4,
            ..Default::default()
        })
        .unwrap();

        assert_ne!(a.scheduled_paths, b.scheduled_paths);
    }

    #[test]
    fn full_traversal_includes_every_chunk_once_per_virtual_epoch() {
        let temp = TempDir::new("full-traversal");
        let files = (0..4)
            .map(|idx| temp.file(&format!("{idx}.binpack")))
            .collect::<Vec<_>>();

        let schedule = build_data_schedule(DataScheduleOptions {
            explicit_paths: files.clone(),
            chunk_shuffle_seed: 9,
            chunk_virtual_epochs: 6,
            ..Default::default()
        })
        .unwrap();

        let expected = unique_sorted_paths(
            files
                .iter()
                .map(|path| fs::canonicalize(path).unwrap())
                .collect(),
        )
        .into_iter()
        .map(|path| path_key(&path))
        .collect::<Vec<_>>();
        for epoch in schedule.scheduled_paths.chunks(expected.len()) {
            let mut got = epoch.iter().map(|path| path_key(path)).collect::<Vec<_>>();
            got.sort();
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn chunk_sample_includes_n_distinct_chunks_per_virtual_epoch() {
        let temp = TempDir::new("sample");
        let files = (0..5)
            .map(|idx| temp.file(&format!("{idx}.binpack")))
            .collect::<Vec<_>>();

        let schedule = build_data_schedule(DataScheduleOptions {
            explicit_paths: files,
            chunk_shuffle_seed: 5,
            chunk_virtual_epochs: 7,
            chunk_sample: Some(2),
            ..Default::default()
        })
        .unwrap();

        assert_eq!(schedule.scheduled_paths.len(), 14);
        for epoch in schedule.scheduled_paths.chunks(2) {
            assert_eq!(epoch.len(), 2);
            assert_ne!(path_key(&epoch[0]), path_key(&epoch[1]));
        }
    }

    #[test]
    fn chunk_sample_greater_than_chunk_count_errors() {
        let temp = TempDir::new("sample-too-large");
        temp.file("a.binpack");

        let error = build_data_schedule(DataScheduleOptions {
            data_dirs: vec![temp.path.clone()],
            chunk_sample: Some(2),
            ..options()
        })
        .unwrap_err();

        assert!(error.to_string().contains("--chunk-sample"));
    }

    #[test]
    fn empty_directory_errors() {
        let temp = TempDir::new("empty-dir");
        let empty = temp.dir("empty");

        let error = build_data_schedule(DataScheduleOptions {
            data_dirs: vec![empty],
            ..options()
        })
        .unwrap_err();

        assert!(error.to_string().contains("contains no .binpack"));
    }

    #[test]
    fn missing_explicit_file_errors() {
        let temp = TempDir::new("missing-explicit");
        let missing = temp.path.join("missing.data");

        let error = build_data_schedule(DataScheduleOptions {
            explicit_paths: vec![missing],
            ..options()
        })
        .unwrap_err();

        assert!(error.to_string().contains("not found"));
    }

    #[test]
    fn mixed_data_and_binpack_inputs_error() {
        let temp = TempDir::new("mixed");
        let data = temp.file("a.data");
        let binpack = temp.file("b.binpack");

        let error = build_data_schedule(DataScheduleOptions {
            explicit_paths: vec![data, binpack],
            ..options()
        })
        .unwrap_err();

        assert!(error.to_string().contains("mixed .binpack"));
    }
}
