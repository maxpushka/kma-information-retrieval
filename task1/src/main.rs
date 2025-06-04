use clap::{Arg, Command};
use indicatif::{ProgressBar, ProgressStyle};
use quick_xml::events::Event;
use quick_xml::Reader;
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use walkdir::WalkDir;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermEntry {
    pub frequency: u32,
    pub documents: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Dictionary {
    pub terms: HashMap<String, TermEntry>,
    pub total_words: u64,
    pub total_documents: u32,
    pub collection_size_bytes: u64,
}

impl Dictionary {
    pub fn new() -> Self {
        Dictionary {
            terms: HashMap::new(),
            total_words: 0,
            total_documents: 0,
            collection_size_bytes: 0,
        }
    }

    pub fn add_term(&mut self, term: String, document: String) {
        let entry = self.terms.entry(term).or_insert(TermEntry {
            frequency: 0,
            documents: Vec::new(),
        });
        entry.frequency += 1;
        if !entry.documents.contains(&document) {
            entry.documents.push(document);
        }
        self.total_words += 1;
    }

    pub fn merge_terms(&mut self, terms: Vec<(String, String)>) {
        for (term, document) in terms {
            self.add_term(term, document);
        }
    }

    pub fn add_file_stats(&mut self, file_size: u64) {
        self.collection_size_bytes += file_size;
        self.total_documents += 1;
    }

    pub fn dictionary_size(&self) -> usize {
        self.terms.len()
    }

    pub fn save_as_binary(&self, path: &str) -> Result<usize, Box<dyn std::error::Error>> {
        let data = bincode::serialize(self)?;
        let size = data.len();
        fs::write(path, data)?;
        Ok(size)
    }

    pub fn save_as_json(&self, path: &str) -> Result<usize, Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        let size = json.len();
        fs::write(path, json)?;
        Ok(size)
    }

    pub fn save_as_text(&self, path: &str) -> Result<usize, Box<dyn std::error::Error>> {
        let mut file = File::create(path)?;
        let header = format!(
            "DICTIONARY STATISTICS\n\
             Total terms: {}\n\
             Total words: {}\n\
             Total documents: {}\n\
             Collection size: {} bytes\n\n\
             TERMS (sorted by frequency):\n",
            self.dictionary_size(),
            self.total_words,
            self.total_documents,
            self.collection_size_bytes
        );
        file.write_all(header.as_bytes())?;

        let mut sorted_terms: Vec<_> = self.terms.iter().collect();
        sorted_terms.sort_by(|a, b| b.1.frequency.cmp(&a.1.frequency));

        for (term, entry) in sorted_terms {
            let line = format!("{}: {} (docs: {})\n", term, entry.frequency, entry.documents.len());
            file.write_all(line.as_bytes())?;
        }

        Ok(fs::metadata(path)?.len() as usize)
    }
}

pub struct FB2Parser {
    word_regex: Regex,
}

impl FB2Parser {
    pub fn new() -> Self {
        FB2Parser {
            word_regex: Regex::new(r"\b[а-яёА-ЯЁa-zA-Z]{3,}\b").unwrap(),
        }
    }

    pub fn parse_file(&self, path: &Path) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut xml_reader = Reader::from_reader(reader);
        xml_reader.trim_text(true);

        let mut words = Vec::new();
        let mut buf = Vec::new();
        let mut in_body = false;

        loop {
            match xml_reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    if e.name().as_ref() == b"body" {
                        in_body = true;
                    }
                }
                Ok(Event::End(ref e)) => {
                    if e.name().as_ref() == b"body" {
                        in_body = false;
                    }
                }
                Ok(Event::Text(e)) => {
                    if in_body {
                        let text = e.unescape()?;
                        for word_match in self.word_regex.find_iter(&text) {
                            let word = word_match.as_str().to_lowercase();
                            if word.len() >= 3 {
                                words.push(word);
                            }
                        }
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => {
                    eprintln!("Error parsing {}: {}", path.display(), e);
                    break;
                }
                _ => {}
            }
            buf.clear();
        }

        Ok(words)
    }
}

pub fn collect_fb2_files(directory: &str) -> Vec<std::path::PathBuf> {
    WalkDir::new(directory)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            e.path()
                .extension()
                .map_or(false, |ext| ext == "fb2")
        })
        .map(|e| e.path().to_path_buf())
        .collect()
}

pub fn build_dictionary(
    files: &[std::path::PathBuf],
    show_progress: bool,
) -> Result<Dictionary, Box<dyn std::error::Error>> {
    let mut dictionary = Dictionary::new();
    
    let pb = if show_progress {
        let pb = ProgressBar::new(files.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
                .unwrap(),
        );
        Arc::new(Mutex::new(Some(pb)))
    } else {
        Arc::new(Mutex::new(None))
    };

    let pb_clone = Arc::clone(&pb);
    
    // Process files in parallel and collect results
    let results: Vec<_> = files
        .par_iter()
        .map(|file_path| {
            let parser = FB2Parser::new();
            
            // Update progress bar
            if let Ok(pb_lock) = pb_clone.lock() {
                if let Some(ref pb) = *pb_lock {
                    pb.set_message(format!("Processing {}", file_path.display()));
                }
            }

            let file_size = match fs::metadata(file_path) {
                Ok(metadata) => metadata.len(),
                Err(e) => {
                    eprintln!("Error reading metadata for {}: {}", file_path.display(), e);
                    return None;
                }
            };

            if file_size < 150_000 {
                eprintln!(
                    "Warning: {} is smaller than 150KB ({} bytes)",
                    file_path.display(),
                    file_size
                );
                return None;
            }

            let document_name = file_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();

            let words = match parser.parse_file(file_path) {
                Ok(words) => words,
                Err(e) => {
                    eprintln!("Error processing {}: {}", file_path.display(), e);
                    return None;
                }
            };

            // Update progress bar
            if let Ok(pb_lock) = pb_clone.lock() {
                if let Some(ref pb) = *pb_lock {
                    pb.inc(1);
                }
            }

            Some((file_size, document_name, words))
        })
        .collect();

    // Merge results into dictionary sequentially
    for result in results {
        if let Some((file_size, document_name, words)) = result {
            dictionary.add_file_stats(file_size);
            
            let terms: Vec<(String, String)> = words
                .into_iter()
                .map(|word| (word, document_name.clone()))
                .collect();
            
            dictionary.merge_terms(terms);
        }
    }

    if let Ok(pb_lock) = pb.lock() {
        if let Some(ref pb) = *pb_lock {
            pb.finish_with_message("Dictionary building completed");
        }
    }

    Ok(dictionary)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("FB2 Dictionary Builder")
        .version("1.0")
        .about("Builds a term dictionary from FB2 text files")
        .arg(
            Arg::new("input")
                .short('i')
                .long("input")
                .value_name("DIRECTORY")
                .help("Input directory containing FB2 files")
                .required(true),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("PREFIX")
                .help("Output file prefix")
                .default_value("dictionary"),
        )
        .arg(
            Arg::new("formats")
                .short('f')
                .long("formats")
                .value_name("FORMATS")
                .help("Serialization formats (binary,json,text)")
                .default_value("binary,json,text"),
        )
        .get_matches();

    let input_dir = matches.get_one::<String>("input").unwrap();
    let output_prefix = matches.get_one::<String>("output").unwrap();
    let formats: Vec<&str> = matches
        .get_one::<String>("formats")
        .unwrap()
        .split(',')
        .collect();

    println!("Collecting FB2 files from: {}", input_dir);
    let files = collect_fb2_files(input_dir);

    if files.is_empty() {
        eprintln!("No FB2 files found in {}", input_dir);
        return Ok(());
    }

    if files.len() < 10 {
        eprintln!(
            "Warning: Found only {} FB2 files (requirement: at least 10)",
            files.len()
        );
    }

    println!("Found {} FB2 files", files.len());
    for file in &files[..std::cmp::min(5, files.len())] {
        println!("  - {}", file.display());
    }
    if files.len() > 5 {
        println!("  ... and {} more", files.len() - 5);
    }

    println!("\nBuilding dictionary...");
    let start_time = Instant::now();
    let dictionary = build_dictionary(&files, true)?;
    let build_time = start_time.elapsed();

    println!("\n=== COLLECTION STATISTICS ===");
    println!("Collection size: {} bytes ({:.2} MB)", 
             dictionary.collection_size_bytes, 
             dictionary.collection_size_bytes as f64 / 1_048_576.0);
    println!("Total documents: {}", dictionary.total_documents);
    println!("Total words: {}", dictionary.total_words);
    println!("Dictionary size: {} unique terms", dictionary.dictionary_size());
    println!("Build time: {:.2?}", build_time);

    println!("\n=== COMPLEXITY ANALYSIS ===");
    println!("Time complexity: O(n*m/p) where n = total words, m = avg word length, p = CPU cores");
    println!("Space complexity: O(k) where k = unique terms");
    println!("Parallel processing: {} logical CPU cores", rayon::current_num_threads());
    println!("Dictionary compression ratio: {:.2}%", 
             (dictionary.dictionary_size() as f64 / dictionary.total_words as f64) * 100.0);

    println!("\n=== SAVING DICTIONARY ===");
    let mut format_sizes = Vec::new();

    for format in &formats {
        let start_time = Instant::now();
        let (file_path, size) = match format.trim() {
            "binary" => {
                let path = format!("{}.bin", output_prefix);
                let size = dictionary.save_as_binary(&path)?;
                (path, size)
            }
            "json" => {
                let path = format!("{}.json", output_prefix);
                let size = dictionary.save_as_json(&path)?;
                (path, size)
            }
            "text" => {
                let path = format!("{}.txt", output_prefix);
                let size = dictionary.save_as_text(&path)?;
                (path, size)
            }
            _ => {
                eprintln!("Unknown format: {}", format);
                continue;
            }
        };
        let save_time = start_time.elapsed();
        format_sizes.push((format.to_string(), size, save_time));
        println!("Saved {} format: {} ({} bytes, {:.2?})", format, file_path, size, save_time);
    }

    println!("\n=== FORMAT COMPARISON ===");
    format_sizes.sort_by_key(|(_, size, _)| *size);
    for (format, size, time) in &format_sizes {
        println!("{:>6}: {:>10} bytes ({:>6.2?})", format, size, time);
    }

    if let Some((smallest_format, smallest_size, _)) = format_sizes.first() {
        if let Some((largest_format, largest_size, _)) = format_sizes.last() {
            let ratio = *largest_size as f64 / *smallest_size as f64;
            println!("Size ratio ({}:{}): {:.2}x", largest_format, smallest_format, ratio);
        }
    }

    println!("\n=== ALGORITHM JUSTIFICATION ===");
    println!("Data Structure: HashMap<String, TermEntry>");
    println!("  - O(1) average lookup/insertion for terms");
    println!("  - Efficient memory usage with hash-based storage");
    println!("  - Supports frequency counting and document tracking");
    println!("Parsing: Parallel SAX-like XML parsing with regex text extraction");
    println!("  - Memory efficient streaming for large files");
    println!("  - Selective parsing (body content only)");
    println!("  - Parallel processing of files using Rayon");
    println!("  - Thread-safe progress reporting with Arc<Mutex<>>"); 
    println!("Serialization formats tested for space/time trade-offs");

    Ok(())
}
