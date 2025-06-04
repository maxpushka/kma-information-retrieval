use bit_vec::BitVec;
use clap::{Arg, Command, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use quick_xml::events::Event;
use quick_xml::Reader;
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
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

#[derive(Debug, Serialize, Deserialize)]
pub struct IncidenceMatrix {
    pub terms: Vec<String>,
    pub documents: Vec<String>,
    pub matrix: Vec<BitVec>,
}

impl IncidenceMatrix {
    pub fn from_dictionary(dictionary: &Dictionary) -> Self {
        let mut terms: Vec<String> = dictionary.terms.keys().cloned().collect();
        terms.sort();
        
        let mut documents: HashSet<String> = HashSet::new();
        for term_entry in dictionary.terms.values() {
            for doc in &term_entry.documents {
                documents.insert(doc.clone());
            }
        }
        let mut documents: Vec<String> = documents.into_iter().collect();
        documents.sort();
        
        let mut matrix = Vec::new();
        for term in &terms {
            let mut row = BitVec::from_elem(documents.len(), false);
            if let Some(term_entry) = dictionary.terms.get(term) {
                for doc in &term_entry.documents {
                    if let Some(doc_idx) = documents.iter().position(|d| d == doc) {
                        row.set(doc_idx, true);
                    }
                }
            }
            matrix.push(row);
        }
        
        IncidenceMatrix {
            terms,
            documents,
            matrix,
        }
    }
    
    pub fn memory_size(&self) -> usize {
        std::mem::size_of::<Self>() +
        self.terms.iter().map(|t| t.len()).sum::<usize>() +
        self.documents.iter().map(|d| d.len()).sum::<usize>() +
        self.matrix.iter().map(|row| row.capacity() / 8).sum::<usize>()
    }
    
    pub fn search(&self, query: &str) -> Result<BitVec, String> {
        let query = query.to_lowercase();
        
        if query.contains(" and ") {
            let parts: Vec<&str> = query.split(" and ").collect();
            let mut result = self.search_term(parts[0].trim())?;
            for part in &parts[1..] {
                let term_result = self.search_term(part.trim())?;
                result.and(&term_result);
            }
            Ok(result)
        } else if query.contains(" or ") {
            let parts: Vec<&str> = query.split(" or ").collect();
            let mut result = self.search_term(parts[0].trim())?;
            for part in &parts[1..] {
                let term_result = self.search_term(part.trim())?;
                result.or(&term_result);
            }
            Ok(result)
        } else if query.starts_with("not ") {
            let term = query.strip_prefix("not ").unwrap().trim();
            let mut result = self.search_term(term)?;
            result.negate();
            Ok(result)
        } else {
            self.search_term(&query)
        }
    }
    
    fn search_term(&self, term: &str) -> Result<BitVec, String> {
        if let Some(term_idx) = self.terms.iter().position(|t| t == term) {
            Ok(self.matrix[term_idx].clone())
        } else {
            Err(format!("Term '{}' not found", term))
        }
    }
    
    pub fn get_matching_documents(&self, result: &BitVec) -> Vec<&String> {
        result.iter()
            .enumerate()
            .filter_map(|(idx, bit)| if bit { Some(&self.documents[idx]) } else { None })
            .collect()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InvertedIndex {
    pub index: HashMap<String, Vec<String>>,
    pub documents: Vec<String>,
}

impl InvertedIndex {
    pub fn from_dictionary(dictionary: &Dictionary) -> Self {
        let mut index = HashMap::new();
        let mut documents = HashSet::new();
        
        for (term, term_entry) in &dictionary.terms {
            index.insert(term.clone(), term_entry.documents.clone());
            for doc in &term_entry.documents {
                documents.insert(doc.clone());
            }
        }
        
        let mut documents: Vec<String> = documents.into_iter().collect();
        documents.sort();
        
        InvertedIndex { index, documents }
    }
    
    pub fn memory_size(&self) -> usize {
        std::mem::size_of::<Self>() +
        self.index.iter().map(|(k, v)| {
            k.len() + std::mem::size_of::<Vec<String>>() + 
            v.iter().map(|s| s.len()).sum::<usize>()
        }).sum::<usize>() +
        self.documents.iter().map(|d| d.len()).sum::<usize>()
    }
    
    pub fn search(&self, query: &str) -> Result<HashSet<String>, String> {
        let query = query.to_lowercase();
        
        if query.contains(" and ") {
            let parts: Vec<&str> = query.split(" and ").collect();
            let mut result = self.search_term(parts[0].trim())?;
            for part in &parts[1..] {
                let term_result = self.search_term(part.trim())?;
                result = result.intersection(&term_result).cloned().collect();
            }
            Ok(result)
        } else if query.contains(" or ") {
            let parts: Vec<&str> = query.split(" or ").collect();
            let mut result = self.search_term(parts[0].trim())?;
            for part in &parts[1..] {
                let term_result = self.search_term(part.trim())?;
                result = result.union(&term_result).cloned().collect();
            }
            Ok(result)
        } else if query.starts_with("not ") {
            let term = query.strip_prefix("not ").unwrap().trim();
            let term_docs = self.search_term(term)?;
            let all_docs: HashSet<String> = self.documents.iter().cloned().collect();
            Ok(all_docs.difference(&term_docs).cloned().collect())
        } else {
            self.search_term(&query)
        }
    }
    
    fn search_term(&self, term: &str) -> Result<HashSet<String>, String> {
        if let Some(docs) = self.index.get(term) {
            Ok(docs.iter().cloned().collect())
        } else {
            Err(format!("Term '{}' not found", term))
        }
    }
}

#[derive(Subcommand)]
enum Commands {
    Build {
        #[arg(short, long)]
        input: String,
        #[arg(short, long, default_value = "dictionary")]
        output: String,
        #[arg(short, long, default_value = "binary,json,text")]
        formats: String,
    },
    Search {
        #[arg(short, long)]
        query: String,
        #[arg(short, long, default_value = "dictionary")]
        dict_file: String,
    },
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
    let cli = Command::new("FB2 Information Retrieval")
        .version("1.0")
        .about("FB2 text processing and Boolean search")
        .subcommand_required(true)
        .subcommand(
            Command::new("build")
                .about("Build dictionary and search structures from FB2 files")
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
                ),
        )
        .subcommand(
            Command::new("search")
                .about("Search using Boolean queries")
                .arg(
                    Arg::new("query")
                        .short('q')
                        .long("query")
                        .value_name("QUERY")
                        .help("Boolean query (e.g., 'term1 and term2', 'term1 or term2', 'not term1')")
                        .required(true),
                )
                .arg(
                    Arg::new("dict_file")
                        .short('d')
                        .long("dict")
                        .value_name("FILE")
                        .help("Dictionary file prefix")
                        .default_value("dictionary"),
                ),
        );

    let matches = cli.get_matches();

    match matches.subcommand() {
        Some(("build", sub_matches)) => {
            let input_dir = sub_matches.get_one::<String>("input").unwrap();
            let output_prefix = sub_matches.get_one::<String>("output").unwrap();
            let formats: Vec<&str> = sub_matches
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

            println!("\n=== BUILDING SEARCH STRUCTURES ===");
            
            let incidence_start = Instant::now();
            let incidence_matrix = IncidenceMatrix::from_dictionary(&dictionary);
            let incidence_time = incidence_start.elapsed();
            let incidence_size = incidence_matrix.memory_size();
            
            let inverted_start = Instant::now();
            let inverted_index = InvertedIndex::from_dictionary(&dictionary);
            let inverted_time = inverted_start.elapsed();
            let inverted_size = inverted_index.memory_size();

            println!("Incidence Matrix: {} bytes, built in {:.2?}", incidence_size, incidence_time);
            println!("Inverted Index: {} bytes, built in {:.2?}", inverted_size, inverted_time);
            
            let matrix_path = format!("{}_matrix.bin", output_prefix);
            let index_path = format!("{}_index.bin", output_prefix);
            
            let matrix_data = bincode::serialize(&incidence_matrix)?;
            fs::write(&matrix_path, matrix_data)?;
            
            let index_data = bincode::serialize(&inverted_index)?;
            fs::write(&index_path, index_data)?;
            
            println!("Saved incidence matrix to: {}", matrix_path);
            println!("Saved inverted index to: {}", index_path);

            println!("\n=== STRUCTURE COMPARISON ===");
            println!("Dictionary (HashMap):  {} bytes", 
                     std::mem::size_of_val(&dictionary) + 
                     dictionary.terms.iter().map(|(k, v)| k.len() + std::mem::size_of_val(v)).sum::<usize>());
            println!("Incidence Matrix:      {} bytes", incidence_size);
            println!("Inverted Index:        {} bytes", inverted_size);
            
            let efficiency_ratio = inverted_size as f64 / incidence_size as f64;
            println!("Space efficiency (Index/Matrix): {:.2}x", efficiency_ratio);

            println!("\n=== DATA STRUCTURE ANALYSIS ===");
            println!("Incidence Matrix:");
            println!("  - Space: O(|T| × |D|) where T=terms, D=documents");
            println!("  - Search: O(|T|) for term lookup + O(|D|) for Boolean operations");
            println!("  - Memory: {} bits per term-document pair", 
                     incidence_matrix.matrix.len() * incidence_matrix.documents.len());
            
            println!("Inverted Index:");
            println!("  - Space: O(|unique_postings|) - only stores actual occurrences");
            println!("  - Search: O(1) for term lookup + O(|posting_lists|) for Boolean operations");
            println!("  - Memory: Variable size based on term distribution");

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
        }
        Some(("search", sub_matches)) => {
            let query = sub_matches.get_one::<String>("query").unwrap();
            let dict_prefix = sub_matches.get_one::<String>("dict_file").unwrap();
            
            let matrix_path = format!("{}_matrix.bin", dict_prefix);
            let index_path = format!("{}_index.bin", dict_prefix);
            
            println!("Loading search structures...");
            
            let matrix_data = fs::read(&matrix_path)?;
            let incidence_matrix: IncidenceMatrix = bincode::deserialize(&matrix_data)?;
            
            let index_data = fs::read(&index_path)?;
            let inverted_index: InvertedIndex = bincode::deserialize(&index_data)?;
            
            println!("Query: {}", query);
            println!("\n=== INCIDENCE MATRIX SEARCH ===");
            let matrix_start = Instant::now();
            match incidence_matrix.search(query) {
                Ok(result) => {
                    let matrix_time = matrix_start.elapsed();
                    let matching_docs = incidence_matrix.get_matching_documents(&result);
                    println!("Found {} documents in {:.2?}", matching_docs.len(), matrix_time);
                    for doc in &matching_docs {
                        println!("  - {}", doc);
                    }
                }
                Err(e) => println!("Error: {}", e),
            }
            
            println!("\n=== INVERTED INDEX SEARCH ===");
            let index_start = Instant::now();
            match inverted_index.search(query) {
                Ok(result) => {
                    let index_time = index_start.elapsed();
                    let mut docs: Vec<_> = result.iter().collect();
                    docs.sort();
                    println!("Found {} documents in {:.2?}", docs.len(), index_time);
                    for doc in &docs {
                        println!("  - {}", doc);
                    }
                }
                Err(e) => println!("Error: {}", e),
            }
        }
        _ => unreachable!(),
    }

    Ok(())
}
