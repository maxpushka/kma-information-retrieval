use clap::{Arg, Command};
use grimoire::{build_dictionary, collect_fb2_files, IncidenceMatrix, InvertedIndex, BigramIndex, CoordinateIndex, QueryParser, FB2Parser};
use std::fs;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Command::new("Grimoire")
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
            handle_build_command(sub_matches)?;
        }
        Some(("search", sub_matches)) => {
            handle_search_command(sub_matches)?;
        }
        _ => unreachable!(),
    }

    Ok(())
}

fn handle_build_command(matches: &clap::ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
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

    println!("\n=== BUILDING SEARCH STRUCTURES ===");
    
    let incidence_start = Instant::now();
    let incidence_matrix = IncidenceMatrix::from_dictionary(&dictionary);
    let incidence_time = incidence_start.elapsed();
    let incidence_size = incidence_matrix.memory_size();
    
    let inverted_start = Instant::now();
    let inverted_index = InvertedIndex::from_dictionary(&dictionary);
    let inverted_time = inverted_start.elapsed();
    let inverted_size = inverted_index.memory_size();

    println!("Building bigram index...");
    let bigram_start = Instant::now();
    let parser = FB2Parser::new();
    let bigram_index = BigramIndex::from_dictionary_with_parser(&dictionary, |doc_name| {
        let file_path = std::path::Path::new(input_dir).join(doc_name);
        parser.parse_file(&file_path)
    })?;
    let bigram_time = bigram_start.elapsed();
    let bigram_size = bigram_index.memory_size();

    println!("Building coordinate index...");
    let coordinate_start = Instant::now();
    let coordinate_index = CoordinateIndex::from_dictionary_with_parser(&dictionary, |doc_name| {
        let file_path = std::path::Path::new(input_dir).join(doc_name);
        parser.parse_file(&file_path)
    })?;
    let coordinate_time = coordinate_start.elapsed();
    let coordinate_size = coordinate_index.memory_size();

    println!("Incidence Matrix: {} bytes, built in {:.2?}", incidence_size, incidence_time);
    println!("Inverted Index: {} bytes, built in {:.2?}", inverted_size, inverted_time);
    println!("Bigram Index: {} bytes, built in {:.2?}", bigram_size, bigram_time);
    println!("Coordinate Index: {} bytes, built in {:.2?}", coordinate_size, coordinate_time);
    
    let matrix_path = format!("{}_matrix.bin", output_prefix);
    let index_path = format!("{}_index.bin", output_prefix);
    let bigram_path = format!("{}_bigram.bin", output_prefix);
    let coordinate_path = format!("{}_coordinate.bin", output_prefix);
    
    let matrix_data = bincode::serialize(&incidence_matrix)?;
    fs::write(&matrix_path, matrix_data)?;
    
    let index_data = bincode::serialize(&inverted_index)?;
    fs::write(&index_path, index_data)?;
    
    let bigram_data = bincode::serialize(&bigram_index)?;
    fs::write(&bigram_path, bigram_data)?;
    
    let coordinate_data = bincode::serialize(&coordinate_index)?;
    fs::write(&coordinate_path, coordinate_data)?;
    
    println!("Saved incidence matrix to: {}", matrix_path);
    println!("Saved inverted index to: {}", index_path);
    println!("Saved bigram index to: {}", bigram_path);
    println!("Saved coordinate index to: {}", coordinate_path);

    println!("\n=== STRUCTURE COMPARISON ===");
    println!("Dictionary (HashMap):  {} bytes", 
             std::mem::size_of_val(&dictionary) + 
             dictionary.terms.iter().map(|(k, v)| k.len() + std::mem::size_of_val(v)).sum::<usize>());
    println!("Incidence Matrix:      {} bytes", incidence_size);
    println!("Inverted Index:        {} bytes", inverted_size);
    println!("Bigram Index:          {} bytes", bigram_size);
    println!("Coordinate Index:      {} bytes", coordinate_size);
    
    let efficiency_ratio = inverted_size as f64 / incidence_size as f64;
    println!("Space efficiency (Index/Matrix): {:.2}x", efficiency_ratio);
    let coordinate_ratio = coordinate_size as f64 / inverted_size as f64;
    println!("Space overhead (Coordinate/Inverted): {:.2}x", coordinate_ratio);

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
    
    println!("Bigram Index:");
    println!("  - Space: O(|unique_bigrams|) - stores two-word combinations");
    println!("  - Search: Optimized for phrase search with exact word order");
    println!("  - Memory: {} bigrams indexed", bigram_index.index.len());
    
    println!("Coordinate Index:");
    println!("  - Space: O(|postings| × |positions|) - stores position information");
    println!("  - Search: Supports phrase and proximity search with position verification");
    println!("  - Memory: Includes position data for each term occurrence");

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

    Ok(())
}

fn handle_search_command(matches: &clap::ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
    let query = matches.get_one::<String>("query").unwrap();
    let dict_prefix = matches.get_one::<String>("dict_file").unwrap();
    
    let matrix_path = format!("{}_matrix.bin", dict_prefix);
    let index_path = format!("{}_index.bin", dict_prefix);
    let bigram_path = format!("{}_bigram.bin", dict_prefix);
    let coordinate_path = format!("{}_coordinate.bin", dict_prefix);
    
    println!("Loading search structures...");
    
    let matrix_data = fs::read(&matrix_path)?;
    let incidence_matrix: IncidenceMatrix = bincode::deserialize(&matrix_data)?;
    
    let index_data = fs::read(&index_path)?;
    let inverted_index: InvertedIndex = bincode::deserialize(&index_data)?;
    
    let bigram_data = fs::read(&bigram_path)?;
    let bigram_index: BigramIndex = bincode::deserialize(&bigram_data)?;
    
    let coordinate_data = fs::read(&coordinate_path)?;
    let coordinate_index: CoordinateIndex = bincode::deserialize(&coordinate_data)?;
    
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

    if query.contains('"') {
        println!("\n=== BIGRAM INDEX PHRASE SEARCH ===");
        let bigram_start = Instant::now();
        match bigram_index.search(query) {
            Ok(result) => {
                let bigram_time = bigram_start.elapsed();
                let mut docs: Vec<_> = result.iter().collect();
                docs.sort();
                println!("Found {} documents in {:.2?}", docs.len(), bigram_time);
                for doc in &docs {
                    println!("  - {}", doc);
                }
            }
            Err(e) => println!("Error: {}", e),
        }
    }

    println!("\n=== COORDINATE INDEX SEARCH ===");
    let coordinate_start = Instant::now();
    match coordinate_index.search(query) {
        Ok(result) => {
            let coordinate_time = coordinate_start.elapsed();
            let mut docs: Vec<_> = result.iter().collect();
            docs.sort();
            println!("Found {} documents in {:.2?}", docs.len(), coordinate_time);
            for doc in &docs {
                println!("  - {}", doc);
            }
        }
        Err(e) => println!("Error: {}", e),
    }

    if query.contains("near/") {
        println!("\n=== PROXIMITY SEARCH EXAMPLE ===");
        println!("Example proximity searches:");
        println!("  near/5(word1 word2) - finds 'word1' and 'word2' within 5 positions");
        println!("  near/10(love peace) - finds 'love' and 'peace' within 10 positions");
    }

    Ok(())
}