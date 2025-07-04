use clap::{Arg, Command};
use grimoire::{
    build_dictionary, collect_fb2_files, BigramIndex, CoordinateIndex, FB2Parser, IncidenceMatrix,
    CompressedInvertedIndex, ParallelSPIMIIndexer, ParquetLoader, QueryParser, WildcardSearchEngine,
};
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
        )
        .subcommand(
            Command::new("parquet-inspect")
                .about("Inspect Parquet file schema and sample data")
                .arg(
                    Arg::new("input")
                        .short('i')
                        .long("input")
                        .value_name("FILE")
                        .help("Parquet file to inspect")
                        .required(true),
                ),
        )
        .subcommand(
            Command::new("parquet-build")
                .about("Build dictionary and search structures from Parquet file")
                .arg(
                    Arg::new("input")
                        .short('i')
                        .long("input")
                        .value_name("FILE")
                        .help("Parquet file to process")
                        .required(true),
                )
                .arg(
                    Arg::new("output")
                        .short('o')
                        .long("output")
                        .value_name("PREFIX")
                        .help("Output file prefix")
                        .default_value("parquet_dictionary"),
                )
                .arg(
                    Arg::new("spimi")
                        .long("spimi")
                        .help("Use SPIMI indexing for large datasets")
                        .action(clap::ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("memory-limit")
                        .long("memory-limit")
                        .value_name("MB")
                        .help("Memory limit for SPIMI indexing in MB")
                        .default_value("512"),
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
        Some(("parquet-inspect", sub_matches)) => {
            handle_parquet_inspect_command(sub_matches)?;
        }
        Some(("parquet-build", sub_matches)) => {
            handle_parquet_build_command(sub_matches)?;
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
    let regular_dictionary = build_dictionary(&files, true)?;
    let build_time = start_time.elapsed();

    println!("Compressing dictionary...");
    let compress_start = Instant::now();
    let dictionary = grimoire::CompressedDictionary::from_dictionary(&regular_dictionary);
    let compress_time = compress_start.elapsed();
    println!("Dictionary compression completed in {:.2?}", compress_time);

    println!("\n=== COLLECTION STATISTICS ===");
    println!(
        "Collection size: {} bytes ({:.2} MB)",
        dictionary.collection_size_bytes,
        dictionary.collection_size_bytes as f64 / 1_048_576.0
    );
    println!("Total documents: {}", dictionary.total_documents);
    println!("Total words: {}", dictionary.total_words);
    println!(
        "Dictionary size: {} unique terms",
        dictionary.dictionary_size()
    );
    println!("Build time: {:.2?}", build_time);

    println!("\n=== BUILDING SEARCH STRUCTURES ===");

    let incidence_start = Instant::now();
    let incidence_matrix = IncidenceMatrix::from_dictionary(&dictionary);
    let incidence_time = incidence_start.elapsed();
    let incidence_size = incidence_matrix.memory_size();

    let inverted_start = Instant::now();
    let inverted_index = CompressedInvertedIndex::from_compressed_dictionary(&dictionary);
    let inverted_time = inverted_start.elapsed();
    let inverted_size = inverted_index.memory_size();

    println!("Building bigram index...");
    println!("  Dictionary has {} unique terms", dictionary.sorted_terms.len());
    let bigram_start = Instant::now();
    let parser = FB2Parser::new();
    let bigram_index = BigramIndex::from_dictionary_with_parser(&dictionary, |doc_name| {
        println!("  Processing document for bigram index: {}", doc_name);
        let file_path = std::path::Path::new(input_dir).join(doc_name);
        let result = parser.parse_file(&file_path);
        if let Ok(ref words) = result {
            println!("    Parsed {} words from {}", words.len(), doc_name);
        } else {
            println!("    Failed to parse {}", doc_name);
        }
        result
    })?;
    let bigram_time = bigram_start.elapsed();
    let bigram_size = bigram_index.memory_size();
    println!(
        "  Bigram index built with {} bigrams",
        bigram_index.index.len()
    );

    println!("Building coordinate index...");
    let coordinate_start = Instant::now();
    let coordinate_index = CoordinateIndex::from_dictionary_with_parser(&dictionary, |doc_name| {
        println!("  Processing document for coordinate index: {}", doc_name);
        let file_path = std::path::Path::new(input_dir).join(doc_name);
        let result = parser.parse_file(&file_path);
        if let Ok(ref words) = result {
            println!("    Parsed {} words from {}", words.len(), doc_name);
        } else {
            println!("    Failed to parse {}", doc_name);
        }
        result
    })?;
    let coordinate_time = coordinate_start.elapsed();
    let coordinate_size = coordinate_index.memory_size();
    println!(
        "  Coordinate index built with {} terms",
        coordinate_index.index.len()
    );

    println!("Building wildcard search engine...");
    let wildcard_start = Instant::now();
    let wildcard_engine = WildcardSearchEngine::from_compressed_dictionary(dictionary.clone());
    let wildcard_time = wildcard_start.elapsed();
    let wildcard_stats = wildcard_engine.memory_size();

    println!(
        "Incidence Matrix: {} bytes, built in {:.2?}",
        incidence_size, incidence_time
    );
    println!(
        "Inverted Index: {} bytes, built in {:.2?}",
        inverted_size, inverted_time
    );
    println!(
        "Bigram Index: {} bytes, built in {:.2?}",
        bigram_size, bigram_time
    );
    println!(
        "Coordinate Index: {} bytes, built in {:.2?}",
        coordinate_size, coordinate_time
    );
    println!(
        "Wildcard Engine: {} bytes, built in {:.2?}",
        wildcard_stats.total_size, wildcard_time
    );

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

    let wildcard_path = format!("{}_wildcard.bin", output_prefix);
    let wildcard_data = bincode::serialize(&wildcard_engine)?;
    fs::write(&wildcard_path, wildcard_data)?;

    println!("Saved incidence matrix to: {}", matrix_path);
    println!("Saved inverted index to: {}", index_path);
    println!("Saved bigram index to: {}", bigram_path);
    println!("Saved coordinate index to: {}", coordinate_path);
    println!("Saved wildcard engine to: {}", wildcard_path);

    println!("\n=== STRUCTURE COMPARISON ===");
    println!("Incidence Matrix:      {} bytes", incidence_size);
    println!("Inverted Index:        {} bytes", inverted_size);
    println!("Bigram Index:          {} bytes", bigram_size);
    println!("Coordinate Index:      {} bytes", coordinate_size);
    println!("Wildcard Engine:       {} bytes", wildcard_stats.total_size);
    println!(
        "  - Suffix Tree:       {} bytes",
        wildcard_stats.suffix_tree_size
    );
    println!(
        "  - Permutation Index: {} bytes",
        wildcard_stats.permutation_index_size
    );
    println!(
        "  - Trigram Index:     {} bytes",
        wildcard_stats.trigram_index_size
    );

    let efficiency_ratio = inverted_size as f64 / incidence_size as f64;
    println!("Space efficiency (Index/Matrix): {:.2}x", efficiency_ratio);
    let coordinate_ratio = coordinate_size as f64 / inverted_size as f64;
    println!(
        "Space overhead (Coordinate/Inverted): {:.2}x",
        coordinate_ratio
    );

    println!("\n=== DATA STRUCTURE ANALYSIS ===");
    println!("Incidence Matrix:");
    println!("  - Space: O(|T| × |D|) where T=terms, D=documents");
    println!("  - Search: O(|T|) for term lookup + O(|D|) for Boolean operations");
    println!(
        "  - Memory: {} bits per term-document pair",
        incidence_matrix.matrix.len() * incidence_matrix.documents.len()
    );

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
        println!(
            "Saved {} format: {} ({} bytes, {:.2?})",
            format, file_path, size, save_time
        );
    }

    println!("\n=== FORMAT COMPARISON ===");
    format_sizes.sort_by_key(|(_, size, _)| *size);
    for (format, size, time) in &format_sizes {
        println!("{:>6}: {:>10} bytes ({:>6.2?})", format, size, time);
    }

    if let Some((smallest_format, smallest_size, _)) = format_sizes.first() {
        if let Some((largest_format, largest_size, _)) = format_sizes.last() {
            let ratio = *largest_size as f64 / *smallest_size as f64;
            println!(
                "Size ratio ({}:{}): {:.2}x",
                largest_format, smallest_format, ratio
            );
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
    let wildcard_path = format!("{}_wildcard.bin", dict_prefix);

    println!("Loading search structures...");

    let matrix_data = fs::read(&matrix_path)?;
    let incidence_matrix: IncidenceMatrix = bincode::deserialize(&matrix_data)?;

    let index_data = fs::read(&index_path)?;
    let inverted_index: CompressedInvertedIndex = bincode::deserialize(&index_data)?;

    let bigram_data = fs::read(&bigram_path)?;
    let bigram_index: BigramIndex = bincode::deserialize(&bigram_data)?;

    let coordinate_data = fs::read(&coordinate_path)?;
    let coordinate_index: CoordinateIndex = bincode::deserialize(&coordinate_data)?;

    let wildcard_data = fs::read(&wildcard_path)?;
    let wildcard_engine: WildcardSearchEngine = bincode::deserialize(&wildcard_data)?;

    println!("Query: {}", query);
    println!("\n=== INCIDENCE MATRIX SEARCH ===");
    let matrix_start = Instant::now();
    match incidence_matrix.search(query) {
        Ok(result) => {
            let matrix_time = matrix_start.elapsed();
            let matching_docs = incidence_matrix.get_matching_documents(&result);
            println!(
                "Found {} documents in {:.2?}",
                matching_docs.len(),
                matrix_time
            );
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

    if query.contains('*') || query.contains('?') {
        println!("\n=== WILDCARD SEARCH ===");
        let wildcard_result = wildcard_engine.search_with_stats(query);

        println!("Strategy: {}", wildcard_result.strategy);
        println!("Search time: {:.2?}", wildcard_result.search_time);

        if let Some(error) = wildcard_result.error {
            println!("Error: {}", error);
        } else {
            let mut docs: Vec<_> = wildcard_result.documents.iter().collect();
            docs.sort();
            println!("Found {} documents", docs.len());
            for doc in &docs {
                println!("  - {}", doc);
            }
        }

        println!("\n=== WILDCARD SEARCH EXAMPLES ===");
        println!("Prefix wildcard: cat* - finds 'cat', 'cats', 'category', etc.");
        println!("Suffix wildcard: *ing - finds 'running', 'testing', 'building', etc.");
        println!("Middle wildcard: c*t - finds 'cat', 'cart', 'coat', etc.");
        println!("Multiple wildcards: *test* - finds any word containing 'test'");
        println!("Single character: c?t - finds 'cat', 'cut', 'cot', etc.");
    }

    Ok(())
}

fn handle_parquet_inspect_command(matches: &clap::ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
    let input_file = matches.get_one::<String>("input").unwrap();

    println!("Inspecting Parquet file: {}", input_file);
    let loader = ParquetLoader::new(input_file);
    loader.inspect_schema()?;

    Ok(())
}

fn handle_parquet_build_command(matches: &clap::ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
    let input_file = matches.get_one::<String>("input").unwrap();
    let output_prefix = matches.get_one::<String>("output").unwrap();
    let use_spimi = matches.get_flag("spimi");
    let memory_limit: usize = matches.get_one::<String>("memory-limit").unwrap().parse()?;

    println!("Processing Parquet file: {}", input_file);
    let loader = ParquetLoader::new(input_file);

    let start_time = Instant::now();
    let documents = loader.load_documents()?;
    let load_time = start_time.elapsed();

    println!("Loaded {} documents in {:.2?}", documents.len(), load_time);

    let dictionary = if use_spimi {
        println!("Building dictionary using SPIMI indexing (memory limit: {} MB)", memory_limit);
        let build_start = Instant::now();

        let doc_pairs: Vec<(String, String)> = documents
            .into_iter()
            .map(|doc| (doc.id, doc.text))
            .collect();

        let indexer = ParallelSPIMIIndexer::new(memory_limit, "./spimi_temp", None)?;
        let regular_dictionary = indexer.build_index(doc_pairs, |processed, total| {
            if processed % 10000 == 0 {
                println!("SPIMI: Processed {}/{} documents", processed, total);
            }
        })?;

        let build_time = build_start.elapsed();
        println!("SPIMI indexing completed in {:.2?}", build_time);

        println!("Compressing dictionary...");
        let compress_start = Instant::now();
        let dictionary = grimoire::CompressedDictionary::from_dictionary(&regular_dictionary);
        let compress_time = compress_start.elapsed();
        println!("Dictionary compression completed in {:.2?}", compress_time);

        dictionary
    } else {
        println!("Building dictionary using traditional method");
        let build_start = Instant::now();

        let mut regular_dictionary = grimoire::Dictionary::new();

        for (i, doc) in documents.iter().enumerate() {
            if i % 10000 == 0 {
                println!("Processing document {}/{}", i, documents.len());
            }

            let words: Vec<String> = doc.text
                .split_whitespace()
                .map(|word| {
                    word.chars()
                        .filter(|c| c.is_alphanumeric())
                        .collect::<String>()
                        .to_lowercase()
                })
                .filter(|word| !word.is_empty() && word.len() > 2)
                .collect();

            regular_dictionary.add_file_stats(doc.text.len() as u64);

            for word in words {
                regular_dictionary.add_term(word, doc.id.clone());
            }
        }

        let build_time = build_start.elapsed();
        println!("Traditional indexing completed in {:.2?}", build_time);

        println!("Compressing dictionary...");
        let compress_start = Instant::now();
        let dictionary = grimoire::CompressedDictionary::from_dictionary(&regular_dictionary);
        let compress_time = compress_start.elapsed();
        println!("Dictionary compression completed in {:.2?}", compress_time);

        dictionary
    };

    println!("\n=== PARQUET COLLECTION STATISTICS ===");
    println!("Collection size: {} bytes ({:.2} MB)",
             dictionary.collection_size_bytes,
             dictionary.collection_size_bytes as f64 / 1_048_576.0);
    println!("Total documents: {}", dictionary.total_documents);
    println!("Total words: {}", dictionary.total_words);
    println!("Dictionary size: {} unique terms", dictionary.dictionary_size());

    // Build search structures
    println!("\n=== BUILDING SEARCH STRUCTURES ===");

    let incidence_start = Instant::now();
    let incidence_matrix = IncidenceMatrix::from_dictionary(&dictionary);
    let incidence_time = incidence_start.elapsed();
    let incidence_size = incidence_matrix.memory_size();

    let inverted_start = Instant::now();
    let inverted_index = CompressedInvertedIndex::from_compressed_dictionary(&dictionary);
    let inverted_time = inverted_start.elapsed();
    let inverted_size = inverted_index.memory_size();

    println!("Building wildcard search engine...");
    let wildcard_start = Instant::now();
    let wildcard_engine = WildcardSearchEngine::from_compressed_dictionary(dictionary.clone());
    let wildcard_time = wildcard_start.elapsed();
    let wildcard_stats = wildcard_engine.memory_size();

    println!("Incidence Matrix: {} bytes, built in {:.2?}", incidence_size, incidence_time);
    println!("Inverted Index: {} bytes, built in {:.2?}", inverted_size, inverted_time);
    println!("Wildcard Engine: {} bytes, built in {:.2?}", wildcard_stats.total_size, wildcard_time);

    // Save indexes
    let matrix_path = format!("{}_matrix.bin", output_prefix);
    let index_path = format!("{}_index.bin", output_prefix);
    let wildcard_path = format!("{}_wildcard.bin", output_prefix);

    let matrix_data = bincode::serialize(&incidence_matrix)?;
    fs::write(&matrix_path, matrix_data)?;

    let index_data = bincode::serialize(&inverted_index)?;
    fs::write(&index_path, index_data)?;

    let wildcard_data = bincode::serialize(&wildcard_engine)?;
    fs::write(&wildcard_path, wildcard_data)?;

    println!("Saved incidence matrix to: {}", matrix_path);
    println!("Saved inverted index to: {}", index_path);
    println!("Saved wildcard engine to: {}", wildcard_path);

    // Save dictionary
    let dict_path = format!("{}.bin", output_prefix);
    let dict_size = dictionary.save_as_binary(&dict_path)?;
    println!("Saved dictionary to: {} ({} bytes)", dict_path, dict_size);

    Ok(())
}
