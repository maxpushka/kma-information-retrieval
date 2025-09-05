use arrow::array::{Array, Int64Array, StringArray};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct ParquetDocument {
    pub id: String,
    pub text: String,
    pub metadata: Option<String>,
}

pub struct ParquetLoader {
    file_path: String,
}

impl ParquetLoader {
    pub fn new<P: AsRef<Path>>(file_path: P) -> Self {
        ParquetLoader {
            file_path: file_path.as_ref().to_string_lossy().to_string(),
        }
    }

    pub fn load_documents(&self) -> Result<Vec<ParquetDocument>, Box<dyn std::error::Error>> {
        let file = File::open(&self.file_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let mut reader = builder.build()?;

        let mut documents = Vec::new();
        println!("Loading documents from Parquet file: {}", self.file_path);
        let mut total_batches = 0;
        let mut total_documents = 0;

        while let Some(batch) = reader.next() {
            let batch = batch?;
            total_batches += 1;
            
            if total_batches % 10 == 0 {
                println!("  Processed {} batches, {} documents so far", total_batches, total_documents);
            }

            let docs = self.process_batch(&batch)?;
            total_documents += docs.len();
            documents.extend(docs);
        }

        println!("Parquet loading complete: {} documents from {} batches", total_documents, total_batches);
        Ok(documents)
    }

    fn process_batch(&self, batch: &RecordBatch) -> Result<Vec<ParquetDocument>, Box<dyn std::error::Error>> {
        let schema = batch.schema();
        let mut documents = Vec::new();

        // Try to find text-like columns in the schema
        let mut text_column_idx = None;
        let mut id_column_idx = None;
        let mut metadata_column_idx = None;

        for (i, field) in schema.fields().iter().enumerate() {
            let name = field.name().to_lowercase();
            if name.contains("text") || name.contains("content") || name.contains("body") {
                text_column_idx = Some(i);
            } else if name.contains("id") && id_column_idx.is_none() {
                id_column_idx = Some(i);
            } else if name.contains("title") || name.contains("subject") || name.contains("category") {
                metadata_column_idx = Some(i);
            }
        }

        // If we don't find obvious text columns, look for string columns
        if text_column_idx.is_none() {
            for (i, field) in schema.fields().iter().enumerate() {
                if matches!(field.data_type(), arrow::datatypes::DataType::Utf8 | arrow::datatypes::DataType::LargeUtf8) {
                    if text_column_idx.is_none() {
                        text_column_idx = Some(i);
                    } else if id_column_idx.is_none() {
                        id_column_idx = Some(i);
                    }
                }
            }
        }

        let text_column_idx = text_column_idx.ok_or("No text column found in Parquet file")?;
        
        // Extract data from the identified columns
        let text_array = batch.column(text_column_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("Text column is not a string array")?;

        let id_array = id_column_idx.map(|idx| batch.column(idx));

        let metadata_array = if let Some(idx) = metadata_column_idx {
            Some(batch.column(idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or("Metadata column is not a string array")?)
        } else {
            None
        };

        for i in 0..batch.num_rows() {
            if text_array.is_valid(i) {
                let text = text_array.value(i);
                let id = if let Some(id_col) = id_array {
                    // Try different array types for ID
                    if let Some(string_arr) = id_col.as_any().downcast_ref::<StringArray>() {
                        if string_arr.is_valid(i) {
                            string_arr.value(i).to_string()
                        } else {
                            format!("doc_{}", i)
                        }
                    } else if let Some(int_arr) = id_col.as_any().downcast_ref::<Int64Array>() {
                        if int_arr.is_valid(i) {
                            int_arr.value(i).to_string()
                        } else {
                            format!("doc_{}", i)
                        }
                    } else {
                        format!("doc_{}", i)
                    }
                } else {
                    format!("doc_{}", i)
                };

                let metadata = if let Some(meta_arr) = metadata_array {
                    if meta_arr.is_valid(i) {
                        Some(meta_arr.value(i).to_string())
                    } else {
                        None
                    }
                } else {
                    None
                };

                documents.push(ParquetDocument {
                    id,
                    text: text.to_string(),
                    metadata,
                });
            }
        }

        Ok(documents)
    }

    pub fn inspect_schema(&self) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(&self.file_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let schema = builder.schema().clone();

        println!("Parquet file schema:");
        for (i, field) in schema.fields().iter().enumerate() {
            println!("  Column {}: {} ({})", i, field.name(), field.data_type());
        }

        // Read a small sample to understand the data
        let mut reader = builder.build()?;
        if let Some(Ok(batch)) = reader.next() {
            println!("\nSample data (first batch, up to 3 rows):");
            for row in 0..batch.num_rows().min(3) {
                println!("  Row {}:", row);
                for col in 0..batch.num_columns() {
                    let column = batch.column(col);
                    let field = schema.field(col);
                    
                    if let Some(string_array) = column.as_any().downcast_ref::<StringArray>() {
                        if string_array.is_valid(row) {
                            let value = string_array.value(row);
                            let preview = if value.len() > 100 {
                                format!("{}...", &value[..97])
                            } else {
                                value.to_string()
                            };
                            println!("    {}: {}", field.name(), preview);
                        }
                    } else if let Some(int_array) = column.as_any().downcast_ref::<Int64Array>() {
                        if int_array.is_valid(row) {
                            println!("    {}: {}", field.name(), int_array.value(row));
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parquet_loader_creation() {
        let loader = ParquetLoader::new("test.parquet");
        assert_eq!(loader.file_path, "test.parquet");
    }
}