# FinePDFs Dataset Extraction Guide

Basic utility for searching, filtering, and extracting samples from the [HuggingFace FinePDFs dataset](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) using their [datatrove](https://github.com/huggingface/datatrove) library.


## Installation

```bash
pip install datatrove[all] datasets pyyaml
```

#### Usage

```bash
# Search for text in documents
python search_filter_extract_from_finepdfs.py --text-search "machine learning" --limit 100

# Filter by URL pattern
python search_filter_extract_from_finepdfs.py --url-pattern "arxiv.org" --limit 50

# Multiple filters (AND logic by default)
python search_filter_extract_from_finepdfs.py \
  --text-search "neural network" \
  --languages eng_Latn \
  --date-from 2023-01-01 \
  --token-count-min 1000 \
  --limit 100

# Use OR logic for filters
python search_filter_extract_from_finepdfs.py \
  --text-search "pytorch" \
  --text-search "tensorflow" \
  --filter-logic OR \
  --limit 50

# Use configuration file
python search_filter_extract_from_finepdfs.py --config extraction_config.yaml
```

#### Filters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--text-search` | Search in document text | `--text-search "deep learning"` |
| `--url-pattern` | Match URL patterns | `--url-pattern "arxiv.org"` |
| `--regex` | Use regex patterns | `--regex --text-search "neural.*network"` |
| `--languages` | Filter by language codes | `--languages eng_Latn fra_Latn` |
| `--dumps` | Filter by CommonCrawl dumps | `--dumps CC-MAIN-2024-42` |
| `--date-from` | Start date (ISO format) | `--date-from 2023-01-01` |
| `--date-to` | End date (ISO format) | `--date-to 2024-12-31` |
| `--token-count-min` | Minimum tokens | `--token-count-min 500` |
| `--token-count-max` | Maximum tokens | `--token-count-max 10000` |
| `--extractors` | Extractor type | `--extractors rolmOCR docling` |
| `--is-truncated` | Truncation status | `--is-truncated false` |
| `--lid-score-min` | Min language ID score | `--lid-score-min 0.9` |

#### Processing Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--limit` | Max samples to extract | 100 |
| `--max-samples-to-check` | Max samples to examine | limit × 100 |
| `--batch-size` | Processing batch size | 100 |
| `--workers` | Parallel workers | 1 |
| `--use-streaming` | Use streaming mode | False |
| `--clean-logs` | Clean logs and force re-extraction | False |

#### Output Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--output-dir` | Output directory | ./extracted_samples |
| `--output-format` | Format (jsonl/parquet/csv) | jsonl |
| `--compression` | Compression type | None |
| `--save-full` | Save complete documents | True |
| `--save-config` | Save config to output | False |


## Configuration Files

You can use YAML or JSON configuration files for complex extraction setups:

```yaml
# extraction_config.yaml
text_search: "machine learning"
languages:
  - eng_Latn
  - fra_Latn
date_from: "2023-01-01"
token_count_min: 1000
limit: 100
use_streaming: true
output_dir: ./ml_papers
```

Then run:
```bash
python search_filter_extract_from_finepdfs.py --config extraction_config.yaml
```

## Dataset Structure

Each extracted document contains:

- `text`: Full document text
- `id`: Unique document identifier
- `url`: Source URL
- `dump`: CommonCrawl dump ID
- `date`: Crawl date
- `language`: Language code
- `token_count`: Number of tokens
- `extractor`: Extraction method (rolmOCR/docling)
- `is_truncated`: Whether document was truncated
- `lid_score`: Language identification confidence
- `page_ends`: Page boundary positions


## Output

Creates an output directory containing:
- Individual document files (if `--save-full` is enabled)
- `extraction_summary.json` with statistics and configuration
- Extracted samples in the specified format (JSONL/Parquet/CSV)

## Notes

- The FinePDFs dataset is large (~20TB), so initial queries may take a bit to start. Use streaming mode for large extractions and better performance overall