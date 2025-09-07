import re
import sys
import json
import shutil
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.writers import JsonlWriter, ParquetWriter
from datatrove.data import Document
from datasets import load_dataset


class FinePDFsExtractor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.samples_found = 0
        self.samples_checked = 0
        self.stats = {
            'total_checked': 0,
            'total_matched': 0,
            'by_language': {},
            'by_dump': {},
            'by_extractor': {}
        }

    def create_filter(self) -> Callable[[Document], bool]:
        filters = []

        if self.config.get('text_search'):
            pattern = self.config['text_search']
            if self.config.get('regex', False):
                filters.append(lambda doc: bool(
                    re.search(pattern, doc.text or '', re.IGNORECASE)))
            else:
                filters.append(lambda doc: pattern.lower()
                               in (doc.text or '').lower())

        if self.config.get('url_pattern'):
            pattern = self.config['url_pattern']
            if self.config.get('regex', False):
                filters.append(lambda doc: bool(re.search(pattern, doc.metadata.get(
                    'url', ''), re.IGNORECASE)) if hasattr(doc, 'metadata') else False)
            else:
                filters.append(lambda doc: pattern.lower() in doc.metadata.get(
                    'url', '').lower() if hasattr(doc, 'metadata') else False)

        if self.config.get('languages'):
            languages = set(self.config['languages'])
            filters.append(lambda doc: doc.metadata.get(
                'language') in languages if hasattr(doc, 'metadata') else False)

        if self.config.get('dumps'):
            dumps = set(self.config['dumps'])
            filters.append(lambda doc: doc.metadata.get('dump')
                           in dumps if hasattr(doc, 'metadata') else False)

        if self.config.get('date_from') or self.config.get('date_to'):
            date_from = datetime.fromisoformat(
                self.config.get('date_from', '1900-01-01'))
            date_to = datetime.fromisoformat(
                self.config.get('date_to', '2100-01-01'))

            def date_filter(doc):
                if hasattr(doc, 'metadata') and doc.metadata.get('date'):
                    try:
                        doc_date = datetime.fromisoformat(
                            doc.metadata['date'].replace('Z', '+00:00'))
                        return date_from <= doc_date <= date_to
                    except:
                        return False
                return False

            filters.append(date_filter)

        if self.config.get('token_count_min') or self.config.get('token_count_max'):
            min_tokens = self.config.get('token_count_min', 0)
            max_tokens = self.config.get('token_count_max', float('inf'))
            filters.append(lambda doc: min_tokens <= doc.metadata.get(
                'token_count', 0) <= max_tokens if hasattr(doc, 'metadata') else False)

        if self.config.get('extractors'):
            extractors = set(self.config['extractors'])
            filters.append(lambda doc: doc.metadata.get(
                'extractor') in extractors if hasattr(doc, 'metadata') else False)

        if 'is_truncated' in self.config:
            is_truncated = self.config['is_truncated']
            filters.append(lambda doc: doc.metadata.get(
                'is_truncated') == is_truncated if hasattr(doc, 'metadata') else False)

        if self.config.get('lid_score_min'):
            min_score = self.config['lid_score_min']
            filters.append(lambda doc: doc.metadata.get(
                'full_doc_lid_score', 0) >= min_score if hasattr(doc, 'metadata') else False)

        if not filters:
            return lambda doc: True

        if self.config.get('filter_logic', 'AND') == 'AND':
            def combined_filter(doc):
                self.samples_checked += 1
                result = all(f(doc) for f in filters)
                if result:
                    self.samples_found += 1
                    self.update_stats(doc)
                return result
        else:
            def combined_filter(doc):
                self.samples_checked += 1
                result = any(f(doc) for f in filters)
                if result:
                    self.samples_found += 1
                    self.update_stats(doc)
                return result

        return combined_filter

    def update_stats(self, doc: Document):
        if hasattr(doc, 'metadata'):
            meta = doc.metadata

            lang = meta.get('language', 'unknown')
            self.stats['by_language'][lang] = self.stats['by_language'].get(
                lang, 0) + 1

            dump = meta.get('dump', 'unknown')
            self.stats['by_dump'][dump] = self.stats['by_dump'].get(
                dump, 0) + 1

            extractor = meta.get('extractor', 'unknown')
            self.stats['by_extractor'][extractor] = self.stats['by_extractor'].get(
                extractor, 0) + 1

    def extract_streaming(self):
        print("Using streaming mode for extraction...")

        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        language = self.config.get('languages', ['eng_Latn'])[
            0] if self.config.get('languages') else 'eng_Latn'

        dataset = load_dataset(
            "HuggingFaceFW/finepdfs",
            name=language,
            split="train",
            streaming=True
        )

        filter_func = self.create_filter()

        extracted_samples = []
        limit = self.config.get('limit', 100)
        max_to_check = self.config.get('max_samples_to_check', limit * 100)
        batch_size = self.config.get('batch_size', 100)

        print(f"Searching for up to {limit} matching documents...")
        print(f"Will check maximum {max_to_check} samples")

        for sample in dataset:
            doc = type('Document', (), {
                'text': sample.get('text', ''),
                'metadata': sample
            })()

            if filter_func(doc):
                extracted_samples.append(sample)

                if self.config.get('save_full', True):
                    doc_path = output_dir / f"doc_{len(extracted_samples):05d}.json"
                    with open(doc_path, 'w') as f:
                        json.dump(sample, f, indent=2)

                print(f"Found match #{len(extracted_samples)} (checked {self.samples_checked} samples)")

                if len(extracted_samples) >= limit:
                    break

            if self.samples_checked % batch_size == 0:
                print(f"Progress: checked {self.samples_checked}, found {len(extracted_samples)} matches")

            if self.samples_checked >= max_to_check:
                print(f"Reached maximum samples to check ({max_to_check})")
                break

        self.save_results(extracted_samples, output_dir)

    def extract_datatrove(self):
        print("Using datatrove pipeline for extraction...")

        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        language = self.config.get('languages', ['eng_Latn'])[
            0] if self.config.get('languages') else 'eng_Latn'

        filter_func = self.create_filter()

        pipeline = [
            ParquetReader(
                f"hf://datasets/HuggingFaceFW/finepdfs/data/{language}/train",
                limit=self.config.get('max_samples_to_check', 10000)
            ),
            LambdaFilter(filter_func)
        ]

        output_format = self.config.get('output_format', 'jsonl')
        if output_format == 'parquet':
            pipeline.append(ParquetWriter(
                str(output_dir),
                output_filename="extracted_samples"
            ))
        else:
            pipeline.append(JsonlWriter(
                str(output_dir),
                output_filename="extracted_samples",
                compression=self.config.get('compression')
            ))

        executor = LocalPipelineExecutor(
            pipeline=pipeline,
            tasks=self.config.get('workers', 1),
            logging_dir=str(output_dir / "logs")
        )

        print(f"Running extraction pipeline with {self.config.get('workers', 1)} workers...")
        executor.run()

        print(f"  Extraction complete!")
        print(f"  Samples checked: {self.samples_checked}")
        print(f"  Samples matched: {self.samples_found}")

    def save_results(self, samples: List[Dict], output_dir: Path):
        summary = {
            'extraction_config': self.config,
            'statistics': {
                'total_checked': self.samples_checked,
                'total_matched': len(samples),
                'by_language': self.stats['by_language'],
                'by_dump': self.stats['by_dump'],
                'by_extractor': self.stats['by_extractor']
            },
            'samples': []
        }

        for sample in samples[:100]:
            summary['samples'].append({
                'id': sample.get('id'),
                'url': sample.get('url'),
                'dump': sample.get('dump'),
                'language': sample.get('language'),
                'token_count': sample.get('token_count'),
                'extractor': sample.get('extractor'),
                'text_preview': sample.get('text', '')[:200] if sample.get('text') else None
            })

        summary_file = output_dir / "extraction_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n Summary saved to: {summary_file}")

        print("\n Extraction Statistics:")
        print(f"  Total samples checked: {self.samples_checked}")
        print(f"  Total matches found: {len(samples)}")

        if self.stats['by_language']:
            print("\n  By language:")
            for lang, count in sorted(self.stats['by_language'].items()):
                print(f"    {lang}: {count}")

        if self.stats['by_dump']:
            print("\n  By dump:")
            for dump, count in sorted(self.stats['by_dump'].items())[:5]:
                print(f"    {dump}: {count}")
            if len(self.stats['by_dump']) > 5:
                print(f"    ... and {len(self.stats['by_dump']) - 5} more")

        if self.stats['by_extractor']:
            print("\n  By extractor:")
            for extractor, count in sorted(self.stats['by_extractor'].items()):
                print(f"    {extractor}: {count}")


def load_config(config_file: str) -> Dict[str, Any]:
    config_path = Path(config_file)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_path, 'r') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        else:
            return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Extract samples from FinePDFs dataset with flexible filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search for arxiv mentions in text
  python search_filter_extract_from_finepdfs.py --text-search "arxiv" --limit 100
  
  # Filter by URL pattern
  python search_filter_extract_from_finepdfs.py --url-pattern "arxiv.org" --limit 50
  
  # Filter by multiple parameters
  python search_filter_extract_from_finepdfs.py \\
    --languages eng_Latn fra_Latn \\
    --date-from 2020-01-01 \\
    --token-count-min 1000 \\
    --batch-size 500
  
  # Use configuration file
  python search_filter_extract_from_finepdfs.py --config extraction_config.yaml
        """
    )

    parser.add_argument('--config', type=str,
                        help='Configuration file (YAML or JSON)')

    parser.add_argument('--text-search', type=str,
                        help='Search pattern in text content')
    parser.add_argument('--url-pattern', type=str,
                        help='Pattern to match in URLs')
    parser.add_argument('--regex', action='store_true',
                        help='Use regex for text/URL patterns')
    parser.add_argument('--languages', nargs='+',
                        help='Filter by language codes (e.g., eng_Latn fra_Latn)')
    parser.add_argument('--dumps', nargs='+',
                        help='Filter by CommonCrawl dump IDs')
    parser.add_argument('--date-from', type=str,
                        help='Start date (ISO format: YYYY-MM-DD)')
    parser.add_argument('--date-to', type=str,
                        help='End date (ISO format: YYYY-MM-DD)')
    parser.add_argument('--token-count-min', type=int,
                        help='Minimum token count')
    parser.add_argument('--token-count-max', type=int,
                        help='Maximum token count')
    parser.add_argument('--extractors', nargs='+',
                        choices=['rolmOCR', 'docling'], help='Filter by extractor type')
    parser.add_argument('--is-truncated', type=bool,
                        help='Filter by truncation status')
    parser.add_argument('--lid-score-min', type=float,
                        help='Minimum language ID score')
    parser.add_argument(
        '--filter-logic', choices=['AND', 'OR'], default='AND', help='Logic for combining filters')

    parser.add_argument('--limit', type=int, default=100,
                        help='Maximum number of samples to extract')
    parser.add_argument('--max-samples-to-check', type=int,
                        help='Maximum samples to check (default: limit * 100)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for processing')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers')
    parser.add_argument('--use-streaming', action='store_true',
                        help='Use streaming mode instead of datatrove')

    parser.add_argument('--output-dir', type=str,
                        default='./extracted_samples', help='Output directory')
    parser.add_argument(
        '--output-format', choices=['jsonl', 'parquet', 'csv'], default='jsonl', help='Output format')
    parser.add_argument('--compression', type=str,
                        help='Compression type (gzip, bz2, etc.)')
    parser.add_argument('--save-full', action='store_true',
                        default=True, help='Save full documents')
    parser.add_argument('--save-config', action='store_true',
                        help='Save configuration to output directory')
    parser.add_argument('--clean-logs', action='store_true',
                        help='Clean up existing logs and datatrove state before execution')

    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        for key, value in vars(args).items():
            if value is not None and key != 'config':
                config[key] = value
    else:
        config = {k: v for k, v in vars(
            args).items() if v is not None and k != 'config'}

    if 'max_samples_to_check' not in config:
        config['max_samples_to_check'] = config.get('limit', 100) * 100

    if config.get('clean_logs'):
        output_dir = Path(config.get('output_dir', './extracted_samples'))
        if output_dir.exists():
            logs_dir = output_dir / 'logs'
            if logs_dir.exists():
                print(f"Cleaning up logs directory: {logs_dir}")
                shutil.rmtree(logs_dir)
            
            for json_file in output_dir.glob('*.json'):
                if 'extraction_summary' not in json_file.name and 'extraction_config' not in json_file.name:
                    print(f"Removing state file: {json_file}")
                    json_file.unlink()
            
            if not config.get('use_streaming'):
                for pattern in ['*.parquet', '*.jsonl', '*.jsonl.gz']:
                    for file in output_dir.glob(pattern):
                        print(f"Removing output file: {file}")
                        file.unlink()
                        
        print("Log cleanup completed. Starting fresh extraction...")

    if config.get('save_config'):
        output_dir = Path(config.get('output_dir', './extracted_samples'))
        output_dir.mkdir(parents=True, exist_ok=True)
        config_file = output_dir / 'extraction_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved to: {config_file}")

    extractor = FinePDFsExtractor(config)

    if config.get('use_streaming'):
        extractor.extract_streaming()
    else:
        extractor.extract_datatrove()


if __name__ == "__main__":
    main()
