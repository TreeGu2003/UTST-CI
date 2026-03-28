import os
import sys
import pandas as pd
import concurrent.futures
import logging
import time
import multiprocessing
from tqdm import tqdm
from langchain_openai import ChatOpenAI
import textstat
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.readability_utils import get_readability_level

# Initialize LLM from environment variables
llm = ChatOpenAI(
    model="gpt-4o-mini",
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/"),
    api_key=os.environ.get("OPENAI_API_KEY", ""),
)

readability_map = {
    'elementary': 'Elementary (Flesch >= 80)',
    'middle': 'Middle School (Flesch 60-80)',
    'high': 'High School (Flesch 40-60)',
    'college': 'College (Flesch < 40)'
}

conversion_rules = {
    'elementary': 'middle',
    'middle': 'high',
    'high': 'college',
    'college': 'elementary'
}

MAX_RETRY_DEPTH = 5

def classify_readability(text):
    """Compute Flesch Reading Ease score and return readability level."""
    try:
        flesch_score = textstat.flesch_reading_ease(text)
        level, _ = get_readability_level(flesch_score)
        return level, flesch_score
    except Exception as e:
        logging.error(f"Readability classification error: {e}")
        return None, None

def rewrite_text(text, input_level, output_level, max_attempts=10):
    """Rewrite text to target readability level using GPT-4o-mini."""
    input_desc = readability_map[input_level]
    output_desc = readability_map[output_level]

    prompt = (
        f"Rewrite the following text to match the {output_desc} readability level "
        f"(Flesch Reading Ease score {output_desc.split('(')[1][:-1]}). "
        f"Adjust vocabulary, sentence length, and complexity to strongly reflect the target readability level. "
        f"For Elementary, use very simple words and short sentences. "
        f"For Middle School, use moderately simple words and slightly longer sentences. "
        f"For High School, use more complex words and varied sentence structures. "
        f"For College, use advanced vocabulary and complex sentence structures. "
        f"Keep the core meaning and content consistent, but adapt the style to be natural and concise. "
        f"Input: {text}\n"
        f"Output:"
    )

    for attempt in range(max_attempts):
        try:
            response = llm.invoke(prompt)
            rewritten_text = response.content.strip()
            predicted_level, flesch_score = classify_readability(rewritten_text)
            if predicted_level == output_level:
                return rewritten_text, flesch_score
            else:
                logging.warning(
                    f"Attempt {attempt + 1}: level {predicted_level} (Flesch: {flesch_score}), expected {output_level}"
                )
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")

        if attempt < max_attempts - 1:
            time.sleep(1)

    logging.error(f"All {max_attempts} attempts failed for text: '{text[:50]}...'")
    return None, None

def process_text(row, output_level, df, input_level, samples_per_pair, _depth=0):
    """Process a single text to generate target readability level output."""
    text = row['input_noprompt']
    input_level = row['label']
    rewritten_text, flesch_score = rewrite_text(text, input_level, output_level)

    if rewritten_text:
        return {
            'input_text': text,
            'input_level': input_level,
            'output_text': rewritten_text,
            'output_level': output_level,
            'flesch_score': flesch_score
        }

    if _depth >= MAX_RETRY_DEPTH:
        logging.error(f"Max retry depth reached for {input_level} -> {output_level}, skipping")
        return None

    logging.info(f"Switching to a new sample for {input_level} -> {output_level}")
    new_subset = df[df['label'] == input_level].sample(n=1, random_state=None)
    new_row = new_subset.iloc[0]
    return process_text(new_row, output_level, df, input_level, samples_per_pair, _depth=_depth + 1)

file_lock = threading.Lock()

def save_to_csv_batch(results, output_path, append=True):
    """Save a batch of results to CSV file (thread-safe)."""
    try:
        df_batch = pd.DataFrame(results)
        mode = 'a' if append else 'w'
        header = not append
        with file_lock:
            df_batch.to_csv(output_path, mode=mode, header=header, index=False)
        logging.info(f"Saved {len(df_batch)} rows to {output_path}")
    except Exception as e:
        logging.error(f"Error saving to CSV: {e}")

def generate_parallel_data(df, output_path, pairs, samples_per_pair=2500, max_workers=None, batch_size=100):
    """Generate parallel readability data and save incrementally to CSV."""
    start_time = time.time()
    results = []
    batch_results = []

    if max_workers is None:
        max_workers = multiprocessing.cpu_count() * 2

    logging.info(f"Starting parallel data generation with {max_workers} workers...")

    for input_level, output_level in pairs:
        logging.info(f"Processing pair: {readability_map[input_level]} -> {readability_map[output_level]}")
        subset = df[df['label'] == input_level].sample(n=samples_per_pair, random_state=42)
        total_samples = len(subset)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(process_text, row, output_level, df, input_level, samples_per_pair): idx
                for idx, row in subset.iterrows()
            }

            with tqdm(total=total_samples, desc=f"Generating {input_level} -> {output_level}", unit="text") as pbar:
                for future in concurrent.futures.as_completed(future_to_index):
                    try:
                        result = future.result()
                        if result:
                            batch_results.append(result)
                            if len(batch_results) >= batch_size:
                                save_to_csv_batch(batch_results, output_path, append=len(results) > 0)
                                results.extend(batch_results)
                                batch_results = []
                    except Exception as e:
                        logging.error(f"Error processing text: {e}")
                    pbar.update(1)

        if batch_results:
            save_to_csv_batch(batch_results, output_path, append=len(results) > 0)
            results.extend(batch_results)
            batch_results = []

    if results:
        logging.info(f"Generated {len(results)} parallel data pairs")

    elapsed_time = time.time() - start_time
    logging.info(f"Completed in {elapsed_time:.2f} seconds. Saved data to {output_path}")

    return pd.DataFrame(results)

def main():
    input_path = "combined_summaries.csv"
    output_path = "parallel_readability_data_1.csv"

    logging.info(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    logging.info(f"Data loaded. Total rows: {len(df)}")

    pairs = [
        ('college', 'middle'),
        ('elementary', 'high'),
        ('high', 'elementary'),
        ('middle', 'college')
    ]

    generate_parallel_data(df, output_path, pairs, samples_per_pair=2500, batch_size=100)

if __name__ == "__main__":
    main()
