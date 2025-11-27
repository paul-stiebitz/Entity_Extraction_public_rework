import streamlit as st
from openai import OpenAI
import time
import json
import os
import concurrent.futures
import re

# =========================================
#  Initialize OpenAI client (Ollama)
# =========================================
@st.cache_resource
def get_openai_client():
    """
    Returns a cached OpenAI client instance pointing to local Ollama endpoint.
    """
    print("[LOG] Initializing OpenAI client...")
    return OpenAI(
        base_url="http://localhost:11434/v1",  # Ollama endpoint
        api_key="ollama"  # Dummy API key required
    )

MAX_RETRIES = 3
OLLAMA_TIMEOUT = 120


# =========================================
#  Entity Extraction Function (Streaming)
# =========================================
def extract_entities(mail_text, entity_types):
    """
    Extracts entities from a single email using token-by-token streaming.
    """
    client = get_openai_client()
    entity_instruction = (
        f"Extract the following entity types: {', '.join(entity_types)}."
        if entity_types else
        "Extract all entities you can identify (names, dates, organizations, amounts, etc.)."
    )

    system_prompt = f"""
You are an expert information extraction assistant. 

You will receive an email text. Identify and extract entities according to the user request.
Output in valid JSON format with the structure:

{{
  "entities": [
    {{
      "type": "<entity type>",
      "text": "<exact phrase>",
      "context": "<short explanation or sentence context>"
    }}
  ]
}}

Be accurate, concise, and preserve the wording of the original text.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{entity_instruction}\n\nEMAIL:\n{mail_text}"}
    ]

    for attempt in range(MAX_RETRIES):
        try:
            print(f"[LOG] Starting streaming extraction attempt {attempt+1}...")
            stream = client.chat.completions.create(
                model="granite3.3:8b",
                messages=messages,
                stream=True,
                max_tokens=512,
                timeout=OLLAMA_TIMEOUT
            )

            # Yield token-by-token for live streaming
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    token = chunk.choices[0].delta.content or ""
                    yield token
            print("[LOG] Streaming extraction completed.")
            return

        except Exception as e:
            print(f"[ERROR] Streaming attempt {attempt+1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                raise e


# =========================================
#  Batch Extraction (Parallel)
# =========================================
def batch_extract(mail_texts, selected_entities, max_workers=8):
    """
    Extracts entities from multiple emails in parallel using threads.
    Returns a list of (index, response_text) tuples.
    """
    results = []
    print(f"[LOG] Starting batch extraction for {len(mail_texts)} emails with {max_workers} workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(lambda m=mail: "".join(list(extract_entities(m, selected_entities)))): i
            for i, mail in enumerate(mail_texts)
        }

        progress_bar = st.progress(0)
        total = len(futures)
        completed = 0

        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            try:
                response_text = future.result()
                results.append((i, response_text))
            except Exception as e:
                results.append((i, f"Error: {str(e)}"))
                print(f"[ERROR] Batch extraction failed for email {i}: {e}")

            completed += 1
            progress_bar.progress(completed / total)

        progress_bar.empty()

    results.sort(key=lambda x: x[0])
    print("[LOG] Batch extraction completed.")
    return results


# =========================================
#  Helper Functions for Timing
# =========================================
def format_seconds(seconds):
    """
    Converts seconds to a formatted string MMmin SSsec.
    """
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}min {secs}sec"


def measure_times(mail_texts, selected_entities, smt_values=[2, 4, 8], output_file="performance.txt"):
    """
    Measures execution times for serial (non-batch) vs parallel (batch) processing.
    Saves results to a text file.
    """
    results = []

    for smt in smt_values:
        print(f"[LOG] Measuring times for SMT={smt}...")
        results.append(f"SMT = {smt}\n")

        # --- Non-Batch (Streaming) ---
        non_batch_start = time.time()
        for mail in mail_texts[:smt]:
            "".join(list(extract_entities(mail, selected_entities)))
        non_batch_end = time.time()
        results.append(f"Non Batch: {format_seconds(non_batch_end - non_batch_start)}\n")
        print(f"[LOG] Non-Batch duration: {format_seconds(non_batch_end - non_batch_start)}")

        # --- Batch (Parallel) ---
        batch_start = time.time()
        batch_extract(mail_texts[:smt], selected_entities, max_workers=smt)
        batch_end = time.time()
        results.append(f"Batch: {format_seconds(batch_end - batch_start)}\n\n")
        print(f"[LOG] Batch duration: {format_seconds(batch_end - batch_start)}")

    # Save results to file
    with open(output_file, "w") as f:
        f.writelines(results)
    print(f"[LOG] Timing results saved to {output_file}")
    return results


# =========================================
#  Streamlit App
# =========================================
def main():
    st.set_page_config(page_title="Entity Extraction Demo", page_icon="🧠", layout="wide")

    # --- Styling ---
    st.markdown("""
        <style>
        :root { --ibm-blue: #0f62fe; --ibm-blue-dark: #0043ce; }
        #MainMenu, footer {visibility: hidden;}
        .ibm-header {
            background: linear-gradient(135deg, var(--ibm-blue), var(--ibm-blue-dark));
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            color: white;
        }
        .stream-output {
            background-color: #1e1e1e;
            color: #d4d4d4;
            padding: 1rem;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            overflow-x: auto;
            white-space: pre-wrap;
        }
        .stream-output .json-key { color: #9cdcfe; font-weight: bold; }
        .stream-output .json-string { color: #ce9178; }
        .stream-output .json-number { color: #b5cea8; }
        .stream-output .json-boolean { color: #569cd6; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="ibm-header">
            <h1>Entity Extraction Playground</h1>
            <p>Powered by Granite4 and Ollama</p>
        </div>
    """, unsafe_allow_html=True)

    # --- Input ---
    st.markdown("### Choose Input Method")
    input_option = st.radio(
        "Select how to provide email text:",
        ["Paste manually", "Read from file path", "Batch from file (parallel)"]
    )

    mail_texts = []
    if input_option == "Paste manually":
        mail_text = st.text_area("Paste your email or message here", height=200)
        if mail_text.strip():
            mail_texts = [mail_text]
    else:
        file_path = st.text_input("Enter path to your .txt file:", value="./example_mails/emails.txt")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            mail_texts = [t.strip() for t in re.split(r'\n\s*---\s*\n', content) if t.strip()]
            st.success(f"Loaded {len(mail_texts)} emails from {file_path}")
            print(f"[LOG] Loaded {len(mail_texts)} emails from {file_path}")
        else:
            st.warning(f"File not found: {file_path}")
            print(f"[WARN] File not found: {file_path}")

    # --- Entity Types ---
    default_entities = ["Person", "Orders", "Organization", "Date", "Time", "Location", "Money", "Product"]
    selected_entities = st.multiselect("Select entities to extract", default_entities, default=["Organization", "Date"])

    # --- Extraction / Streaming ---
    if st.button("Extract Entities", use_container_width=True):
        if not mail_texts:
            st.error("Please enter or load at least one email.")
            return

        if input_option == "Batch from file (parallel)":
            st.info(f"Running parallel extraction for {len(mail_texts)} emails...")
            results = batch_extract(mail_texts, selected_entities, max_workers=8)
            for i, res_text in results:
                st.subheader(f"Email #{i+1}")
                try:
                    st.json(json.loads(res_text))
                except:
                    st.code(res_text, language="json")
        else:
            for i, mail in enumerate(mail_texts, start=1):
                st.subheader(f"Email #{i}")
                st.markdown(f"```\n{mail}\n```")
                stream_area = st.empty()
                output = ""
                for token in extract_entities(mail, selected_entities):
                    output += token
                    display_text = output
                    # Apply JSON color formatting
                    display_text = re.sub(r'("(?:\\.|[^"\\])*")', r'<span class="json-string">\1</span>', display_text)
                    display_text = re.sub(r'\b(true|false|null)\b', r'<span class="json-boolean">\1</span>', display_text)
                    display_text = re.sub(r'\b(\d+(\.\d+)?)\b', r'<span class="json-number">\1</span>', display_text)
                    display_text = re.sub(r'(<span class="json-string">"(\w+)"</span>)(\s*:)', r'<span class="json-key">\2</span>:', display_text)
                    stream_area.markdown(f'<div class="stream-output">{display_text}</div>', unsafe_allow_html=True)

    # --- Timing Measurement ---
    st.markdown("---")
    st.markdown("### Measure Execution Times")
    if st.button("Measure Times for SMT 2,4,8"):
        if not mail_texts:
            st.error("Please load emails first")
        else:
            timing_results = measure_times(mail_texts, selected_entities)
            st.text("".join(timing_results))
            st.success("Timing results saved in timing_results.txt")

    st.caption("Built with Streamlit and Granite4")

if __name__ == "__main__":
    main()
