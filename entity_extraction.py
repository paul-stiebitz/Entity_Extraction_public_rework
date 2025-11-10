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
    return OpenAI(
        base_url="http://localhost:11434/v1",  # Ollama endpoint
        api_key="ollama"  # Dummy but required
    )

MAX_RETRIES = 3
OLLAMA_TIMEOUT = 120


# =========================================
#  Entity Extraction Function
# =========================================
def extract_entities(mail_text, entity_types):
    
    print("Mail text non batch")
    print(mail_text)

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

Be accurate, concise, and preserve the wording of the original text. Note that this is an e-mail from a customer for us. Extract the important customer information.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{entity_instruction}\n\nEMAIL:\n{mail_text}"}
    ]

    for attempt in range(MAX_RETRIES):
        try:
            stream = client.chat.completions.create(
                model="granite4:tiny-h",
                messages=messages,
                stream=True,
                max_tokens=512,
                timeout=OLLAMA_TIMEOUT
            )

            response_text = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    token = chunk.choices[0].delta.content or ""
                    response_text += token
            return response_text

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                raise e


# =========================================
#  Batch Extraction (Parallel)
# =========================================
def batch_extract(mail_texts, selected_entities, max_workers=8):
    results = []
    
    print("Mail batch")
    print(mail_texts)
    print("-----------")
    for i, mail in enumerate(mail_texts):
        print(f"Mail {i}: {mail}")


    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_entities, mail, selected_entities): i
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

            completed += 1
            progress_bar.progress(completed / total)

        progress_bar.empty()

    results.sort(key=lambda x: x[0])
    return results


# =========================================
#  Streamlit App
# =========================================
def main():
    st.set_page_config(
        page_title="Entity Extraction Demo",
        page_icon="🧠",
        layout="wide"
    )

    # --- Styling ---
    st.markdown("""
        <style>
        :root {
            --ibm-blue: #0f62fe;
            --ibm-blue-dark: #0043ce;
        }
        #MainMenu, footer {visibility: hidden;}
        .ibm-header {
            background: linear-gradient(135deg, var(--ibm-blue), var(--ibm-blue-dark));
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div class="ibm-header">
            <h1>🧠 Entity Extraction Playground</h1>
            <p>Powered by Granite4 and Ollama</p>
        </div>
        """, unsafe_allow_html=True)

    # --- Input Selection ---
    st.markdown("### 📥 Choose Input Method")
    input_option = st.radio(
        "Select how to provide email text:",
        ["Paste manually", "Read from file path", "Batch from file (parallel)"]
    )

    mail_texts = []

    if input_option == "Paste manually":
        mail_text = st.text_area(
            "✉️ Paste your email or message here",
            height=200,
            placeholder="Hi John, please schedule a meeting with Sarah from IBM next Tuesday at 3 PM about the new POWER10 project..."
        )
        if mail_text.strip():
            mail_texts = [mail_text]

    else:
        file_path = st.text_input(
            "📄 Enter path to your .txt file:",
            value="./example_mails/emails.txt",
            help="Provide a path to a text file containing one or more emails separated by blank lines."
        )

        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                mail_texts = [t.strip() for t in re.split(r'\n\s*---\s*\n', content) if t.strip()]
                st.success(f"✅ Loaded {len(mail_texts)} emails from {file_path}")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        else:
            st.warning(f"⚠️ File not found: {file_path}")

    # --- Entity Types ---
    st.markdown("### 🏷️ Select Entity Types")
    default_entities = ["Person", "Orders", "Organization", "Date", "Time", "Location", "Money", "Product"]
    selected_entities = st.multiselect(
        "Select entities you want to extract",
        default_entities,
        default=["Organization", "Date"]
    )

    # --- Run Extraction ---
    if st.button("🚀 Extract Entities", use_container_width=True):
        if not mail_texts:
            st.error("Please enter or load at least one email.")
            return

        if input_option == "Batch from file (parallel)":
            st.info(f"Running parallel extraction for {len(mail_texts)} emails...")
            with st.spinner("Processing in parallel..."):
                results = batch_extract(mail_texts, selected_entities, max_workers=8)

                for i, res_text in results:
                    st.subheader(f"📨 Email #{i+1}")
                    try:
                        parsed = json.loads(res_text)
                        st.json(parsed)
                    except Exception:
                        st.code(res_text, language="json")

        else:
            for i, mail in enumerate(mail_texts, start=1):
                st.subheader(f"📨 Email #{i}")
                st.markdown(f"```\n{mail}\n```")
                st.info("Extracting entities... please wait.")
                with st.spinner("Communicating with model..."):
                    try:
                        response_text = extract_entities(mail, selected_entities)
                        try:
                            result = json.loads(response_text)
                            st.success("✅ Entities extracted successfully!")
                            st.json(result)
                        except Exception:
                            st.warning("⚠️ Model output was not perfect JSON; showing raw output.")
                            st.code(response_text, language="json")
                    except Exception as e:
                        st.error(f"Error extracting entities: {str(e)}")

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit and Granite4")


if __name__ == "__main__":
    main()

