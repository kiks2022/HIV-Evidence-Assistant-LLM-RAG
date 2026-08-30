# 🧬 HIV Clinical Evidence Assistant

A clinical evidence research tool built for researchers. It retrieves real PubMed research and current HIV treatment guidelines, compares them side by side, and flags when they conflict — so the clinician can make their own informed judgment rather than getting a single answer from a black box.

**Live demo:** [HIV Clinical Evidence Assistant · Streamlit](https://hiv-evidence-assistant.streamlit.app/)

---

## The problem this solves

Anyone working in HIV care knows the frustration: US guidelines recommend one thing, WHO says another, and the Nigerian FMOH follows a third path — usually because of cost and drug availability, not because the evidence is different. A clinician in Lagos prescribing TDF/3TC/DTG does not need to know what Biktarvy costs in San Francisco. They need to know whether the evidence supports their regimen, and whether there is anything in the recent literature that should change their practice.

Standard chatbots give you one answer. This tool gives you the full picture — guideline recommendation, research evidence, where they agree or disagree, and what it means for clinical practice in your specific setting.

---

## What it does

Every question goes through four structured sections:

**Guideline Position** — presents both the WHO/Nigeria FMOH recommendation and the US DHHS recommendation side by side, with explicit labels for which applies where. If they differ, it explains why.

**Research Evidence** — retrieves relevant abstracts from PubMed and summarises what the published literature actually says, prioritising African studies where available.

**Conflict or Agreement** — explicitly states whether the research supports or contradicts the guidelines, and explains the source of any disagreement. This section always appears even when there is no conflict, so you know the system checked.

**Clinical Implication** — lays out the practical considerations for both resource-rich and resource-limited settings, with a clear statement that Nigerian clinicians should follow WHO and FMOH guidelines rather than DHHS recommendations.

---

## Example questions it handles

- What is the recommended first-line ART regimen for treatment-naive adults in Nigeria?
- What are the drug interactions between rifampicin and dolutegravir in HIV/TB coinfection?
- What does the evidence say about long-acting injectable ART compared to daily oral regimens?
- Is TDF or TAF safer for patients with renal impairment?
- How should ART be managed during pregnancy?
- What does evidence say about PrEP effectiveness in women?
- What is the evidence that undetectable viral load prevents HIV transmission?
- How does depression affect ART adherence in people living with HIV?

---

## How it works

The system is built on a RAG (Retrieval-Augmented Generation) architecture — which means it does not rely on the language model's memory. Instead it searches a local database of trusted documents and hands the relevant evidence directly to the model with strict instructions to cite everything and never speculate beyond what the evidence supports.

**Data sources:**
- ~500 PubMed abstracts across HIV treatment, prevention, and TB coinfection topics
- 17 manually curated guideline sections from WHO 2021, Nigerian FMOH, DHHS, and CDC — with particular attention to Nigeria-specific protocols including PEPFAR implementing partners, viral load monitoring, PMTCT, and drug supply

**Retrieval:**
- Text chunks embedded using `sentence-transformers/all-MiniLM-L6-v2`
- Cosine similarity search in pure numpy — no LangChain, no ChromaDB, no SQLite schema conflicts
- Embeddings pre-computed and stored as a numpy file so startup takes seconds not minutes

**Generation:**
- Groq API with `openai/gpt-oss-20b`
- Structured system prompt enforces the four-section output format
- Thinking blocks stripped automatically if the model includes internal reasoning
- Retry loop handles API rate limits without crashing

**Conflict detection:**
- Scans the answer for conflict phrases and no-conflict phrases
- Only flags genuine conflicts — not just answers that contain the word "however"

---

## Tech stack

| Layer | Tool |
|-------|------|
| Frontend | Streamlit |
| LLM | Groq — GPT-OSS 20B |
| Embeddings | sentence-transformers |
| Vector search | numpy cosine similarity |
| Data | PubMed Entrez API + manual guideline curation |
| Hosting | Streamlit Community Cloud |

---

## Why no LangChain

The original version used LangChain, ChromaDB, and HuggingFace wrappers. Every deployment failed because of SQLite schema conflicts between the Colab environment and the server, deprecated package warnings, and a cold start time of 2 to 3 minutes that triggered Streamlit's CPU throttle.

The current version uses direct API calls, pure numpy for retrieval, and pre-computed embeddings. The requirements file went from 12 packages to 5. Cold start is under 15 seconds. No more schema errors.

---

## Limitations

- Guideline sections are manually curated snapshots — they do not update automatically when DHHS or WHO releases new guidance
- PubMed abstracts only — full text articles are not retrieved
- The system is for evidence research and education only. It is not a substitute for clinical judgment or patient-specific medical advice

---

## Background

This project was built by someone with an MPH background working at the intersection of public health, pharmacy, and data science. The Nigeria-specific focus came from a real frustration with tools that present US guidelines as if they are globally applicable — they are not, and the difference matters clinically.


---

## Running locally - How to run on your local computer

```bash
git clone https://github.com/kiks2022/HIV-Evidence-Assistant-LLM-RAG
cd HIV-Evidence-Assistant-LLM-RAG
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
streamlit run app.py
```

---

## Roadmap - What Next?

- **Hybrid retrieval** — add BM25 keyword search alongside the current semantic search so drug names and acronyms like TDF, DTG, and cabotegravir are retrieved more reliably
- **Reranking** — after the initial retrieval, pass the chunks through a cross-encoder model to re-score them by true relevance before sending to the LLM
- **Live guideline updates** — replace the static guideline JSON with a scheduled fetch from WHO, DHHS, and FMOH sources so the knowledge base stays current without manual updates
- **Drug interaction checker** — a companion tool that retrieves evidence-based interaction data for any combination of antiretrovirals and co-medications, with severity grading and clinical management recommendations
- **Evaluation harness** — a labeled set of 50 clinical questions with expected answers to measure retrieval precision and answer faithfulness, making improvement measurable rather than subjective
