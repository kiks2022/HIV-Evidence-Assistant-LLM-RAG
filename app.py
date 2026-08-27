import os
import re
import time
import json
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HIV Clinical Evidence Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0a0e17; color: #e2e8f0; }
    .stTextArea textarea { background-color: #111827 !important; color: #e2e8f0 !important; border: 1px solid #1e2d40 !important; border-radius: 8px !important; }
    .stButton button { background-color: #00d4aa !important; color: #0a0e17 !important; font-weight: 600 !important; border: none !important; border-radius: 6px !important; }
    .stButton button:hover { background-color: #00b894 !important; }
    section[data-testid="stSidebar"] { background-color: #111827 !important; }
    .header-box { background:#111827; border:1px solid #1e2d40; border-radius:10px; padding:18px 24px; margin-bottom:24px; }
    .header-title { font-size:16px; font-weight:600; color:#e2e8f0; }
    .header-sub { font-size:11px; color:#64748b; font-family:monospace; margin-top:4px; }
    .conflict-box { padding:14px 18px; border-radius:8px; margin-bottom:16px; border:1px solid; }
    .conflict-yes { background:rgba(248,113,113,0.08); border-color:rgba(248,113,113,0.4); }
    .conflict-no  { background:rgba(0,212,170,0.06);  border-color:rgba(0,212,170,0.3); }
    .conflict-title-yes { color:#f87171; font-weight:700; font-size:13px; font-family:monospace; }
    .conflict-title-no  { color:#00d4aa; font-weight:700; font-size:13px; font-family:monospace; }
    .conflict-msg { color:#94a3b8; font-size:12px; margin-top:4px; }
    .answer-box { background:#111827; border:1px solid #1e2d40; border-radius:10px; padding:20px 24px; margin-bottom:16px; line-height:1.8; font-size:14px; }
    .section-heading { color:#00d4aa; font-weight:700; font-size:11px; letter-spacing:0.1em; text-transform:uppercase; font-family:monospace; margin-top:18px; margin-bottom:6px; padding-bottom:4px; border-bottom:1px solid #1e2d40; display:block; }
    .refs-box { background:#111827; border:1px solid #1e2d40; border-radius:10px; padding:18px 24px; margin-bottom:16px; }
    .refs-title { font-family:monospace; font-size:10px; letter-spacing:0.1em; text-transform:uppercase; color:#64748b; margin-bottom:14px; }
    .ref-group { font-size:12px; font-weight:600; margin-bottom:8px; margin-top:14px; }
    .ref-group-g { color:#00d4aa; }
    .ref-group-p { color:#0ea5e9; }
    .ref-card { border:1px solid #1e2d40; border-radius:7px; padding:10px 14px; margin-bottom:8px; font-size:12px; }
    .ref-card-g { border-left:3px solid #00d4aa; }
    .ref-card-p { border-left:3px solid #0ea5e9; }
    .ref-cit-g { color:#00d4aa; font-weight:600; margin-bottom:3px; }
    .ref-cit-p { color:#0ea5e9; font-weight:600; margin-bottom:3px; }
    .ref-sec { color:#94a3b8; font-size:11px; margin-bottom:3px; }
    .ref-url { color:#64748b; font-size:10px; font-family:monospace; word-break:break-all; }
    .disclaimer { margin-top:14px; padding-top:12px; border-top:1px solid #1e2d40; font-size:10px; color:#475569; }
    .stat-block { font-family:monospace; font-size:11px; color:#64748b; line-height:2; }
    .stat-val { color:#00d4aa; }
</style>
""", unsafe_allow_html=True)


# ── Vector search (no LangChain, no ChromaDB) ─────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_resources():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    with open("chunks.json") as f:
        data = json.load(f)
    texts     = [d["text"]     for d in data]
    metadatas = [d["metadata"] for d in data]
    embeddings = np.load("embeddings.npy")
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return model, embeddings, texts, metadatas, groq_client, len(texts)


def retrieve(question, model, embeddings, texts, metadatas, k=8):
    """Pure numpy cosine similarity search — no LangChain needed."""
    q_emb = model.encode([question], normalize_embeddings=True)[0]
    scores = embeddings @ q_emb  # cosine similarity since embeddings are normalized
    top_k  = np.argsort(scores)[::-1][:k]
    return [{"text": texts[i], "metadata": metadatas[i], "score": float(scores[i])} for i in top_k]


# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an HIV/AIDS clinical evidence research assistant supporting researchers, clinicians, pharmacists and MPH professionals globally with specific relevance to Nigeria and sub-Saharan Africa.

IMPORTANT CONTEXT:
- In Nigeria and most of sub-Saharan Africa the standard first-line ART regimen is TDF/3TC/DTG per WHO 2021 and Nigerian FMOH guidelines
- TAF-based regimens are preferred in US DHHS guidelines but TDF remains WHO preferred globally due to cost
- Always present BOTH WHO/Nigerian guidelines AND DHHS guidelines when relevant
- Explicitly state when recommendations differ between high-income and low-income country guidelines

Structure EVERY response using EXACTLY these four sections with these exact headings:

GUIDELINE POSITION:
Present WHO/Nigerian AND DHHS recommendations where they differ. Use citation tags like [WHO Guidelines 2021], [Nigeria FMOH Guidelines], [DHHS ARV Guidelines].

RESEARCH EVIDENCE:
What published research says. Reference studies by name and year. Prioritise African studies where available.

CONFLICT OR AGREEMENT:
State explicitly whether research supports or contradicts the guideline. Always appear even if no conflict.

CLINICAL IMPLICATION:
Considerations for both settings. Nigerian clinicians should follow WHO and FMOH not DHHS. End with disclaimer.

Citation rules:
- Use descriptive tags like [WHO Guidelines 2021], [Nigeria FMOH Guidelines], [DHHS ARV Guidelines]
- Never use Source 1 or Source 2
- Never invent citations
- Start immediately with GUIDELINE POSITION: no preamble"""


# ── Citation tag parser ────────────────────────────────────────────────────────
def parse_citation_tag(source_str, doc_type):
    if doc_type == "guideline":
        if "WHO"        in source_str: return "WHO Guidelines 2021"
        if "Nigeria"    in source_str or "NACA" in source_str: return "Nigeria FMOH Guidelines"
        if "Perinatal"  in source_str or "Pregnant" in source_str: return "DHHS Perinatal Guidelines"
        if "Opportunist" in source_str or "OI" in source_str: return "DHHS OI Guidelines"
        if "PrEP"       in source_str and "CDC" in source_str: return "CDC PrEP Guidelines"
        if "CDC"        in source_str: return "CDC Guidelines"
        return "DHHS ARV Guidelines"
    year_match = re.search(r"\((\d{4})\)", source_str)
    return "PubMed " + (year_match.group(1) if year_match else "n.d.")


# ── RAG function ───────────────────────────────────────────────────────────────
def hiv_rag(question, model, embeddings, texts, metadatas, groq_client):
    docs = retrieve(question, model, embeddings, texts, metadatas, k=8)

    guideline_docs = [d for d in docs if d["metadata"].get("type") == "guideline"][:3]
    pubmed_docs    = [d for d in docs if d["metadata"].get("type") == "pubmed"][:3]

    if not guideline_docs and not pubmed_docs:
        return {"answer": "No relevant evidence retrieved.", "sources": [], "has_conflict": False}

    guideline_citations = [parse_citation_tag(d["metadata"].get("source", ""), "guideline") for d in guideline_docs]
    pubmed_citations    = [parse_citation_tag(d["metadata"].get("source", ""), "pubmed")    for d in pubmed_docs]

    guideline_context = "GUIDELINE EVIDENCE:\n\n"
    for doc, tag in zip(guideline_docs, guideline_citations):
        guideline_context += "[" + tag + "]\nURL: " + doc["metadata"].get("url", "") + "\n\n" + doc["text"] + "\n\n"

    pubmed_context = "RESEARCH EVIDENCE (PubMed):\n\n"
    for doc, tag in zip(pubmed_docs, pubmed_citations):
        pubmed_context += "[" + tag + "]\nURL: " + doc["metadata"].get("url", "") + "\n\n" + doc["text"] + "\n\n"

    full_prompt = (
        SYSTEM_PROMPT + "\n\n"
        "Use the retrieved evidence below to answer the clinical question.\n\n" +
        guideline_context + pubmed_context +
        "\nQUESTION:\n" + question + "\n\n"
        "Start immediately with GUIDELINE POSITION: — no preamble, no thinking. "
        "Follow all four sections exactly."
    )

    for attempt in range(3):
        try:
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": full_prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            answer = response.choices[0].message.content
            answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
            if "GUIDELINE POSITION:" in answer:
                answer = answer[answer.index("GUIDELINE POSITION:"):]
            break
        except Exception as e:
            if attempt == 2: raise e
            time.sleep((attempt + 1) * 3)

    conflict_phrases    = ["conflict between","contradicts","disagrees with","inconsistent with","in conflict","directly contradicts","evidence conflicts","no consensus","debate remains"]
    no_conflict_phrases = ["no conflict","in agreement","supports the guideline","aligns with","consistent with the guideline","no direct conflict","does not conflict"]
    al = answer.lower()
    has_conflict = any(p in al for p in conflict_phrases) and not any(p in al for p in no_conflict_phrases)

    sources   = []
    seen_urls = set()
    for doc, tag in zip(guideline_docs, guideline_citations):
        url = doc["metadata"].get("url", "")
        if url not in seen_urls:
            seen_urls.add(url)
            sources.append({"citation": tag, "type": "guideline",
                            "source": doc["metadata"].get("source",""),
                            "url": url, "section": doc["metadata"].get("section","")})
    for doc, tag in zip(pubmed_docs, pubmed_citations):
        url = doc["metadata"].get("url", "")
        if url not in seen_urls:
            seen_urls.add(url)
            sources.append({"citation": tag, "type": "pubmed",
                            "source": doc["metadata"].get("source",""),
                            "url": url, "section": ""})

    return {"answer": answer, "sources": sources, "has_conflict": has_conflict}


# ── Format answer HTML ─────────────────────────────────────────────────────────
def format_answer(text):
    for section in ["GUIDELINE POSITION:", "RESEARCH EVIDENCE:", "CONFLICT OR AGREEMENT:", "CLINICAL IMPLICATION:"]:
        text = text.replace(section,
            f'</div><span class="section-heading">{section}</span><div class="section-content">')
    text = '<div class="section-content">' + text + '</div>'
    text = text.replace('<div class="section-content"></div>', '')
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    return text


# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
  <div class="header-title">🧬 HIV Clinical Evidence Assistant</div>
  <div class="header-sub">PubMed + WHO 2021 + DHHS + Nigeria FMOH &nbsp;·&nbsp; Powered by Groq</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Example Queries")
    examples = [
        "What is the recommended first-line ART regimen for treatment-naive adults in Nigeria?",
        "What are the drug interactions between rifampicin and dolutegravir in HIV/TB coinfection?",
        "What does evidence say about long-acting injectable ART vs daily oral regimens?",
        "What does evidence say about PrEP effectiveness in women?",
        "How should ART be managed during pregnancy?",
        "What prophylaxis is recommended for HIV patients with low CD4 count?",
        "What is the evidence that undetectable viral load prevents HIV transmission?",
        "Is TDF or TAF safer for patients with renal impairment?",
        "How does depression affect ART adherence in people living with HIV?",
    ]
    for ex in examples:
        label = ex[:58] + "..." if len(ex) > 58 else ex
        if st.button(label, key=ex, use_container_width=True):
            st.session_state["selected_question"] = ex
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div class="stat-block">
    Model &nbsp;&nbsp;&nbsp; <span class="stat-val">GPT-OSS 20B</span><br>
    Provider &nbsp; <span class="stat-val">Groq</span><br>
    Sources &nbsp;&nbsp; <span class="stat-val">PubMed + Guidelines</span><br>
    Context &nbsp;&nbsp; <span class="stat-val">WHO · DHHS · Nigeria</span>
    </div>
    """, unsafe_allow_html=True)

# Load resources
try:
    model, embeddings, texts, metadatas, groq_client, n_docs = load_resources()
    st.sidebar.markdown(f'<div class="stat-block"><span class="stat-val">✅ {n_docs} chunks indexed</span></div>', unsafe_allow_html=True)
except Exception as e:
    st.error(f"Failed to load: {e}")
    st.stop()

# Handle selected question from sidebar
selected = st.session_state.get("selected_question", "")

question = st.text_area(
    "Clinical Evidence Question",
    value=selected,
    placeholder="e.g. What are the drug interactions between rifampicin and dolutegravir in HIV/TB coinfection?",
    height=100,
    key="q_input"
)

col1, _ = st.columns([1, 5])
with col1:
    run = st.button("Analyse Evidence", type="primary", use_container_width=True)

# Auto-run when question selected from sidebar
if selected and not run:
    run = True
    st.session_state.pop("selected_question", None)

if run and question.strip():
    with st.spinner("Retrieving evidence and generating analysis..."):
        try:
            result = hiv_rag(question.strip(), model, embeddings, texts, metadatas, groq_client)

            # Conflict banner
            if result["has_conflict"]:
                st.markdown("""<div class="conflict-box conflict-yes">
                  <div class="conflict-title-yes">⚠️ CONFLICT DETECTED</div>
                  <div class="conflict-msg">Guidelines and research present differing positions. Review both carefully before clinical decision making.</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="conflict-box conflict-no">
                  <div class="conflict-title-no">✅ NO CONFLICT</div>
                  <div class="conflict-msg">Guidelines and research are in agreement on this topic.</div>
                </div>""", unsafe_allow_html=True)

            # Answer
            st.markdown(f'<div class="answer-box">{format_answer(result["answer"])}</div>',
                        unsafe_allow_html=True)

            # References
            guidelines = [s for s in result["sources"] if s["type"] == "guideline"]
            pubmed     = [s for s in result["sources"] if s["type"] == "pubmed"]

            refs = '<div class="refs-box"><div class="refs-title">References</div>'
            refs += '<div class="ref-group ref-group-g">Clinical Practice Guidelines</div>'
            if guidelines:
                for s in guidelines:
                    refs += f'''<div class="ref-card ref-card-g">
                        <div class="ref-cit-g">[{s["citation"]}]</div>
                        {f'<div class="ref-sec">{s["section"]}</div>' if s.get("section") else ""}
                        {f'<a class="ref-url" href="{s["url"]}" target="_blank">{s["url"]}</a>' if s.get("url") else ""}
                    </div>'''
            else:
                refs += '<div style="color:#64748b;font-size:12px;margin-bottom:10px;">None retrieved</div>'

            refs += '<div class="ref-group ref-group-p">Primary Literature (PubMed)</div>'
            for s in pubmed:
                refs += f'''<div class="ref-card ref-card-p">
                    <div class="ref-cit-p">[{s["citation"]}]</div>
                    <div class="ref-sec">{s["source"]}</div>
                    {f'<a class="ref-url" href="{s["url"]}" target="_blank">{s["url"]}</a>' if s.get("url") else ""}
                </div>'''

            refs += '<div class="disclaimer">⚠️ For clinical evidence research and education only. Not a substitute for professional clinical judgment or official guidelines.</div>'
            refs += '</div>'
            st.markdown(refs, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")

elif run:
    st.warning("Please enter a clinical question.")
