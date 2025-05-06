import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate

# === Paths ===
flan_model_id = "google/flan-t5-base"
embedding_model_id = "sentence-transformers/all-MiniLM-L6-v2"
flan_local_path = "C:/Agent/models/flan-t5-base"
embedding_local_path = "./models/all-MiniLM-L6-v2"

# === Check model is already downloaded ===
def is_model_downloaded(path):
    return os.path.exists(os.path.join(path, "model.safetensors"))

# === Load or Download FLAN-T5 ===
if not is_model_downloaded(flan_local_path):
    print("Downloading FLAN-T5-Base with safetensors...")
    tokenizer = AutoTokenizer.from_pretrained(flan_model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        flan_model_id,
        trust_remote_code=True,
        use_safetensors=True
    )
    os.makedirs(flan_local_path, exist_ok=True)
    tokenizer.save_pretrained(flan_local_path)
    model.save_pretrained(flan_local_path)
else:
    print("Loading FLAN-T5-Base from local path...")
    tokenizer = AutoTokenizer.from_pretrained(flan_local_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        flan_local_path,
        trust_remote_code=True,
        use_safetensors=True
    )

# === LLM pipeline ===
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)
llm = HuggingFacePipeline(pipeline=pipe)

# === Embeddings ===
if not os.path.exists(embedding_local_path):
    print("Downloading Sentence Transformer Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_id)
else:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_local_path)

# === FAQs ===
faq_pairs = [
    ("How many credits do I need to graduate?", "You need 120 credits to graduate."),
    ("Can I drop a course after the deadline?", "You need special permission to drop after the deadline.")
]

# === Vector store setup ===
texts = [a for _, a in faq_pairs]
metadatas = [{"source": "FAQ"} for _ in faq_pairs]
db = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
retriever = db.as_retriever(search_kwargs={"k": 1})

# === Prompt Template ===
custom_prompt = PromptTemplate(
    input_variables=["summaries", "question"],
    template=(
        "You are a helpful assistant at a university.\n"
        "Based only on the following information, answer the student's question in full sentences.\n"
        "If the answer is not present, reply: 'I don't know based on the provided information.'\n\n"
        "Information:\n{summaries}\n\n"
        "Student's Question: {question}\n\n"
        "Answer:"
    )
)

# === QA Chain ===
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt}
)

# === Query Example ===
user_query = "How many credits do I need to graduate?"

docs = retriever.invoke(user_query)
print("\nRetrieved Document(s):")
for doc in docs:
    print(f"- Content: {doc.page_content}")
    print(f"  Metadata: {doc.metadata}")

response = qa_chain.invoke({"question": user_query})
print("\nFinal Answer:", response["answer"])
print("Sources:", response.get("sources", "FAQ"))
print("\nEnd of Program")
