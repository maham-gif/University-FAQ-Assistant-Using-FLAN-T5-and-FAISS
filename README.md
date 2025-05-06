# University-FAQ-Assistant-Using-FLAN-T5-and-FAISS
Description
This application uses a combination of advanced natural language processing (NLP) models and vector search techniques to build an intelligent FAQ assistant for a university setting. It leverages the FLAN-T5 language model for question answering and SentenceTransformers for creating embeddings for vector search.

The system retrieves relevant information from a predefined set of FAQ pairs, processes a user's query, and generates the best possible response, incorporating both the information stored in the database and the generative capabilities of the FLAN-T5 model.

The setup uses FAISS for fast vector retrieval and Hugging Face's transformers to handle the text generation and embeddings.

Features
FLAN-T5: A fine-tuned language model that generates human-like responses to queries based on a given context.

Sentence Embeddings: Uses SentenceTransformers to convert FAQ text into vector space for efficient retrieval.

Vector Search: Leverages FAISS for quick and efficient retrieval of relevant FAQ data based on user queries.

Custom Prompt: Custom-built prompt template ensures that responses adhere to the university-specific context.

Data-driven Responses: The assistant generates answers based on provided FAQ pairs and retrieves the most relevant information to answer user questions.

Local Model Handling: Checks if the models are downloaded or not, ensuring that the application works offline with locally stored models once they're set up.

Technologies Used
Python 3

Hugging Face Transformers (FLAN-T5 and SentenceTransformers)

LangChain (for integrating pipelines, embeddings, and vector stores)

FAISS (for fast vector retrieval)

TensorFlow/PyTorch (required for FLAN-T5 and SentenceTransformers models)

OS (for file handling and local model management)

How the Application Works
Model Download and Setup:
The application first checks if the FLAN-T5 and SentenceTransformer models are already downloaded. If not, it downloads them and stores them locally. This allows the application to function offline after initial setup.

Question Answering Pipeline:
The core of the assistant is based on the FLAN-T5 model, which is used to generate human-like responses to user queries. The input question is passed to the model, and it generates a response based on the information it has in context. The context comes from a database of FAQs that are embedded into vector space using SentenceTransformers.

Vector Search with FAISS:
The FAQ pairs are converted into vectors, and the application uses FAISS for quick retrieval of the most relevant FAQ based on a user's query. The retrieved FAQ is then processed and passed into the FLAN-T5 model to generate the final response.

Custom Prompt:
A custom prompt template is used to ensure that responses are both contextually accurate and appropriately framed according to university-specific requirements.

Final Answer Generation:
The question is processed, the relevant FAQ pair is retrieved, and the answer is generated using the FLAN-T5 model. If no relevant information is found, the assistant replies, "I don't know based on the provided information."

Project Structure
faq_assistant.py: Main file containing the application logic

models/: Directory for storing downloaded models (e.g., FLAN-T5 and SentenceTransformers)

faiss_db/: Directory where the FAISS vector database is stored for fast retrieval

faq_pairs: Predefined list of FAQ pairs used to train the assistant

How to Run the Application
Install Dependencies:
Ensure Python 3 is installed, then install the necessary packages by running:

bash
Copy
Edit
pip install transformers langchain faiss-cpu sentence-transformers
Run the Application:

Navigate to the directory containing faq_assistant.py and run:

bash
Copy
Edit
python faq_assistant.py
Check Model Setup:

The application will check if the models are already downloaded. If not, it will automatically download FLAN-T5 and SentenceTransformers models. Ensure you have internet access for the first time setup.

Example Query:

After starting the application, you can query the assistant with a question such as:
"How many credits do I need to graduate?"

The assistant will process the query, retrieve the relevant FAQ, and provide an answer based on the given context.

Example Output
Upon querying, for example, "How many credits do I need to graduate?", the application will retrieve the relevant FAQ entry and output:

plaintext
Copy
Edit
Retrieved Document(s):
- Content: You need 120 credits to graduate.
  Metadata: {'source': 'FAQ'}

Final Answer: You need 120 credits to graduate.
Sources: FAQ
End of Program
FAQ Pairs Example
You can customize the faq_pairs list with new question-answer pairs. For example:

python
Copy
Edit
faq_pairs = [
    ("How many credits do I need to graduate?", "You need 120 credits to graduate."),
    ("Can I drop a course after the deadline?", "You need special permission to drop after the deadline."),
]
Future Enhancements
Expand FAQ Database: Add more FAQ pairs to the vector store to improve the assistant's range of knowledge.

Improve Retrieval: Experiment with different retrieval methods or vector databases like Pinecone or Weaviate.

Multilingual Support: Integrate multilingual capabilities by using different FLAN models.

Web Interface: Build a web interface for user interaction with the assistant.

Credits
This project integrates various cutting-edge AI technologies, including Hugging Face's FLAN-T5 model, SentenceTransformers, LangChain, and FAISS, to create a high-performance, context-aware FAQ assistant for educational institutions.
