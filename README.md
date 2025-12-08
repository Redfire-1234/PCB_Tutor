Class 12 PCB MCQ Generator (Biology • Chemistry • Physics) 

Live Demo  
Try the app here →  
[Class 12 PCB MCQ Generator (HuggingFace Space)](https://huggingface.co/spaces/Redfire-1234/PCB_Tutor)  

References:  

- [PyTorch](https://pytorch.org/)  
- [Hugging Face — Transformers](https://huggingface.co/docs/transformers)  
- [Flask](https://flask.palletsprojects.com/)  
- [Docker](https://www.docker.com/)  
- [Sentence-Transformers](https://www.sbert.net/)  
- [FAISS](https://faiss.ai/)  
- [NumPy](https://numpy.org/) / [Pandas](https://pandas.pydata.org/)  


Project Overview:  
This project generates Class 12 PCB MCQs (Biology, Chemistry, Physics) using a RAG-based system built with FAISS, Sentence Transformers, and Qwen 2.5 LLM.  
A Dockerized Flask app serves the model, allowing users to enter any topic from the NCERT syllabus and instantly receive 5 MCQs with answers and explanations.  
The system retrieves subject-specific textbook content from a vector database, ensuring accurate, context-based MCQ generation.  

Dataset / Knowledge Base:  
Extracted from NCERT Class 12 Biology, Chemistry, and Physics textbooks  
Content split into small chunks (~500 characters)  
Stored separately as:  
Biology vector index  
Chemistry vector index  
Physics vector index  

Labels / Subjects:  
The system supports 3 subjects:  
Biology  
Chemistry   
Physics  
Each subject has its own FAISS index.  

Preprocessing:  
Text cleaning and chunking  
Embeddings using all-MiniLM-L6-v2  
FAISS similarity search  
Query embedding + top-5 retrieval before generation  

Features:  
Model:  
Qwen 2.5 (1.5B Instruct) for MCQ generation  
Low temperature (0.15) for accuracy  
Generates:  
5 questions  
4 options  
Correct answer  
Short explanation  

Retrieval (RAG):  
Retrieves relevant textbook content using FAISS  
Ensures MCQs are based only on NCERT topics  

Flask + Docker Web App:  
Topic sent via POST request  
App performs:  
     Embed query  
     Retrieve chunks  
     Generate MCQs  
Returns result as:  
     JSON  
     Or displayed in UI  
Fully containerized using Docker for stable deployment  

Deployment (HuggingFace Spaces — Docker):  
HuggingFace automatically builds container using Dockerfile  
No Streamlit/Gradio required  
Public URL provided instantly  
Easy to update by pushing new commits  

Installation:  
Required Libraries:  
torch  
transformers  
sentence-transformers  
faiss-cpu  
numpy  
flask  

Saved Models / Files:  
FAISS indexes:  
     faiss_bio.bin  
     faiss_chem.bin  
     faiss_phy.bin  
Text chunks:   
     bio_chunks.pkl  
     chem_chunks.pkl  
     phy_chunks.pkl  
Qwen model downloaded automatically at runtime (or can be cached)  

Results:  
Generates 5 high-quality MCQs  
Displays:  
     Question  
     Options  
     Correct answer  
     Explanation  
UI output resembles:  
Q1. Which cell organelle is known as the powerhouse of the cell?  
A) Nucleus  
B) Mitochondria  
C) Ribosome  
D) Golgi complex  
Answer: B — Generates ATP by oxidative phosphorylation  
