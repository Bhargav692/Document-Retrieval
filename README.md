# Document-Retrieval
Overview
Document retrieval for question answering (QA) is a fundamental task in modern natural language processing (NLP). This project aims to develop a robust pipeline that integrates document retrieval and question answering mechanisms, focusing on addressing real-world retrieval challenges, particularly low retrieval precision. By leveraging advanced retrieval techniques and QA models, the system strives to provide accurate answers to user queries, even under suboptimal retrieval conditions.

Features
Pipeline Integration
A modular design with separate document retrieval and question answering components.
Advanced Retrieval Techniques
Candidate Generation: Uses sparse (e.g., BM25) and dense retrieval methods (e.g., DPR, Sentence-BERT).
Re-ranking: Incorporates transformer-based models (e.g., BERT, T5) for refining retrieved results.
Resilient QA Model
Fine-tuned QA models extract accurate answers from retrieved documents, even in noisy retrieval conditions.
Scalability
Supports massive datasets using efficient Approximate Nearest Neighbor (ANN) libraries like FAISS.
Error Handling for Ambiguous Queries
Implements query rewriting/paraphrasing to mitigate ambiguity in user queries.

Objectives
Enhance QA Resilience: Design models capable of robust performance despite low-precision retrieval.
Optimize Retrieval Precision: Experiment with hybrid retrieval and re-ranking methods.
Performance Analysis: Evaluate the system with metrics like precision, recall, EM, and F1 score.

Pipeline Architecture
1. Document Retrieval Module
Candidate Generation: Fetches a broad set of documents using sparse/dense retrieval methods.
Re-ranking: Prioritizes the most relevant documents using transformer models.
2. Question Answering Module
Fine-tuned QA model (e.g., BERT-based) extracts specific answers from the top-ranked documents.

Dataset
The project uses SQuAD 2.0 for training and evaluation:
SQuAD 2.0 provides context-document pairs with corresponding queries and ground-truth answers, including unanswerable questions to test model robustness.

Evaluation Metrics
Document Retrieval
Precision@k: Proportion of relevant documents in the top k retrieved documents.
Recall@k: Fraction of all relevant documents retrieved in the top k.
Question Answering
Exact Match (EM): Fraction of answers that match the ground truth exactly.
F1 Score: Harmonic mean of precision and recall for answer spans.

Key Challenges and Solutions
1. Low Retrieval Precision
Challenge: Irrelevant or noisy documents are included in retrieval outputs.
Solution:
Use hybrid retrieval (dense + sparse).
Train QA systems to filter noise and perform robustly.
2. Scalability for Large Corpora
Challenge: Efficient retrieval across massive datasets.
Solution:
Leverage ANN libraries like FAISS for scalable retrieval.
3. Ambiguous Queries
Challenge: Ambiguity in queries results in irrelevant retrieval results.
Solution:
Implement query rewriting/paraphrasing to clarify user intent using language models.

Results
QA model accuracy was a high accuracy with top-10 (71.48%); in top-1 a low
accuray (43.22%) EM on SQuAD 2.0 when fed with retrieved documents when we
used TF-IDF Vectorizer.
QA model accuracy was a good accuracy with top-10 (12.15%); in top-1 a low
accuray (3.07%) EM on SQuAD 2.0 when fed with retrieved documents when we
used Word2Vec / Embedding.

Installation
Download SQuAD 2.0 from official site and place it in the data/ folder.


Usage
Train the QA Model
python train_qa_model.py --dataset data/squad2.0.json --output model/qa_model  
Run the Retrieval-Augmented QA Pipeline
python qa_pipeline.py --query "What is natural language processing?"  
Evaluate the System
python evaluate.py --model model/qa_model --dataset data/squad2.0.json  

Future Work
Enhance retrieval precision with multi-stage re-ranking.
Explore other datasets like Natural Questions and TriviaQA.
Investigate generative QA models for improved answer generation.

Acknowledgments
This project was inspired by challenges in Retrieval-Augmented Generation (RAG) systems and leverages open-source tools like Hugging Face Transformers and FAISS.

