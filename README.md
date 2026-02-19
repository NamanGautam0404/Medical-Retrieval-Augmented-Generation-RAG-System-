# AI Chatbot with Disease Prediction using Neo4j and RAG



<div align="center">
  
  ![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
  ![Neo4j](https://img.shields.io/badge/Neo4j-Graph%20Database-green.svg)
  ![Gradio](https://img.shields.io/badge/Gradio-UI%20Interface-orange.svg)
  ![Transformers](https://img.shields.io/badge/Transformers-NLP-yellow.svg)
  ![Accuracy](https://img.shields.io/badge/Accuracy-97%25-brightgreen.svg)
  
  <h3>An intelligent AI-powered medical assistant combining graph databases, machine learning, and NLP</h3>
  <img width="836" height="519" alt="image" src="https://github.com/user-attachments/assets/91a110b0-32f4-40f4-8a87-240cac388abd" />

  <img width="824" height="430" alt="image" src="https://github.com/user-attachments/assets/79017172-28cf-4e8a-9f6c-133ec626e360" />

  
  <p>
 
  </p>
</div>

---

## 📋 Table of Contents
- [Introduction](#-introduction)
- [Objectives](#-objectives)
- [System Architecture](#-system-architecture)
- [Project Components](#-project-components)
- [Tech Stack](#-tech-stack)
- [NLP Models Used](#-nlp-models-used)
- [Installation](#-installation)
- [Usage](#-usage)
- [Output Examples](#-output-examples)
- [Results](#-results)
- [Challenges Faced](#-challenges-faced)
- [Future Scope](#-future-scope)
- [Contributors](#-contributors)

---

## 📖 Introduction

This project implements an AI-powered medical assistant that combines graph-based retrieval (Neo4j), machine learning-based disease prediction (97% accuracy), and transformer-based NLP models. The system allows users to ask medical queries or provide symptoms to receive contextual medical information and predicted diseases.

**Key Features:**
- 🤖 Medical chatbot using Neo4j graph database
- 🔬 Disease prediction with 97% accuracy
- 📸 Skin disease detection using CNN
- 🕸️ Graph-based knowledge retrieval
- 🎨 User-friendly Gradio interface

---

## 🎯 Objectives

| Objective | Description |
|-----------|-------------|
| **Chatbot Development** | Answer medical queries using Neo4j graph database |
| **Disease Prediction** | ML model to predict diseases from symptoms (97% accuracy) |
| **User Interface** | Create intuitive Gradio interface |
| **NLP Integration** | Combine NLP, graph querying, and AI prediction |

---

## 🏗️ System Architecture
<img width="669" height="567" alt="image" src="https://github.com/user-attachments/assets/44c39210-eefc-4c2d-a17f-a964372bb3a0" />


**Components:**
1. **User Input**: Natural language queries or symptom lists
2. **Query Processing**: NLP classification
3. **Graph Retrieval**: Neo4j queries for medical data
4. **Disease Prediction**: ML model for symptom analysis
5. **Response Output**: Combined results via Gradio

---

## 🔧 Project Components

### 4.1 Neo4j Graph Database

**Purpose**: Store and retrieve structured medical data (diseases, symptoms, treatments)

**Implementation:**
- Nodes: Diseases, symptoms, treatments, medications
- Edges: Relationships like "HAS_ANSWER", "HAS_QUESTION"
- Hybrid search using LangChain for semantic + keyword querying

**Advantages:**
- Efficient query resolution for complex relationships
- Stores medical entity relationships effectively
- Quick context retrieval for user queries

### 4.2 Disease Prediction Model

**Purpose**: Predict diseases from user symptoms

**Implementation:**
- **Data Source**: Symptom-disease dataset
- **Feature Engineering**: TF-IDF vectorization
- **Model**: Voting Classifier (Random Forest + Logistic Regression)
- **Accuracy**: 97% on validation data

**Output:**
<img width="859" height="179" alt="image" src="https://github.com/user-attachments/assets/511d769e-217f-4505-b293-b267c6ed1b0b" />


### 4.3 Gradio Interface

**Features:**
- Input fields for questions/symptoms
- Real-time interaction
- Image upload for skin disease detection
- Conversation history tracking
- Clean, intuitive design

### 4.4 NLP Query Answering

Multiple pre-trained models integrated for medical question answering.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Graph Database | Neo4j 5.14+ |
| ML Framework | Scikit-learn 1.3+ |
| NLP Framework | Transformers 4.35+ |
| Deep Learning | TensorFlow/Keras |
| Web Interface | Gradio 4.0+ |
| Graph Integration | LangChain 0.0.340 |
| Embeddings | Sentence-Transformers |

---

## 🤖 NLP Models Used

| Model | Type | Parameters | Purpose |
|-------|------|------------|---------|
| **EleutherAI/gpt-neo-2.7B** | GPT-Neo | 2.7B | General conversation |
| **jianghc/medical_chatbot** | T5-small | 60M | Medical QA |
| **google/gemma-2-9b-it** | Gemma | 2.9B | Medical NLP |
| **DistilBERT** | BERT variant | 66M | Question answering |
| **all-MiniLM-L6-v2** | Sentence-BERT | 22M | Embeddings |

---
