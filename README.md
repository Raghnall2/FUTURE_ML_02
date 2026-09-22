# Support Ticket Classification System

An enterprise-ready Natural Language Processing (NLP) system designed to automatically classify, categorize, and route IT and enterprise service desk tickets into their appropriate operational departments.

---

## 🌍 Real-World Applications & Business Value

Modern organizations and enterprises handle thousands of internal and external support requests daily—ranging from hardware malfunctions and access requests to HR inquiries and software provisioning.

### The Problem
- **Manual Triage Bottlenecks:** Human support agents spend significant hours reading, sorting, and manually routing incoming tickets.
- **Routing Delays & Misdirection:** Incorrect manual routing causes tickets to bounce between departments, driving up resolution times and frustrating employees or customers.
- **SLA Breaches:** Critical issues (e.g., system access or hardware failures) can get buried under routine requests.

### The Solution & Business Impact
- **Instant Automatic Ticket Triage:** Classifies unstructured raw text into targeted departments within milliseconds upon ticket creation.
- **Reduced Mean Time to Resolution (MTTR):** Eliminates triage delay, instantly assigning tickets to the right subject matter experts.
- **Higher SLA Compliance:** Prioritizes workloads intelligently and prevents misrouted requests.
- **Scalable Support Operations:** Seamlessly absorbs spikes in ticket volume without increasing support headcount.

---

## 🎯 Target Categories & Taxonomies

The system categorizes support requests across **8 primary operational departments**:

1. **Access:** Permissions, account unlocks, and network login access.
2. **Administrative rights:** Elevated administrative privileges and local computer rights.
3. **Hardware:** Physical device issues, broken monitors, laptops, keyboards, and printers.
4. **HR Support:** Leave policies, employee benefits, payroll, and insurance inquiries.
5. **Internal Project:** Internal developer and team project tooling requests.
6. **Purchase:** Procurement of equipment, software licenses, and accessories.
7. **Storage:** Shared drives, disk quotas, cloud storage, and folder access.
8. **Miscellaneous:** General inquiries and unclassified service requests.

---

## ⚙️ How the Project Works (End-to-End Workflow)

The system is built as a complete, modular machine learning pipeline consisting of 5 core stages:

```
Raw Ticket Text ➔ Text Cleaning & Lemmatization ➔ TF-IDF Feature Extraction ➔ Multi-Class Classifier ➔ Department Routing & Prediction
```

### 1. Text Preprocessing & Normalization
Incoming support tickets are often noisy, containing email headers, URLs, code snippets, typos, and polite conversational filler. The cleaning engine standardizes text through:
- Removing emails, URLs, HTML tags, and special punctuation characters.
- Filtering out common conversational noise and domain-specific stopwords (e.g., *"hello"*, *"please"*, *"thanks"*).
- Tokenization and morphological normalization via **WordNet Lemmatization** to reduce words to their dictionary root forms (e.g., *"printers"* $\rightarrow$ *"printer"*, *"connecting"* $\rightarrow$ *"connect"*).

### 2. Feature Extraction (TF-IDF Vectorization)
The cleaned text is transformed into high-dimensional numerical vectors using **Term Frequency-Inverse Document Frequency (TF-IDF)**:
- Captures both individual terms and contextual pairs via **Unigram and Bigram** feature extraction (`ngram_range=(1, 2)`).
- Applies **sublinear term frequency scaling** to dampen the effect of repetitive words.
- Filters extreme outliers with document frequency thresholds (`min_df=2`, `max_df=0.95`).

### 3. Model Training & Class Balancing
The vector representations are fed into a high-performance **Linear Support Vector Classifier (`LinearSVC`)** (with **Logistic Regression** support):
- **Class-Weight Balancing:** Automatically adjusts loss penalties inversely proportional to class frequencies, ensuring minority classes (e.g., *Administrative rights*, *Internal Project*) receive balanced attention alongside large classes (e.g., *Hardware*, *HR Support*).
- **Stratified Splitting:** Preserves the exact class proportion across training and test splits.

### 4. Evaluation & Diagnostics
The system computes multi-metric evaluation to validate generalization:
- **Overall Accuracy:** ~85%
- **Macro & Weighted Precision, Recall, and F1-Score:** ~85%
- Automated generation of normalized **Confusion Matrices** and **Per-Class F1-Score** visualizations saved to reports.

### 5. Production-Ready Inference Pipeline
The trained vectorizer, model, and label encoders are packaged into an end-to-end `TicketClassificationPipeline` object that accepts raw, unseen ticket strings directly and outputs the destination category in real time.

---

## 🚀 How to Run the Project

### 1. Data Cleaning & Preprocessing
Cleans the raw dataset and prepares processed data:
```bash
python -m src.data.clean_text
```

### 2. Train and Save the End-to-End Pipeline
Fits the TF-IDF vectorizer and classifier, evaluates the test split, saves model artifacts, and runs sample inferences:
```bash
python -m src.pipeline
```

### 3. Run Comprehensive Model Evaluation
Computes all 7 configured metrics (Accuracy, Macro/Weighted Precision, Recall, F1), generates classification reports, and exports confusion matrix plots:
```bash
python -m src.evaluation.evaluate
```

---

## 💡 Example Real-World Prediction

| Incoming Ticket | Predicted Department |
| :--- | :---: |
| *"Need access to the finance department shared drive folder"* | **Storage** |
| *"My laptop screen is flickering and power cable is broken"* | **Hardware** |
| *"Inquiry regarding employee health insurance and annual leave policy"* | **HR Support** |
| *"Please grant local admin permissions to install Docker Desktop"* | **Administrative rights** |

---

## 📚 Reference Books & Further Reading

1. **Speech and Language Processing (3rd Edition Draft)**
   *Authors:* Daniel Jurafsky & James H. Martin (Stanford University)  
   *Relevance:* Foundational text covering tokenization, TF-IDF vector representations, multi-class classification, and precision/recall/F1 evaluation dynamics.

2. **Applied Text Analysis with Python: Enabling Language-Aware Data Products with Machine Learning**
   *Authors:* Benjamin Bengfort, Rebecca Bilbro, & Tony Ojeda (O'Reilly Media)  
   *Relevance:* Industry standard guide for building scalable NLP pipelines, corpus vectorization, model diagnosis, and text preprocessing workflows using Python and scikit-learn.

3. **Natural Language Processing with Python: Analyzing Text with the Natural Language Toolkit**
   *Authors:* Steven Bird, Ewan Klein, & Edward Loper (O'Reilly Media / NLTK Project)  
   *Relevance:* Comprehensive resource detailing text normalization, tokenization, stopword filtering, and morphological lemmatization.

4. **Introduction to Information Retrieval**
   *Authors:* Christopher D. Manning, Prabhakar Raghavan, & Hinrich Schütze (Cambridge University Press)  
   *Relevance:* Detailed mathematical and theoretical underpinnings of TF-IDF weighting, vector space models, and document categorization.

5. **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (3rd Edition)**
   *Author:* Aurélien Géron (O'Reilly Media)  
   *Relevance:* Practical engineering guide for end-to-end scikit-learn pipelines, model serialization (`joblib`), classification metrics, and handling class imbalances.

