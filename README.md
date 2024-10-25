This repository contains a Python script that uses Sentence-BERT (SBERT), a modification of BERT tailored for sentence embeddings, to perform semantic search and categorize questions based on their meaning. This tool is designed to help categorize user questions and provide follow-up suggestions in domains like mortgage advisory. By embedding questions and comparing them with predefined sets, it assigns questions to the best-fitting category and suggests contextually relevant follow-ups.

## Features
- Semantic Categorization of Questions: Categorizes user questions into four categories: Purchase, Refinance, Loan Options, and Others, based on semantic similarity.
- Dynamic Follow-Up Suggestions: Recommends follow-up questions based on the matched category, filtering out suggestions too similar to previously asked questions.
- Pre-Trained Model Flexibility: Uses SentenceTransformer with SBERT-large-nli-v2 for sentence embeddings, with options to switch to alternative models like SGPT-5.8B-weightedmean-msmarco-specb-bitfit for specific use cases.

## How It Works

1. Input Question Embedding:

- Prompts the user to enter a question.
- Generates an embedding for the question using SBERT, which captures its meaning in vector form.

2. Categorization via Cosine Similarity:

- Compares the input question's embedding with average embeddings from predefined question sets (e.g., Purchase, Refinance).
- Uses cosine similarity to identify the best-matching category, flagging questions that don’t closely match any category.

3. Filtered Follow-Up Recommendations:

- Suggests up to three follow-up questions from the matched category.
- Avoids redundancy by filtering out any suggestions with high similarity to previously asked questions.
