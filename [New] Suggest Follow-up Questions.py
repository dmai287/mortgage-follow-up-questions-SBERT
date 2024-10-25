# SGPT:GPTSentenceEmbeddingsforSemantic Search

# Usage instruction: https://github.com/Muennighoff/sgpt
# Evaluation results: https://arxiv.org/abs/2202.08904

# pip install transformers torch
import torch
import random
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('Muennighoff/SBERT-large-nli-v2')
# model = SentenceTransformer('Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit')

input_question = input("Enter your question: ")

# SBERT-large-nli-v2 refers to a pre-trained model for the Sentence-BERT (SBERT) architecture. 
# SBERT is a modification of the BERT model for sentence embeddings. 
# Instead of outputting word embeddings like BERT, SBERT is designed to provide embeddings for whole sentences or paragraphs.

# SBERT: Stands for Sentence BERT, which indicates that the model is designed to produce sentence-level embeddings.
# - large: This might refer to the size of the underlying BERT model, suggesting that it's based on a "large" variant of BERT, which has more parameters and potentially provides more accurate representations, at the cost of being more computationally intensive.
# - nli: This suggests that the model was trained or fine-tuned on Natural Language Inference (NLI) tasks, which involve determining the relationship (entailment, contradiction, or neutral) between two sentences.
# - v2: This probably indicates the second version of the model.

asked_embeddings = []

# Function to find already asked question.
def is_similar_to_asked_questions(new_question_embedding):
    if asked_embeddings:
        # Concatenate embeddings along the first dimension to create a single tensor
        all_asked_embeddings = torch.cat(asked_embeddings, dim=0)
        cos_similarities = util.pytorch_cos_sim(new_question_embedding, all_asked_embeddings).numpy()
        
        # Checking against all asked questions
        for score in cos_similarities[0]:
            if score > 0.9:
                return True
    return False

# Define the three sets.
purchase_follow_up_questions = [
    "What are the available purchase options?",
    "What is the duration or term of my mortgage?",
    "Is my interest rate fixed for the entire term, or is it adjustable?",
    "How much should I set aside for a down payment?",
    "Can I get an estimate of what my monthly payment might be?",
    "Will I be required to pay Private Mortgage Insurance (PMI) when purchase a house?",
    "Are there any special features or provisions in my mortgage agreement that I should be aware of? when purchase a house?",
    "Can I Look At A House Without Preapproval?",
    "How early should you get pre-approved for a house?",
    "Should I get pre-approval before visiting the house?",
    "Can you provide more details about the pricing?",
    "Are there any discounts or promotions for purchases?"
]

refinance_follow_up_questions = [
    "What are the interest rates for refinancing?",
    "What is the requirement for refinance?",
    "Tell me about refinancing process",
    "What types of loans are available for refinancing?",
    "How long is the payback or break-even period when considering the costs of refinancing?",
    "Will I need private mortgage insurance (PMI) with my refinance?",
    "How long do I anticipate staying in the property post-refinance?",
    "Is there a prepayment penalty or exit fee if I exit my current mortgage when refinance?",
    "Will I need to undergo a new home appraisal when refinancing?",
    "Can you explain the benefits of refinancing?"
]

loan_options_follow_up_questions = [
    "What types of loan products are available",
    "How do the interest rates differ between these loan options?",
    "What are the loan term options?",
    "How does each loan option affect monthly payments and the total interest paid over the life of the loan?",
    "Do any of the loan options have provisions for early repayment without penalties?",
    "How do the down payment requirements vary between these loan options?",
    "What are the criteria or eligibility requirements for each loan type",
    "Can you explain the eligibility criteria for loans?",
    "How does the loan application process work?"
]

# Generic follow-up questions for the 'other' category
other_follow_up_questions = [
    "I want to know more about the process when buy a house",
    "I want to refinance",
    "Why working with Pacificwide?",
    "Tell me more about Pacificwide?",
    "What can the Pacificwide AI Mortgage Advisor do?"
]

# Combine all sets.
all_sets = [purchase_follow_up_questions, refinance_follow_up_questions, loan_options_follow_up_questions, other_follow_up_questions]

# Encode all sentences from these sets.
embeddings_purchase = model.encode(purchase_follow_up_questions, convert_to_tensor=True)
embeddings_refinance = model.encode(refinance_follow_up_questions, convert_to_tensor=True)
embeddings_loan_options = model.encode(loan_options_follow_up_questions, convert_to_tensor=True)
embeddings_others = model.encode(other_follow_up_questions, convert_to_tensor=True)

# Compute the embedding for input question.
embeddings_question = model.encode(input_question, convert_to_tensor=True)

# Compute the average embedding for each set.
average_embedding_purchase = torch.mean(embeddings_purchase, dim=0, keepdim=True)
average_embedding_refinance = torch.mean(embeddings_refinance, dim=0, keepdim=True)
average_embedding_loan_options = torch.mean(embeddings_loan_options, dim=0, keepdim=True)
average_embedding_others = torch.mean(embeddings_others, dim=0, keepdim=True)

average_embeddings = [average_embedding_purchase, average_embedding_refinance, average_embedding_loan_options, average_embedding_others]

# Compute the cosine similarity of the input question's embedding with each of the set's average embedding.
similarities = [util.pytorch_cos_sim(embeddings_question, avg_emb).item() for avg_emb in average_embeddings]


# Determine which category the question belongs to.
categories = ["Purchase", "Refinance", "Loan Options", "Others"]
max_similarity = max(similarities)

if max_similarity < 0.6:
    print(f"The input question: '{input_question}' doesn't closely match any category.")
else:
    max_index = similarities.index(max_similarity)
    print(f"The input question: '{input_question}' belongs to the '{categories[max_index]}' category.")

    # Filter out questions similar to the input question from the relevant set.
    filtered_questions = [q for q in all_sets[max_index] if not is_similar_to_asked_questions(model.encode(q, convert_to_tensor=True))]

    random_samples = random.sample(filtered_questions, min(3, len(filtered_questions)))  # Handle case when the list has less than 3 items.

    print("\nHere are some other questions from the same category:")
    for idx, question in enumerate(random_samples, 1):
        print(f"{idx}. {question}")
