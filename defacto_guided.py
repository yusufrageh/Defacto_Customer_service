import os
import requests
import json
import pandas as pd
from flask import Flask, request, jsonify
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
import faiss
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Load API key
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Load multiple FAQ datasets
faq_df_general = pd.read_csv("defacto_faq_data.csv")
faq_df_products = pd.read_csv("men_clothes.csv")
faq_df_shipping = pd.read_csv("orders_men300.csv")

# Define column mappings for each file
column_mappings = {
    'general': {
        'question_col': 'question',  # column used for similarity search
        'all_cols': list(faq_df_general.columns)  # automatically get all columns
    },
    'products': {
        'question_col': 'product_name',
        'all_cols': list(faq_df_products.columns)
    },
    'shipping': {
        'question_col': 'order_id',
        'all_cols': list(faq_df_shipping.columns)
    }
}

# Initialize the LLM
llm = ChatOpenAI(
    openai_api_key=os.getenv("GROQ_API_KEY"),
    temperature=1,
    openai_api_base="https://api.groq.com/openai/v1",
    model="llama-3.3-70b-versatile"
)
memory = ConversationBufferWindowMemory(k=5, return_messages=True)
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)



# 1. Define the text splitter
def split_text(text, chunk_size=150, chunk_overlap=20):
    """Split text into smaller chunks for better similarity search."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n", ".", "!", "?", " ", ""]
    )
    return splitter.split_text(text)

# 2. Load embedding model (Must be before using it)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Define preprocessing function
def preprocess_faq_data(df, category):
    """Preprocess FAQ data by splitting text into smaller chunks."""
    question_col = column_mappings[category]['question_col']
    
    processed_rows = []
    
    for _, row in df.iterrows():
        original_text = row[question_col]

        if pd.notna(original_text):
            chunks = split_text(original_text)
            for chunk in chunks:
                new_row = row.copy()
                new_row[question_col] = chunk
                processed_rows.append(new_row)

    return pd.DataFrame(processed_rows)


# 5. Preprocess FAQ data before indexing
faq_df_general = preprocess_faq_data(faq_df_general, 'general')
faq_df_products = preprocess_faq_data(faq_df_products, 'products')
faq_df_shipping = preprocess_faq_data(faq_df_shipping, 'shipping')

# 6. Define FAISS indexing function
def create_faiss_index(df, category):
    """Create FAISS index for a dataframe with preprocessed question chunks."""
    question_col = column_mappings[category]['question_col']
    
    if question_col not in df.columns:
        raise ValueError(f"Column '{question_col}' not found in {category} FAQ file")

    texts = df[question_col].tolist()  # Use preprocessed questions
    vectors = np.array([embedding_model.embed_query(q) for q in texts]).astype("float32")

    if vectors.size == 0:
        raise ValueError("No embeddings generated. Check your data.")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    return index, vectors

# 7. Create FAISS indices for each dataset
try:
    index_general, vectors_general = create_faiss_index(faq_df_general, 'general')
    index_products, vectors_products = create_faiss_index(faq_df_products, 'products')
    index_shipping, vectors_shipping = create_faiss_index(faq_df_shipping, 'shipping')
except ValueError as e:
    print(f"Error creating indices: {e}")

# 8. Define a function to format FAQ entries
def format_faq_entry(row, category):
    """Format all columns of a FAQ entry into a readable string"""
    columns = column_mappings[category]['all_cols']
    formatted_parts = []

    for col in columns:
        if pd.notna(row[col]):  # Only include non-null values
            formatted_parts.append(f"{col}: {row[col]}")

    return "\n".join(formatted_parts)


def mmr(doc_embeddings, query_embedding, top_k=3, lambda_param=0.5):
    """Calculate Maximum Marginal Relevance (MMR)."""
    query_similarity = cosine_similarity(query_embedding, doc_embeddings)[0]
    doc_similarity = cosine_similarity(doc_embeddings)
    
    selected_indices = []
    candidate_indices = list(range(len(doc_embeddings)))
    
    selected_indices.append(np.argmax(query_similarity))
    candidate_indices.remove(selected_indices[0])
    
    for _ in range(min(top_k - 1, len(candidate_indices))):
        mmr_scores = []
        for i in candidate_indices:
            relevance = query_similarity[i]
            diversity = max([doc_similarity[i][j] for j in selected_indices])
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
            mmr_scores.append(mmr_score)
        
        selected_index = candidate_indices[np.argmax(mmr_scores)]
        selected_indices.append(selected_index)
        candidate_indices.remove(selected_index)
    
    return selected_indices

def search_specific_faq(question, recipient_id):
    """Search FAQs in the category selected by the user."""

    # Get category from user selection (default to 'general' if missing)
    category = user_context.get(recipient_id, 'general').lower()

    # Select the appropriate dataset and vector index
    if category == 'general':
        df, index, vectors = faq_df_general, index_general, vectors_general
    elif category == 'products':
        df, index, vectors = faq_df_products, index_products, vectors_products
    elif category == 'shipping':
        df, index, vectors = faq_df_shipping, index_shipping, vectors_shipping
    else:
        return [{"full_entry": "Invalid category selection.", "Category": "Unknown"}]

    # Column mapping for question retrieval
    question_col = column_mappings[category]['question_col']

    # Convert user query into vector
    query_vector = np.array([embedding_model.embed_query(question)]).astype("float32")

    # Perform nearest-neighbor search
    distances, indices = index.search(query_vector, 10)

    # Handle case when no results are found
    if len(indices[0]) == 0:
        return [{"full_entry": "No matching FAQs found.", "Category": category.capitalize()}]

    # Use precomputed embeddings instead of recomputing them
    doc_embeddings = vectors[indices[0]]

    # Apply MMR to rank the most relevant FAQs
    mmr_indices = mmr(doc_embeddings, query_vector, top_k=3, lambda_param=0.5)

    # Format the retrieved results
    results = [
        {
            "full_entry": format_faq_entry(df.iloc[indices[0][i]], category),
            "Category": category.capitalize()
        }
        for i in mmr_indices
    ]

    return results



def query_llm_for_faq(user_question, recipient_id):
    """Enhanced version that searches FAQs based on selected category."""
    try:
        # Search FAQs based on the user's selected category
        primary_results = search_specific_faq(user_question, recipient_id)

        # Format the FAQ results for better readability
        formatted_faqs = "\n\n".join([entry["full_entry"] for entry in primary_results])

        # Construct the LLM prompt
        prompt = f"""
        You are a customer service chatbot for Defacto.
        Answer the user's question using the provided information.
        Be concise and give a short answer that is very relevant to the question.
        If any column contains a link, provide it with your answer.
        
        User question: "{user_question[:500]}"

        Most relevant information:
        {formatted_faqs if formatted_faqs else "No relevant FAQ found."}

        If unsure, say: "I am not sure, please contact support."
        """

        response = conversation.predict(input=prompt)
        return response

    except Exception as e:
        print(f"❌ Error querying LLM: {e}")
        return "I am currently experiencing issues. Please try again later."


VERIFY_TOKEN = "RMR_123456"
PAGE_ACCESS_TOKEN = "EAA4dZBp7wPpMBO8BOUvk3OR6iumoo97185TxNZCD6QFG3ZAkv4zhEr8ghLztWXmu6VMvOQ93LPgmx52IQRHJmTVHkR9hes0xcfArPNZC8UBZC52bvcQyzQcHEWECLlAZASzhk0XQ77NOosEAl7mU2GNZBk4FJhByhComh0tZBneZAJPPII7l4dc3HJo0K2gyguaaxLwZDZD"

@app.route("/", methods=["GET"])
def home():
    return "Defacto Chatbot is running!"

@app.route("/webhook", methods=["GET"])
def verify():
    """Verify webhook with Facebook."""
    token_sent = request.args.get("hub.verify_token")
    if token_sent == VERIFY_TOKEN:
        return request.args.get("hub.challenge")
    return "Invalid verification token", 403



user_context = {}  # Store user-selected categories

def send_message(recipient_id, message_text):
    """Send a response message to the user via Facebook Messenger API."""
    try:
        url = "https://graph.facebook.com/v18.0/me/messages"
        headers = {"Content-Type": "application/json"}
        params = {"access_token": PAGE_ACCESS_TOKEN}
        payload = {
            "recipient": {"id": recipient_id},
            "message": {"text": message_text}
        }
        response = requests.post(url, headers=headers, params=params, json=payload)

        if response.status_code != 200:
            print(f"❌ Error sending message: {response.text}")

    except Exception as e:
        print(f"❌ Error in send_message: {e}")


def handle_postback(recipient_id, payload):
    """Handle button clicks (postbacks) and store the user's selection."""
    global user_context  # Ensure we modify the global user context

    if payload == "FAQ_SELECTED":
        user_context[recipient_id] = "general"
        send_message(recipient_id, "You selected FAQs. Please type your question.")
    elif payload == "PRODUCT_SELECTED":
        user_context[recipient_id] = "Products"
        send_message(recipient_id, "You selected Products. What would you like to know?")
    elif payload == "SHIPPING_SELECTED":
        user_context[recipient_id] = "Shipping"
        send_message(recipient_id, "You selected Shipping. Please ask about shipping details.")

def show_buttons(recipient_id):
    """Send a message with buttons to guide the user."""
    url = "https://graph.facebook.com/v18.0/me/messages"
    headers = {"Content-Type": "application/json"}
    params = {"access_token": PAGE_ACCESS_TOKEN}

    payload = {
        "recipient": {"id": recipient_id},
        "message": {
            "attachment": {
                "type": "template",
                "payload": {
                    "template_type": "button",
                    "text": "How can I assist you today?",
                    "buttons": [
                        {
                            "type": "postback",
                            "title": "FAQs",
                            "payload": "FAQ_SELECTED"
                        },
                        {
                            "type": "postback",
                            "title": "Products",
                            "payload": "PRODUCT_SELECTED"
                        },
                        {
                            "type": "postback",
                            "title": "Shipping",
                            "payload": "SHIPPING_SELECTED"
                        }
                    ]
                }
            }
        }
    }

    response = requests.post(url, headers=headers, params=params, json=payload)
    if response.status_code != 200:
        print(f"❌ Error sending buttons: {response.text}")

# Modify webhook() to call show_buttons() when a user first interacts
@app.route("/webhook", methods=["POST"])
def webhook():
    """Receive messages from Facebook and respond."""
    try:
        data = request.get_json()
        print(f"📩 Received Data: {data}")

        if not data or "object" not in data:
            return "Invalid request", 400

        if data.get("object") == "page":
            for entry in data.get("entry", []):
                for messaging_event in entry.get("messaging", []):
                    sender_id = messaging_event["sender"]["id"]

                    # Handle Postbacks (Button Clicks)
                    if messaging_event.get("postback"):
                        payload = messaging_event["postback"]["payload"]
                        handle_postback(sender_id, payload)
                    
                    # Handle Text Messages
                    elif messaging_event.get("message"):
                        message_text = messaging_event["message"].get("text", "")
                        print(f"📨 Received message: {message_text} from {sender_id}")

                        if sender_id in user_context:
                            category = user_context[sender_id]  # Get the last selected category
                            query_text = f"{category} - {message_text}"

                            # 🛠️ **Fix: Pass `recipient_id` to `query_llm_for_faq()`**
                            response = query_llm_for_faq(query_text, sender_id)  # ✅ Fix: Now passing `sender_id`

                            print(f"🤖 LLM Response: {response}")
                            send_message(sender_id, response)  # Send response to user
                        else:
                            # If no category is selected, show buttons first
                            show_buttons(sender_id)

        return "Message Processed", 200

    except Exception as e:
        print(f"❌ Webhook Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)