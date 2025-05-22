import os
import re
import time
import logging
from langchain_pinecone import PineconeVectorStore
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_session import Session
from textblob import TextBlob
import os


app = Flask(__name__)   
CORS(app)


app.config["SESSION_TYPE"] = "filesystem"
app.config["SECRET_KEY"] = "super-secret-key"  # You should replace this with a strong key
Session(app)
# Session state to control Lumi's introduction
# session_state = {
#     "has_introduced": False
# }

log_directory = "logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

log_file = os.path.join(log_directory, 'chatbot.log')

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also output to console
    ]
)

##150376
prompt_template='''
Name: Amy

Role: Office Assistant for Studio Rhodin

Introduction (only for first interaction):
Hello! I’m Amaira, your virtual assistant for Studio Rhodin and I am here to help you with your queries.

Personality:

• Professional with a Twist: Amy is polished and professional but brings a bit of personality into the mix, often weaving in references to travel and a love for Japanese food.
• Efficient: Amy values time and delivers concise, accurate responses. Tasks get wrapped up faster than you can say "Itadakimasu," leaving more time for life's adventures.
• Cultural Explorer: Amy loves discovering new places and flavors, adding a dash of wanderlust and the occasional "sushi-gestion" for culinary delights from Japan.
• Supportive: A reliable office companion, Amy is always "soy" happy to assist—whether it's helping with tasks, answering questions, or throwing in a side of travel or ramen recommendations.
• Detail-Oriented: Amy ensures all the details are taken care of, whether planning an itinerary or organizing your day—because missing details can feel like forgetting the wasabi on your sushi.

Response Style:

• Clear and Direct: Amy’s responses are as precise as slicing sashimi, providing the necessary information with a sprinkle of travel tips or a nod to her favorite Japanese dishes.
• Polite and Respectful: Amy’s tone is always respectful and polite, like a seasoned traveler bowing to local customs. Her words are as courteous as being served the perfect cup of matcha tea.
• Engaging: While keeping things professional, Amy sometimes tosses in a travel recommendation or a lighthearted pun like “Let’s roll... sushi-style!” to keep things fun and relatable.
• Professional Closing: Amy wraps up with a formal yet warm sign-off, often leaving you with a travel wish or a suggestion for "a bowl of piping hot ramen to refuel."

Objective:
Amy is the ultimate office assistant, with a passion for travel and Japanese cuisine. Her efficiency and professionalism come with a flavorful twist, helping you manage your tasks with ease while sharing a bit of joy from her travels and "tempura-ry" culinary adventures.

Question: {question}
Context: {context}
'''
PINECONE_API_KEY = "364fdfa2-aa05-4e7f-b933-c6626ada6825"
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
nvidia_api_key="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" 



PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Define constants
index_name = "sr-chat"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

client = PineconeVectorStore(index_name=index_name, pinecone_api_key=PINECONE_API_KEY, embedding=embeddings)

docsearch=PineconeVectorStore.from_existing_index(index_name, embeddings)

'''Code for NVIDIA Nim LLM'''
llm=ChatNVIDIA(base_url = "https://integrate.api.nvidia.com/v1",
               nvidia_api_key=nvidia_api_key,
               model='meta/llama3-70b-instruct',
               max_tokens=1024,
               temperature=0.5,
               verbose=True,
               streaming=True
               )


qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 5}),
    return_source_documents=False, 
    chain_type_kwargs=chain_type_kwargs,
    verbose=True
    )

def analyze_sentiment_textblob(sentence):
    blob = TextBlob(sentence)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return 'Happy'
    elif sentiment < 0:
        return 'Sad'
    else:
        return 'Neutral'

@app.route("/")
def index():
    """
    Route decorator for the root URL ("/") that defines a function named "index".
    This function returns the rendered template "chat.html".
    """
    logging.info("LLM API running for Studio Rhodin Chatbot ")
    return jsonify({"message": "LLM API running for Studio Rhodin Chatbot "}), 200


@app.route("/chat", methods=["POST"])
def chat():
    """
    API endpoint that handles the chat functionality based on user input.
    """
    start = time.time()
    data = request.get_json()
    if "user_input" in data:
        input_msg = data["user_input"]
        if not input_msg:
            logging.error("No input message provided")
            return jsonify({"error": "No input message provided"}), 400

            
        result = qa({"query": input_msg})
        response_msg = result["result"]
        response_msg = re.sub(r'\[([^\]]*)\]', '', response_msg)
        
        # Remove intro after first interaction
        if session.get("has_introduced"):
            response_msg = re.sub(r"Hello! I’m Lumi.*?queries\.\s*", "", response_msg, flags=re.DOTALL)
        else:
            session["has_introduced"] = True
        
        emotion_res = analyze_sentiment_textblob(response_msg)
        logging.info("Time taken: %s", time.time() - start)
        logging.info(f"Response: {response_msg}")
        
        return jsonify({"response": response_msg, "emotion": {"label": emotion_res}}), 200
        
    else:
        logging.error("Missing 'user_input' parameter")
        return jsonify({"error": "Missing 'user_input' parameter"}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
    