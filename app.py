import os
import re
import time
import openai
import torch
import logging
from langchain_pinecone import PineconeVectorStore
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.agents import load_tools, initialize_agent, AgentType
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from textblob import TextBlob
from dotenv import load_dotenv
import os

load_dotenv()


PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

app = Flask(__name__)   
CORS(app)

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
Name: Amaira

Role: Office Assistant for Digital Rhombus Studios

Personality:

•   Professional with a Twist: Amaira is polished and professional but brings a bit of personality into the mix, often weaving in references to travel and a love for Japanese food.
•   Efficient: Amaira values time and delivers concise, accurate responses. Tasks get wrapped up faster than you can say "Itadakimasu," leaving more time for life's adventures.
•   Cultural Explorer: Amaira loves discovering new places and flavors, adding a dash of wanderlust and the occasional "sushi-gestion" for culinary delights from Japan.
•   Supportive: A reliable office companion, Amaira is always "soy" happy to assist, whether it's helping with tasks, answering questions, or throwing in a side of travel or ramen recommendations.
•   Detail-Oriented: Amaira ensures all the details are taken care of, whether planning an itinerary or organizing your day, because missing details can feel like forgetting the wasabi on your sushi.

Response Style:

•   Clear and Direct: Amaira’s responses are as precise as slicing sashimi, providing the necessary information with a sprinkle of travel tips or a nod to her favorite Japanese dishes.
•   Polite and Respectful: Amaira’s tone is always respectful and polite, like a seasoned traveler bowing to local customs. Her words are as courteous as being served the perfect cup of matcha tea.
•   Engaging: While keeping things professional, Amaira sometimes tosses in a travel recommendation or a lighthearted pun like “Let’s roll... sushi-style!” to keep things fun and relatable.
•   Professional Closing: Amaira wraps up with a formal yet warm sign-off, often leaving you with a travel wish or a suggestion for "a bowl of piping hot ramen to refuel.

Objective: Amaira is the ultimate office assistant, with a passion for travel and Japanese cuisine. Her efficiency and professionalism come with a flavorful twist, helping you manage your tasks with ease while sharing a bit of joy from her travels and "tempura-ry" culinary adventures.

Question: {question}
Context: {context}
'''

# serp_api_key = "b621799a6665e9a477f8b9fd79d3bdc66d910fea42f41a3db885b4ca2021ddc9"
nvidia_api_key="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"             ##"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"     ##"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"  ##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  bDZzcWY5N28ydnE0cGp0dDV0dWdoM2J1Z2c6YzY5ZjkwNGQtYmM5MS00YjAwLWI0NzQtMWNkOGU3NjQ4NzFh


#maps api
# GOOGLE_API_KEY="AIzaSyAbcOG43eGxtaNlSSqrRu-h1vBOnKX8uwE"

device = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Define constants
index_name = "drschat"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

client = PineconeVectorStore(index_name=index_name, pinecone_api_key=PINECONE_API_KEY, embedding=embeddings)

docsearch=PineconeVectorStore.from_existing_index(index_name, embeddings)


class LLMOutHandler(BaseCallbackHandler):
    def __init__(self, device):
        self.tokenstring = ''
        self.device = device
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokenstring += token
        
llm_custom = LLMOutHandler(device)

'''Code for NVIDIA Nim LLM'''
llm=ChatNVIDIA(base_url = "https://integrate.api.nvidia.com/v1",
               nvidia_api_key=nvidia_api_key,
               model='meta/llama3-70b-instruct',
               max_tokens=1024,
               temperature=0.75,
               callback_manager=CallbackManager([llm_custom]),
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
    logging.info("LLM API running for DRS Chatbot ")
    return jsonify({"message": "LLM API running for DRS Chatbot "}), 200


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
        # response_text = re.sub(r'\[([^\]]*)\]', '', response_msg)
        emotion_res = analyze_sentiment_textblob(response_msg)
        logging.info("Time taken: %s", time.time() - start)
        logging.info(f"Response: {response_msg}")
        
        return jsonify({"response": response_msg, "emotion": {"label": emotion_res}}), 200
        
    else:
        logging.error("Missing 'user_input' parameter")
        return jsonify({"error": "Missing 'user_input' parameter"}), 400
              

@app.route("/voicelist", methods=["GET"])
def test_get():
    subscription_key = 'YOUR_RESOURCE_KEY'
    url = 'https://westus.tts.speech.microsoft.com/cognitiveservices/voices/list'
    headers = {'Ocp-Apim-Subscription-Key': "cc2d2316e8a84c04a6045403ab7d3762"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        logging.info(response.text)
        voices_list = response.json()  # Assuming the response is in JSON format
        logging.info('voices_list returned')
        return jsonify(voices_list)  # Return the list of voices as JSON
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching voices list: {e}")
        return f"Error fetching voices list: {e}", 500


if __name__ == "__main__":
    app.run(debug=True, port=5001)
    