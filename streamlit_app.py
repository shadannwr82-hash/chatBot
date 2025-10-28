# # install all necessary packages

# from itertools import zip_longest
# import streamlit as st
# from langchain_community.chat_models import ChatOpenAI

# from langchain.schema import(
#     SystemMessage, 
#     HumanMessage,
#     AIMessage
# )

# import os


# # load open_api_key
# openapi_key = st.secrets["OPENAI_API_KEY"]
# base_url   = "https://openrouter.ai/api/v1"
# model_name = "openai/gpt-4o"

# # set streamlit page

# st.set_page_config(page_title="ZMQ ChatBot")
# st.title("ZMQ AI Mentor")

# # session state management 
# if "generated" not in st.session_state:
#     st.session_state["generated"]=[]
# if "past" not in st.session_state:
#     st.session_state["past"]=[]
# if "entered_prompt" not in st.session_state:
#     st.session_state["entered_prompt"]=[] 

# # Initialize Chat Model

# chat = ChatOpenAI(
#     temperature  = 0.5,
#     model_name   = model_name,
#     # openapi_key  = openapi_key,
#     # openapi_key  = "OPENAI_API_KEY",
#     api_key=st.secrets["OPENAI_API_KEY"],

#     openapi_base = base_url,
#     max_tokens   = 200 

# )    

# # Build message history

# def build_message_list():
#     messages =[
#         SystemMessage(
#             content=(
#                 "Your name is AI Mentor. You are an AI Technical Expert for Artificial Intelligence, "
#                 "here to guide and assist students with their AI-related questions. "
#                 "Keep responses short (max 100 words), clear, and helpful."
#             )
#         )
#     ]

#     # for human_msg, ai_msg, in zip_longest(st.session_state['past'], st.session_state['generated']):
#     for human_msg, ai_msg in zip_longest(st.session_state["past"], st.session_state["generated"]):
#         if human_msg:
#             messages.append(HumanMessage(content=human_msg))
#         if ai_msg:
#             messages.append(AIMessage(content=ai_msg))
#     return messages

# # Generate AI response

# def generate_response():
#     messages    = build_message_list()
#     ai_response = chat(messages)
#     return ai_response.content

# # handle User Input

# def submit():
#     st.session_state.entered_prompt = st.session_state.prompt_input
#     st.session_state.prompt_input = ""

# st.text_input("YOU: ", key= "prompt_input", on_change=submit)

# if st.session_state.entered_prompt:
#     user_query = st.session_state.entered_prompt
#     st.session_state.past.append(user_query)
#     output = generate_response()
#     st.session_state.generated.response(output)

# # Display Chat History
# if st.session_state["generated"]:
#     for i in range(len(st.session_state["generated"])-1, -1, -1):
#        message(st.session_state["generated"][i], key = str(i))
#        message(st.session_state['past'][i], is_user = True, key = str(i) + "_user")



# # install all necessary packages

# from itertools import zip_longest
# import streamlit as st
# from streamlit_chat import message  # <- required to display chat messages
# from langchain_community.chat_models import ChatOpenAI
# from langchain.schema import SystemMessage, HumanMessage, AIMessage

# # ---------------------------------------------------------------------
# # Load API Key and Model Config
# # ---------------------------------------------------------------------
# api_key = st.secrets["OPENAI_API_KEY"]
# base_url = "https://openrouter.ai/api/v1"
# model_name = "openai/gpt-4o"

# # ---------------------------------------------------------------------
# # Streamlit Page Config
# # ---------------------------------------------------------------------
# st.set_page_config(page_title="ZMQ ChatBot")
# st.title("ZMQ AI Mentor")

# # ---------------------------------------------------------------------
# # Session State Management
# # ---------------------------------------------------------------------
# if "generated" not in st.session_state:
#     st.session_state["generated"] = []
# if "past" not in st.session_state:
#     st.session_state["past"] = []
# if "entered_prompt" not in st.session_state:
#     st.session_state["entered_prompt"] = ""

# # ---------------------------------------------------------------------
# # Initialize Chat Model (✅ FIXED HERE)
# # ---------------------------------------------------------------------
# # chat = ChatOpenAI(
# #     temperature=0.5,
# #     model_name=model_name,
# #     api_key=api_key,
# #     base_url=base_url,  # ✅ replaced openapi_base
# #     max_tokens=200,
# # )

# from langchain_community.chat_models import ChatOpenAI

# chat = ChatOpenAI(
#     temperature=0.5,
#     model_name=model_name,
#     openai_api_key=st.secrets["OPENAI_API_KEY"],
#     openai_api_base="https://openrouter.ai/api/v1",
#     model="gpt-3.5-turbo",
#     max_tokens=200,
# )


# # ---------------------------------------------------------------------
# # Build Message History
# # ---------------------------------------------------------------------
# def build_message_list():
#     messages = [
#         SystemMessage(
#             content=(
#                 "Your name is AI Mentor. You are an AI Technical Expert for Artificial Intelligence, "
#                 "here to guide and assist students with their AI-related questions. "
#                 "Keep responses short (max 100 words), clear, and helpful."
#             )
#         )
#     ]

#     for human_msg, ai_msg in zip_longest(st.session_state["past"], st.session_state["generated"]):
#         if human_msg:
#             messages.append(HumanMessage(content=human_msg))
#         if ai_msg:
#             messages.append(AIMessage(content=ai_msg))
#     return messages

# # ---------------------------------------------------------------------
# # Generate AI Response
# # ---------------------------------------------------------------------
# def generate_response():
#     messages = build_message_list()
#     ai_response = chat(messages)
#     return ai_response.content

# # ---------------------------------------------------------------------
# # Handle User Input
# # ---------------------------------------------------------------------
# def submit():
#     st.session_state.entered_prompt = st.session_state.prompt_input
#     st.session_state.prompt_input = ""

# st.text_input("YOU:", key="prompt_input", on_change=submit)

# if st.session_state.entered_prompt:
#     user_query = st.session_state.entered_prompt
#     st.session_state.past.append(user_query)
#     output = generate_response()
#     st.session_state.generated.append(output)  # ✅ fixed append

# # ---------------------------------------------------------------------
# # Display Chat History
# # ---------------------------------------------------------------------
# if st.session_state["generated"]:
#     for i in range(len(st.session_state["generated"]) - 1, -1, -1):
#         message(st.session_state["generated"][i], key=str(i))
#         message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")



# ---------------------------------------------------------------------
# Import Required Packages
# ---------------------------------------------------------------------

from itertools import zip_longest
import streamlit as st
from streamlit_chat import message  # to display chat messages

# ---------------------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------------------
st.set_page_config(page_title="ZMQ Offline ChatBot")
st.title("ZMQ Offline AI Mentor")

# ---------------------------------------------------------------------
# Session State Management
# ---------------------------------------------------------------------
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "entered_prompt" not in st.session_state:
    st.session_state["entered_prompt"] = ""

# # ---------------------------------------------------------------------
# # Rule-Based Response Logic
# # ---------------------------------------------------------------------
# def offline_chatbot_response(user_input: str) -> str:
#     """Generate a rule-based offline response."""
#     user_input = user_input.lower().strip()

#     # Greetings
#     if any(word in user_input for word in ["hi", "hello", "hey"]):
#         return "Hello! 👋 I'm your offline AI mentor. How can I help you today?"

#     # AI / ML questions
#     elif "machine learning" in user_input or "ml" in user_input:
#         return "Machine Learning is a part of AI where systems learn from data instead of explicit programming."

#     elif "artificial intelligence" in user_input or "ai" in user_input:
#         return "Artificial Intelligence is about creating smart systems that can think, learn, and make decisions."

#     elif "python" in user_input:
#         return "Python is the most popular language for AI and IoT. Would you like me to share how to start learning it?"

#     elif "iot" in user_input:
#         return "IoT connects devices using the internet so they can share data. For example, a smart bulb you control with your phone."

#     elif "thank" in user_input:
#         return "You're welcome! 😊 Keep learning and experimenting."

#     elif "who are you" in user_input:
#         return "I’m ZMQ AI Mentor, your offline assistant for learning AI, ML, and IoT topics."

#     elif "bye" in user_input:
#         return "Goodbye! 👋 Have a great day and keep coding."

#     # Fallback (if no rule matched)
#     else:
#         return "I'm not sure about that yet. Try asking me about AI, ML, Python, or IoT."

# ---------------------------------------------------------------------
# Rule-Based Offline Chatbot (Enhanced with 100+ Expert Rules)
# ---------------------------------------------------------------------
def offline_chatbot_response(user_input: str) -> str:
    """Generate a rule-based offline response with 100+ AI/IoT/Python learning rules."""
    user_input = user_input.lower().strip()

    # -----------------------------------------------------------------
    # GREETINGS & SMALL TALK
    # -----------------------------------------------------------------
    if any(word in user_input for word in ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]):
        return "Hello! 👋 I'm your offline AI mentor. How are you doing today?"

    elif "how are you" in user_input:
        return "I'm doing great, thank you for asking! How about you?"

    elif "what's up" in user_input or "wassup" in user_input:
        return "Just here to help you learn AI, ML, and IoT! What are you working on today?"

    elif "thank" in user_input:
        return "You're most welcome! 😊 Keep learning and building."

    elif "bye" in user_input or "see you" in user_input:
        return "Goodbye! 👋 Keep coding and stay curious."

    elif "who are you" in user_input:
        return "I’m ZMQ Mentor — your personal offline guide for AI, Machine Learning, IoT, and Python."

    elif "your name" in user_input:
        return "I’m called ZMQ AI Mentor — nice to meet you!"

    # -----------------------------------------------------------------
    # AI / MACHINE LEARNING BASICS
    # -----------------------------------------------------------------
    elif "artificial intelligence" in user_input or "what is ai" in user_input:
        return "Artificial Intelligence is about making machines think and act intelligently like humans."

    elif "machine learning" in user_input or "what is ml" in user_input:
        return "Machine Learning teaches computers to learn from data instead of explicit programming."

    elif "deep learning" in user_input:
        return "Deep Learning is a subset of ML that uses neural networks with many layers — inspired by the human brain."

    elif "neural network" in user_input:
        return "A neural network is a computational model designed to simulate how human brains process information."

    elif "supervised learning" in user_input:
        return "Supervised Learning is when models learn from labeled data — input with known output."

    elif "unsupervised learning" in user_input:
        return "Unsupervised Learning finds hidden patterns or structures in unlabeled data."

    elif "reinforcement learning" in user_input:
        return "Reinforcement Learning trains an agent to make decisions by rewarding good actions and punishing bad ones."

    elif "difference between ai and ml" in user_input:
        return "AI is the broad concept of machines being smart; ML is a subset of AI that learns from data."

    elif "dataset" in user_input:
        return "A dataset is a collection of data used for training and testing machine learning models."

    elif "overfitting" in user_input:
        return "Overfitting happens when a model learns the training data too well but performs poorly on unseen data."

    elif "underfitting" in user_input:
        return "Underfitting occurs when a model is too simple and fails to capture data patterns."

    elif "accuracy" in user_input:
        return "Accuracy measures how often your model’s predictions are correct."

    elif "confusion matrix" in user_input:
        return "A confusion matrix shows how well your classification model performs by comparing predicted vs actual labels."

    elif "feature" in user_input:
        return "A feature is an input variable used by a model to make predictions."

    elif "label" in user_input:
        return "A label is the target output or category that the model is trying to predict."

    elif "training" in user_input:
        return "Training is the process where a model learns from historical data."

    elif "testing" in user_input:
        return "Testing checks how well the model performs on new, unseen data."

    # -----------------------------------------------------------------
    # PYTHON PROGRAMMING
    # -----------------------------------------------------------------
    elif "python" in user_input:
        return "Python is the most popular language for AI, ML, and IoT because of its simplicity and strong libraries."

    elif "how to learn python" in user_input:
        return "Start with basic syntax, loops, and functions. Then move to libraries like NumPy, Pandas, and Matplotlib."

    elif "numpy" in user_input:
        return "NumPy is a Python library for numerical computing and handling large multi-dimensional arrays."

    elif "pandas" in user_input:
        return "Pandas is great for data analysis and manipulation using DataFrames."

    elif "matplotlib" in user_input:
        return "Matplotlib helps visualize data through charts, graphs, and plots."

    elif "sklearn" in user_input or "scikit-learn" in user_input:
        return "Scikit-learn is a library for building ML models — classification, regression, clustering, and more."

    elif "tensorflow" in user_input:
        return "TensorFlow is an open-source library for deep learning and neural networks."

    elif "pytorch" in user_input:
        return "PyTorch is a popular deep learning library known for flexibility and dynamic computation graphs."

    elif "python loop" in user_input:
        return "Python supports 'for' and 'while' loops for iteration. Example: for i in range(5): print(i)"

    elif "function" in user_input:
        return "A Python function is defined using 'def'. Example: def greet(): print('Hello!')"

    elif "list" in user_input:
        return "Lists store multiple items in one variable, e.g. fruits = ['apple', 'banana', 'cherry']"

    elif "dictionary" in user_input:
        return "A dictionary stores key-value pairs, e.g. student = {'name':'John','age':21}"

    elif "tuple" in user_input:
        return "Tuples are immutable lists. Example: coordinates = (10, 20)"

    elif "string" in user_input:
        return "Strings are sequences of text, e.g. name = 'Shadan'. You can use slicing and concatenation on them."

    # -----------------------------------------------------------------
    # IOT & HARDWARE
    # -----------------------------------------------------------------
    elif "iot" in user_input:
        return "IoT connects physical devices via the Internet so they can collect and exchange data."

    elif "esp32" in user_input:
        return "ESP32 is a powerful microcontroller with Wi-Fi and Bluetooth — perfect for IoT projects."

    elif "esp8266" in user_input:
        return "ESP8266 is a low-cost Wi-Fi microchip that connects sensors and devices to the internet."

    elif "raspberry pi" in user_input:
        return "Raspberry Pi is a mini computer used in robotics, IoT, and AI projects."

    elif "arduino" in user_input:
        return "Arduino is an open-source microcontroller board great for beginners to build electronics projects."

    elif "sensor" in user_input:
        return "A sensor detects physical changes (like temperature, light, or motion) and converts them into signals."

    elif "actuator" in user_input:
        return "An actuator converts electrical signals into physical action, like motors or relays."

    elif "mqtt" in user_input:
        return "MQTT is a lightweight protocol for sending IoT data between devices and servers."

    elif "emqx" in user_input:
        return "EMQX is an open-source MQTT broker for real-time IoT data exchange."

    elif "blynk" in user_input:
        return "Blynk is a platform for building mobile and web dashboards to control IoT devices remotely."

    elif "wi-fi credentials" in user_input or "wifi" in user_input:
        return "You can store Wi-Fi credentials in SPIFFS or EEPROM and connect automatically on boot."

    elif "spiffs" in user_input:
        return "SPIFFS (SPI Flash File System) lets ESP boards store small files like Wi-Fi credentials or configs."

    elif "dht11" in user_input:
        return "DHT11 is a sensor that measures temperature and humidity — great for simple IoT weather stations."

    elif "relay" in user_input:
        return "A relay is an electrically operated switch used to control high voltage devices using low voltage signals."

    elif "web server" in user_input:
        return "A web server on ESP hosts a webpage that interacts with IoT devices via HTML forms or AJAX."

    # -----------------------------------------------------------------
    # DATA SCIENCE & ANALYTICS
    # -----------------------------------------------------------------
    elif "data science" in user_input:
        return "Data Science combines statistics, programming, and domain expertise to extract insights from data."

    elif "data analysis" in user_input:
        return "Data Analysis means cleaning, visualizing, and interpreting data to make informed decisions."

    elif "eda" in user_input or "exploratory data analysis" in user_input:
        return "EDA helps you understand data patterns, detect outliers, and prepare datasets for ML."

    elif "csv" in user_input:
        return "A CSV file stores tabular data in text form — each line represents a data record separated by commas."

    elif "vgsales.csv" in user_input:
        return "The 'vgsales.csv' dataset tracks video game sales by platform, year, and region — good for EDA practice."

    # -----------------------------------------------------------------
    # CAREER & LEARNING
    # -----------------------------------------------------------------
    elif "how to start ai" in user_input:
        return "Start with Python → Math (Linear Algebra, Statistics) → ML → DL → Projects → Portfolio."

    elif "how to start iot" in user_input:
        return "Begin with Arduino or ESP32 → Sensors → Wi-Fi → MQTT → Dashboard → Cloud Integration."

    elif "career" in user_input:
        return "AI and IoT careers are in high demand. Focus on projects, certifications, and consistent learning."

    elif "project" in user_input:
        return "Try small projects like IoT temperature monitor, AI chatbot, or voice-controlled smart light."

    elif "portfolio" in user_input:
        return "A strong portfolio should include GitHub projects, documentation, and live demos."

    elif "free resources" in user_input or "free course" in user_input:
        return "You can explore free learning at YouTube (freeCodeCamp, Tech With Tim, Code With Harry) and Coursera audits."

    # -----------------------------------------------------------------
    # MOTIVATION & PRODUCTIVITY
    # -----------------------------------------------------------------
    elif "motivate" in user_input or "motivation" in user_input:
        return "Every expert was once a beginner. Keep learning — your consistency is more powerful than talent."

    elif "lazy" in user_input:
        return "Try breaking tasks into small goals. Start with 10 minutes daily — momentum will follow."

    elif "how to focus" in user_input:
        return "Eliminate distractions, set a timer (Pomodoro), and reward yourself after each focused session."

    elif "give me quote" in user_input:
        return "“The future belongs to those who learn more skills and combine them creatively.” – Robert Greene"

    elif "learn faster" in user_input:
        return "Use active recall, teaching what you learn, and work on real projects — not just theory."

    # -----------------------------------------------------------------
    # FALLBACK
    # -----------------------------------------------------------------
    else:
        return "Hmm 🤔 I’m not sure about that yet. Try asking about AI, ML, IoT, Python, or your learning path."



# ---------------------------------------------------------------------
# Handle User Input
# ---------------------------------------------------------------------
def submit():
    st.session_state.entered_prompt = st.session_state.prompt_input
    st.session_state.prompt_input = ""

st.text_input("YOU:", key="prompt_input", on_change=submit)

if st.session_state.entered_prompt:
    user_query = st.session_state.entered_prompt
    st.session_state.past.append(user_query)

    # Use offline rule-based response instead of API
    output = offline_chatbot_response(user_query)
    st.session_state.generated.append(output)

# ---------------------------------------------------------------------
# Display Chat History
# ---------------------------------------------------------------------
if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")



# # ---------------------------------------------------------------------
# # Import Required Packages
# # ---------------------------------------------------------------------
# import streamlit as st
# from streamlit_chat import message
# from PyPDF2 import PdfReader
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # ---------------------------------------------------------------------
# # Streamlit Page Config
# # ---------------------------------------------------------------------
# st.set_page_config(page_title="📘 ZMQ PDF ChatBot")
# st.title("📘 ZMQ Offline Chatbot (Answer from about.pdf)")

# # ---------------------------------------------------------------------
# # Session State
# # ---------------------------------------------------------------------
# if "generated" not in st.session_state:
#     st.session_state["generated"] = []
# if "past" not in st.session_state:
#     st.session_state["past"] = []
# if "entered_prompt" not in st.session_state:
#     st.session_state["entered_prompt"] = ""

# # ---------------------------------------------------------------------
# # Load and preprocess PDF
# # ---------------------------------------------------------------------
# @st.cache_data
# def load_pdf_text(pdf_path):
#     """Extract text from PDF file."""
#     pdf_reader = PdfReader(pdf_path)
#     text = ""
#     for page in pdf_reader.pages:
#         text += page.extract_text() or ""
#     return text

# @st.cache_resource
# def prepare_pdf_knowledgebase(pdf_path):
#     """Prepare vectorized chunks for semantic search."""
#     raw_text = load_pdf_text(pdf_path)
#     # Split into chunks (roughly 300 words each)
#     chunks = [raw_text[i:i + 1000] for i in range(0, len(raw_text), 1000)]
#     vectorizer = TfidfVectorizer(stop_words='english')
#     vectors = vectorizer.fit_transform(chunks)
#     return chunks, vectorizer, vectors

# # Load your PDF file here (must exist in the same folder)
# # PDF_FILE = "about.pdf"
# PDF_FILE = "AboutUS.pdf"
# chunks, vectorizer, vectors = prepare_pdf_knowledgebase(PDF_FILE)

# # ---------------------------------------------------------------------
# # PDF-based Answer Retrieval
# # ---------------------------------------------------------------------
# def pdf_chatbot_response(user_input):
#     """Find the most relevant chunk from the PDF."""
#     query_vec = vectorizer.transform([user_input])
#     similarity_scores = cosine_similarity(query_vec, vectors)
#     best_match_idx = similarity_scores.argmax()
#     best_chunk = chunks[best_match_idx]

#     # Shorten long answers
#     if len(best_chunk) > 500:
#         best_chunk = best_chunk[:500] + "..."
#     return best_chunk

# # ---------------------------------------------------------------------
# # Handle User Input
# # ---------------------------------------------------------------------
# def submit():
#     st.session_state.entered_prompt = st.session_state.prompt_input
#     st.session_state.prompt_input = ""

# st.text_input("YOU:", key="prompt_input", on_change=submit)

# if st.session_state.entered_prompt:
#     user_query = st.session_state.entered_prompt
#     st.session_state.past.append(user_query)

#     # Get answer from PDF
#     output = pdf_chatbot_response(user_query)
#     st.session_state.generated.append(output)

# # ---------------------------------------------------------------------
# # Display Chat History
# # ---------------------------------------------------------------------
# if st.session_state["generated"]:
#     for i in range(len(st.session_state["generated"]) - 1, -1, -1):
#         message(st.session_state["generated"][i], key=str(i))
#         message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")







