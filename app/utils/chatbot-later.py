import streamlit as st
import re
from collections import defaultdict
from detoxify import Detoxify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from ner_utils import load_model_and_tokenizer, mask_names_and_orgs
# Load environment variables
load_dotenv()

# Women's Resource Hub (Modular Configuration)
WOMENS_RESOURCES = {
    "emergency_contacts": ["National Helpline: 112", "Women's Commission: 123-456-7890"],
    "health_resources": {"Physical Health": "https://example.com", "Mental Health": "https://example.org"},
    "safety_features": {"Location Sharing": True, "Emergency SOS": False}
}

# ------------------------------------------
# Caching resource-intensive models
@st.cache_resource(show_spinner=False)
def load_detoxify_model():
    return Detoxify('original')

@st.cache_resource(show_spinner=False)
def load_emotion_detector():
    from transformers import pipeline
    return pipeline("sentiment-analysis")

detoxify_model = load_detoxify_model()
emotion_detector = load_emotion_detector()

# ------------------------------------------
# Initialize User Profile and Conversation Log
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {'name': None, 'last_checkin': None}
if 'conversation_log' not in st.session_state:
    st.session_state.conversation_log = []

# ------------------------------------------
# Anonymization functions to remove personal data (names, emails, phone numbers)
def anonymize_names(text):
    # Simple regex to replace capitalized words that might be names
    pattern = r'\b[A-Z][a-z]+\b'
    return re.sub(pattern, "[NAME]", text)

def anonymize_text(text):
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    text = re.sub(r'\b\d{10,}\b', '[PHONE]', text)
    ner_model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
    ner_model, ner_tokenizer = load_model_and_tokenizer(ner_model_name)
    id2label = ner_model.config.id2label
    text = mask_names_and_orgs(text, ner_model, ner_tokenizer, id2label)
    return text

# ------------------------------------------
# Advanced Emergency Escalation: Simulate notifying a human operator
def escalate_emergency(user_query):
    st.write("**Emergency escalation activated!** Notifying a human operator.")
    st.session_state.emergency_alert = f"Emergency detected: {user_query}"
    # Integration with an actual notification system can be added here.

# ------------------------------------------
# Fixed prompt for the retrieval chain (no A/B testing)
def get_context_retriever_chain(vectordb):
    if 'llm' not in st.session_state:
        st.session_state.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.3,
            convert_system_message_to_human=True
        )
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    system_prompt = (
        "You are an empathetic and supportive assistant specialized in women's health and well-being. "
        "Provide detailed advice and proactive resource recommendations. "
        "Maintain a trauma-informed approach and clarify ambiguous queries before answering. "
        "Always include a disclaimer that your advice is informational."
        "\nContext: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    return create_retrieval_chain(
        retriever,
        create_stuff_documents_chain(st.session_state.llm, prompt)
    )

def get_response(question, chat_history, vectordb):
    chain = get_context_retriever_chain(vectordb)
    response = chain.invoke({"input": question, "chat_history": chat_history})
    return response["answer"], response["context"]

# ------------------------------------------
# Proactive Engagement: Periodic check-in function
def proactive_checkin():
    st.write("**Check-In:** How are you feeling today?")
    checkin_response = st.text_input("Your check-in response:")
    if checkin_response:
        st.write("Thanks for sharing. We're here for you!")
        st.session_state.user_profile['last_checkin'] = checkin_response

# ------------------------------------------
# Detailed Feedback Collection: Slider rating and additional comments
def collect_feedback():
    st.write("Please rate the response:")
    rating = st.slider("Rating", 1, 5, 3)
    comments = st.text_area("Additional Comments")
    if st.button("Submit Feedback", key="feedback_submit"):
        st.session_state.feedback = {'rating': rating, 'comments': comments}
        st.write("Thank you for your feedback!")

# ------------------------------------------
# Enhanced Logging: Log conversation entries along with sentiment scores
def log_conversation(role, message, sentiment=None):
    st.session_state.conversation_log.append({'role': role, 'message': message, 'sentiment': sentiment})

# ------------------------------------------
# Main Chat Function integrating all functionalities
def chat(chat_history, vectordb):
    # Display resource sections (Emergency, Health, Safety)
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.expander("🚨 Emergency Contacts", expanded=False):
            st.write("📞 Immediate Assistance:")
            for contact in WOMENS_RESOURCES["emergency_contacts"]:
                st.markdown(f"- {contact}")
    with col2:
        with st.expander("🏥 Health Resources", expanded=False):
            for category, link in WOMENS_RESOURCES["health_resources"].items():
                st.markdown(f"[{category}]({link})")
    with col3:
        with st.expander("🛡️ Safety Tools", expanded=False):
            for feature, status in WOMENS_RESOURCES["safety_features"].items():
                status_icon = "✅" if status else "🚧"
                st.markdown(f"{status_icon} {feature}")
    
    # Proactive check-in feature
    proactive_checkin()
    
    # Chat input field
    user_query = st.chat_input("Ask a question:")
    if user_query:
        # Anonymize the user input
        user_query = anonymize_text(user_query)
        
        # Toxicity check for user query
        toxicity_results = detoxify_model.predict(user_query)
        st.write("Debug - User Query Toxicity:", toxicity_results)
        if toxicity_results.get('toxicity', 0) > 0.5:
            st.write("Your message was flagged as toxic. Please rephrase and try again.")
            return chat_history
        
        # Check for emergency keywords and escalate if necessary
        emergency_keywords = ["suicide", "self-harm", "crisis", "immediately", "overdose", "end my life", "danger", "depressed"]
        if any(keyword in user_query.lower() for keyword in emergency_keywords):
            escalate_emergency(user_query)
        
        # Perform sentiment analysis
        sentiment = emotion_detector(user_query)[0]
        st.write(f"[Sentiment Analysis] Detected: {sentiment['label']} (score: {sentiment['score']:.2f})")
        log_conversation("Human", user_query, sentiment=sentiment)
        
        # Generate response using the retrieval chain
        response, context = get_response(user_query, chat_history, vectordb)
        toxicity_results_response = detoxify_model.predict(response)
        st.write("Debug - Response Toxicity:", toxicity_results_response)
        if toxicity_results_response.get('toxicity', 0) > 0.5:
            st.write("The generated response was flagged as toxic. Regenerating response.")
            response, context = get_response(user_query, chat_history, vectordb)
        
        log_conversation("AI", response)
        
        # Update chat history with the user query and chatbot response
        chat_history += [HumanMessage(content=user_query), AIMessage(content=response)]
        
        # Sidebar: Display source metadata for transparency
        with st.sidebar:
            metadata_dict = defaultdict(list)
            for doc in context:
                metadata = doc.metadata
                metadata_dict[metadata.get('source', 'Unknown')].append(str(metadata.get('page', 'N/A')))
            for source, pages in metadata_dict.items():
                st.write(f"Source: {source}")
                st.write(f"Pages: {', '.join(pages)}")
        
        # Contextual Resource Recommendations based on sentiment (example)
        if sentiment and sentiment.get('label').lower() in ['negative', 'sad', 'fear']:
            st.write("It seems you might be feeling down. Please consider these additional resources:")
            for contact in WOMENS_RESOURCES["emergency_contacts"]:
                st.markdown(f"- {contact}")
        
        # Detailed Feedback Collection
        collect_feedback()
    
    # Display full chat history
    for message in chat_history:
        role = "AI" if isinstance(message, AIMessage) else "Human"
        with st.chat_message(role):
            st.write(message.content)
    
    # Enhanced Analytics Dashboard in sidebar: Show conversation log
    with st.sidebar.expander("Conversation Analytics", expanded=False):
        st.write("Conversation Log:")
        st.write(st.session_state.conversation_log)
    
    return chat_history
