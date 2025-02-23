import streamlit as st
from collections import defaultdict
from detoxify import Detoxify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from utils.ner_utils import load_model_and_tokenizer, mask_names_and_orgs
import re

# Load environment variables
load_dotenv()

# -----------------------------
# Women's Resource Hub (Modular Configuration)
WOMENS_RESOURCES = {
    "emergency_contacts": ["National Helpline: 112", "Women's Commission: 123-456-7890"],
    "resources": {        "Meditation": "https://portal.wellbeats.com/?redirectTo=%2Fhome",
                          "Podcasts": "https://myhperewards.com/podcasts/index.html",
                          "Finance": "https://myhperewards.com/pdf/hpe-global-financial-resources.pdf",
                          "Mental health": "https://myhperewards.com/webinars/index.html",
                          "Music": "https://www.headspace.com/login?redirectOnSuccess=https%3A%2F%2Fmy.headspace.com%2Fmodes%2Fmeditate"},
    "safety_features": {"Location Sharing": True, "Emergency SOS": False}
}

# -----------------------------
# Caching resource-intensive models to improve performance.
@st.cache_resource(show_spinner=False)
def load_detoxify_model():
    return Detoxify('original')

@st.cache_resource(show_spinner=False)
def load_emotion_detector():
    # For example, using spaCy or a Hugging Face pipeline could be integrated here.
    # FIXME for future
    from transformers import pipeline
    return pipeline("sentiment-analysis")

if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {'name': None, 'preferred_language': 'en', 'last_checkin': None}
    
detoxify_model = load_detoxify_model()
emotion_detector = load_emotion_detector()

# -----------------------------
# Function to anonymize names from user input.
def anonymize_names(text):
    # Using regex for simple anonymization; consider integrating spaCy for advanced NER.
    # FIXME for future
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    text = re.sub(r'\b\d{10,}\b', '[PHONE]', text)
    ner_model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
    ner_model, ner_tokenizer = load_model_and_tokenizer(ner_model_name)
    id2label = ner_model.config.id2label
    text = mask_names_and_orgs(text, ner_model, ner_tokenizer, id2label)
    return text

# -----------------------------
# Anonymize text (including emails, phone numbers, and names)
def anonymize_text(text):
    #FIXME for future
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    text = re.sub(r'\b\d{10,}\b', '[PHONE]', text)
    text = anonymize_names(text)
    return text

# -----------------------------
def get_context_retriever_chain(vectordb):
    # Cached LLM model initialization.
    if 'llm' not in st.session_state:
        st.session_state.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.3,  # Slightly higher for empathy
            convert_system_message_to_human=True
        )
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    system_prompt = (
        "You are a supportive women's health assistant. Provide:\n"
        "1. Gender-sensitive responses with a trauma-informed approach\n"
        "2. Clarify ambiguous health queries before answering\n"
        "3. Prioritize safety resources when detecting distress signals\n"
        "4. Maintain strict confidentiality boundaries\n"
        "Please respond with plain text, without using  any markdown formatting"
        "Strictly don't include * in response"
        "Context: {context}"
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
    print(response)
    # return anonymize_text(response["answer"]), response["context"]
    return (response["answer"]), response["context"]


# Proactive Engagement: Periodic check-in function
def proactive_checkin():
    st.write("**Check-In:** How are you feeling today?")
    checkin_response = st.text_input("Your check-in response:")
    if checkin_response:
        st.write("Thanks for sharing. We're here for you!")
        st.session_state.user_profile['last_checkin'] = checkin_response

def handle_greetings_and_thanks(question):
    greetings = ["hi", "hello", "hey"]
    thanks = ["thanks", "thank you"]
    response= None
    if question:
        if question.lower() in greetings:
            response = "Hello! How can I assist you today?"
        elif question.lower() in thanks:
            response = "You're welcome! If you have any other questions, feel free to ask."
        else:
            response= None
    else:
        response = None
    return response     
    
def chat(chat_history, vectordb):
    # Display resource sections
    st.title("🌸 EmpowHer - Women's Support Chat")
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.expander("🚨 Emergency Contacts", expanded=False):
            st.write("📞 Immediate Assistance:")
            for contact in WOMENS_RESOURCES["emergency_contacts"]:
                st.markdown(f"- {contact}")
    with col2:
        with st.expander("💡 Resources", expanded=False):
            for category, link in WOMENS_RESOURCES["resources"].items():
                st.markdown(f"[{category}]({link})")
    with col3:
        with st.expander("🛡️ Safety Tools", expanded=False):
            for feature, status in WOMENS_RESOURCES["safety_features"].items():
                status_icon = "✅" if status else "🚧"
                st.markdown(f"{status_icon} {feature}")
    
    # Proactive check-in feature -- Something to think about
    #proactive_checkin()
    
    # Chat input field
    user_query = st.chat_input("Ask a question:")
    response = handle_greetings_and_thanks(user_query)

    if (user_query is not None and user_query != "") :
        if response:
            chat_history += [HumanMessage(content=user_query), AIMessage(content=response)]
        else:

            # Anonymize the user input  

            user_query = anonymize_text(user_query)
            # Check toxicity of user query
            toxicity_results = detoxify_model.predict(user_query)
            #st.write("Debug - User Query Toxicity:", toxicity_results)
            if toxicity_results.get('toxicity', 0) > 0.5:
                response= "Your message was flagged as toxic. Please rephrase and try again."
                chat_history += [HumanMessage(content=user_query), AIMessage(content=response)]
                # st.write("Your message was flagged as toxic. Please rephrase and try again.")
            else:
                # Optional: Use emotion detection to adjust responses or provide resources
                sentiment = emotion_detector(user_query)[0]
                st.write(f"[Sentiment Analysis] Detected: {sentiment['label']} (score: {sentiment['score']:.2f})")
                
                # Generate response based on user's query, chat history, and vectorstore
                response, context = get_response(user_query, chat_history, vectordb)
                toxicity_results_response = detoxify_model.predict(response)
                st.write("Debug - Response Toxicity:", toxicity_results_response)
                if toxicity_results_response.get('toxicity', 0) > 0.5:
                    st.write("The generated response was flagged as toxic. Regenerating response.")
                    response, context = get_response(user_query, chat_history, vectordb)
                
                # Append user query and chatbot response to chat history.
                chat_history += [HumanMessage(content=user_query), AIMessage(content=response)]
                
                # Sidebar: Display source metadata for transparency.
                with st.sidebar:
                    metadata_dict = defaultdict(list)
                    for doc in context:
                        metadata = doc.metadata
                        metadata_dict[metadata.get('source', 'Unknown')].append(str(metadata.get('page', 'N/A')))
                    for source, pages in metadata_dict.items():
                        st.write(f"Source: {source}")
                        st.write(f"Pages: {', '.join(pages)}")
    
    # # Display the full chat history.
    # for message in chat_history:
    #     role = "AI" if isinstance(message, AIMessage) else "Human"
    #     with st.chat_message(role):
    #         st.write(message.content)
    # Display full chat history with bot responses on the left and user input on the right
     # Chat history display
    for message in chat_history:
        if isinstance(message, AIMessage):
            with st.container():
                st.markdown(
                    f"<div style='background-color:#f0f0f0; padding:10px; border-radius:12px; text-align:left;'>"
                    f"🤖 <strong>Bot:</strong> {message.content}"
                    f"</div>", unsafe_allow_html=True
                )
        else:
            with st.container():
                st.markdown(
                    f"<div style='background-color:#e0c7f7; padding:10px; border-radius:12px; text-align:right;'>"
                    f"👩 <strong>You:</strong> {message.content}"
                    f"</div>", unsafe_allow_html=True
                )
    
    # Optional: Allow users to provide feedback for continuous improvement.
    if any(isinstance(msg, AIMessage) for msg in chat_history):
        feedback = st.radio("Was this response helpful?", ["Yes", "No"], index=0, key="feedback")
        if st.button("Submit Feedback"):
            # Log or store the feedback as needed.
            st.write("Thank you for your feedback!")
    st.markdown("---")
    st.caption("EmpowHer can make mistakes sometimes")

    
    return chat_history
