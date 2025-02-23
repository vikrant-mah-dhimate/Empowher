import streamlit as st
from collections import defaultdict
from detoxify import Detoxify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv


# Women's Resource Hub (Modular Configuration)
WOMENS_RESOURCES = {
    "emergency_contacts": ["National Helpline: 112", "Women's Commission: 123-456-7890"],
    "health_resources": {"Physical Health": "https://example.com", "Mental Health": "https://example.org"},
    "safety_features": {"Location Sharing": True, "Emergency SOS": False}
}

def get_context_retriever_chain(vectordb):
    """
    Create a context retriever chain for generating responses based on the chat history and vector database

    Parameters:
    - vectordb: Vector database used for context retrieval

    Returns:
    - retrieval_chain: Context retriever chain for generating responses
    """
    # Load environment variables (gets api keys for the models)
    load_dotenv()
    # Cached model initialization
    if 'llm' not in st.session_state:
        st.session_state.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.3,  # Slightly higher for empathy
            convert_system_message_to_human=True
        )
    retriever = vectordb.as_retriever()
    system_prompt = """You are a supportive women's health assistant. Provide:
    1. Gender-sensitive responses with trauma-informed approach
    2. Clarify ambiguous health queries before answering
    3. Prioritize safety resources when detecting distress signals
    4. Maintain strict confidentiality boundaries
    Context: {context}"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
                                              
    return create_retrieval_chain(
        vectordb.as_retriever(search_kwargs={"k": 5}),
        create_stuff_documents_chain(st.session_state.llm, prompt)
    )

def get_response(question, chat_history, vectordb):
    """
    Generate a response to the user's question based on the chat history and vector database

    Parameters:
    - question (str): The user's question
    - chat_history (list): List of previous chat messages
    - vectordb: Vector database used for context retrieval

    Returns:
    - response: The generated response
    - context: The context associated with the response
    """
    chain = get_context_retriever_chain(vectordb)
    response = chain.invoke({"input": question, "chat_history": chat_history})
    return response["answer"], response["context"]

def chat(chat_history, vectordb):
    """
    Handle the chat functionality of the application

    Parameters:
    - chat_history (list): List of previous chat messages
    - vectordb: Vector database used for context retrieval

    Returns:
    - chat_history: Updated chat history
    """
    # Women's Resources Header
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
                
    user_query = st.chat_input("Ask a question:")
    if user_query is not None and user_query != "":
        toxicity_results = Detoxify('original').predict(user_query)
        print("QUERY:", toxicity_results)
        if toxicity_results['toxicity'] > 0.5:
            st.write("Your message was flagged as toxic. Please rephrase and try again.")
        else:
            # Generate response based on user's query, chat history and vectorstore
            response, context = get_response(user_query, chat_history, vectordb)
            toxicity_results = Detoxify('original').predict(response)
            print("RESPONSE:", toxicity_results)
            if toxicity_results['toxicity'] > 0.5:
                st.write("The generated response was flagged as toxic. Regenerating response.")
                response, context = get_response(user_query, chat_history, vectordb)
            # Update chat history. The model uses up to 10 previous messages to incorporate into the response
            chat_history = chat_history + [HumanMessage(content=user_query), AIMessage(content=response)]
            # Display source of the response on sidebar
            with st.sidebar:
                metadata_dict = defaultdict(list)
                for metadata in [doc.metadata for doc in context]:
                    metadata_dict[metadata['source']].append(metadata['page'])
                for source, pages in metadata_dict.items():
                    st.write(f"Source: {source}")
                    st.write(f"Pages: {', '.join(map(str, pages))}")
    # Display chat history
    for message in chat_history:
        with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
            st.write(message.content)
    return chat_history