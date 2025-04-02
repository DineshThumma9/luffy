import streamlit as st
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from logic import get_vecstore_from_url, get_response

st.set_page_config(page_title="RAG", page_icon=":books:", layout="wide")
st.title("RAG")
st.markdown("## Retrieval Augmented Generation")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL", placeholder="Enter the URL of the website to scrape")

# Ensure a website URL is provided
if not website_url:
    st.info("Please enter a website URL")
else:
    st.success(f"Website URL: {website_url}")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            SystemMessage(content="You are a helpful AI chatbot."),
            AIMessage(content="Hello! How can I assist you today?"),
        ]

    # Initialize vector store if not already set
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vecstore_from_url(website_url)

    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)

    # Input field at the bottom
    user_input = st.chat_input("Type your message...")

    if user_input:
        # Append user input to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_input))

        # Create a placeholder for the AI response
        response_placeholder = st.empty()

        # Show spinner while generating response
        with st.spinner("Thinking..."):
            response_text = get_response(user_input)  # Get AI response

        # Update UI with the AI response
        response_placeholder.markdown(response_text)
        st.session_state.chat_history.append(AIMessage(content=response_text))

        # Rerun to refresh UI and move input to the bottom
        st.rerun()
