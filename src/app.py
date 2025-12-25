import streamlit as st
from rag import get_rag_chain  # Import your logic

st.set_page_config(page_title="RAG Knowledge Assistant", page_icon="🤖")

st.title("🤖 Enterprise Knowledge Assistant")

# 1. The Cache Decorator
# This function runs ONCE. If you refresh the page, it grabs the saved chain from memory.
@st.cache_resource
def load_chain():
    print("Initializing RAG Chain (This should only happen once)...")
    return get_rag_chain()

# 2. Load the chain
try:
    chain = load_chain()
    st.success("System Ready: Connected to Knowledge Base")
except Exception as e:
    st.error(f"Failed to load system: {e}")
    st.stop()

# 3. Session State (Memory for the chat history)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Handle User Input
if prompt := st.chat_input("Ask a question about your document..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke(prompt)
            st.markdown(response)
    
    # Add AI message to history
    st.session_state.messages.append({"role": "assistant", "content": response})