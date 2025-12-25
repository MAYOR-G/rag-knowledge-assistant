import streamlit as st
import time

st.set_page_config(page_title="Advanced RAG Assistant", page_icon="🧠")
st.title("🧠 Advanced RAG (Hybrid + Memory + Citations)")

# Debug Message 1
status_placeholder = st.empty()
status_placeholder.info("Initializing System... Please wait.")

# Import Logic (Inside try-block to catch import errors)
try:
    from rag_advanced import get_advanced_chain
    from langchain_core.messages import HumanMessage, AIMessage
except ImportError as e:
    st.error(f"CRITICAL IMPORT ERROR: {e}")
    st.stop()

# 1. Load Chain with Caching
@st.cache_resource
def load_chain():
    return get_advanced_chain()

try:
    chain = load_chain()
    status_placeholder.success("System Ready: Hybrid Search & Memory Active")
    time.sleep(1) # Show success briefly
    status_placeholder.empty() # Clear the message
except Exception as e:
    status_placeholder.error(f"Failed to load system: {e}")
    st.stop()

# 2. Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 3. Display Chat
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# 4. Input Loop
if prompt := st.chat_input("Ask a question..."):
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = chain.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history
                })
                
                answer = response["answer"]
                sources = response["context"]

                st.markdown(answer)

                # Citations
                with st.expander("📚 View Sources (Citations)"):
                    for i, doc in enumerate(sources):
                        page = doc.metadata.get('page', 'Unknown')
                        source_file = doc.metadata.get('source', 'Unknown')
                        snippet = doc.page_content[:100].replace("\n", " ")
                        st.markdown(f"**{i+1}. Page {page}** ({source_file})")
                        st.caption(f"_{snippet}..._")
                
                st.session_state.chat_history.append(AIMessage(content=answer))
            
            except Exception as e:
                st.error(f"Error generating answer: {e}")