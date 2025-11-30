import streamlit as st
from ai_tutor.rag_youtube import YouTubeRAG


# --------------------------------
# Streamlit Page Setup
# --------------------------------
st.set_page_config(page_title="YouTube Tutor", layout="wide")

# Global state
if "video_id" not in st.session_state:
    st.session_state.video_id = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

yt_rag = YouTubeRAG(
    llm_model="llama3.2:1b",    # Use small model
    whisper_size="base",         # You already use this
)


# --------------------------------
# Sidebar Search
# --------------------------------
st.sidebar.header("Search YouTube Topic")
query = st.sidebar.text_input("Enter topic", "")

if st.sidebar.button("Search"):
    if not query.strip():
        st.sidebar.error("Please enter a topic.")
    else:
        results = yt_rag.search_youtube(query)
        if not results:
            st.sidebar.error("No results found.")
        else:
            top = results[0]
            st.session_state.video_id = top["video_id"]
            st.sidebar.success(f"Loaded: {top['title']}")


# --------------------------------
# Main Page Layout
# --------------------------------
left, right = st.columns([2, 1])

# --------------------------------
# LEFT SIDE → Show Video
# --------------------------------
with left:
    st.title("📺 YouTube Classroom Tutor")

    if st.session_state.video_id:
        vid = st.session_state.video_id
        st.video(f"https://www.youtube.com/watch?v={vid}")
        st.info("Ask anything about this video on the chatbox.")

    else:
        st.write("🔍 Search for a YouTube topic using the sidebar.")


# --------------------------------
# RIGHT SIDE → Chat Interface
# --------------------------------
with right:
    st.subheader("💬 Ask the Tutor")

    # Show chat history
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").write(msg)
        else:
            st.chat_message("assistant").write(msg)

    # User input
    prompt = st.chat_input("Ask your question...")

    if prompt and st.session_state.video_id:
        # Save user message
        st.session_state.chat_history.append(("user", prompt))
        st.chat_message("user").write(prompt)

        with st.spinner("Thinking..."):
            try:
                answer = yt_rag.ask_video(prompt, st.session_state.video_id)
            except Exception as e:
                answer = f"⚠️ Error: {str(e)}"

        # Save assistant reply
        st.session_state.chat_history.append(("assistant", answer))
        st.chat_message("assistant").write(answer)

    elif prompt and not st.session_state.video_id:
        st.warning("Please search and load a YouTube video first.")
