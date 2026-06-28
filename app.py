import streamlit as st

from rag import ask_hr_question

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="💼",
    layout="wide"
)

# ==========================================
# Title
# ==========================================

st.title("💼 Zyro Dynamics HR Help Desk")

st.markdown(
    """
Ask any HR-related question about:

- Leave Policy
- Work From Home
- Compensation & Benefits
- Employee Handbook
- Performance Reviews
- Travel & Expense
- IT & Data Security
- POSH Policy
- Onboarding & Separation

The assistant answers only from Zyro Dynamics HR policy documents.
"""
)

# ==========================================
# Chat History
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# User Input
# ==========================================

question = st.chat_input("Ask your HR question...")

if question:

    # Show user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Searching HR policies..."):

            response = ask_hr_question(question)

            st.markdown(response["answer"])

            if response["sources"]:

                st.markdown("---")
                st.markdown("### 📄 Sources")

                for source in response["sources"]:

                    st.write(
                        f"**{source['document']}** — Page {source['page']}"
                    )

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response["answer"]
        }
    )