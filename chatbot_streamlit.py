import streamlit as st
import requests

# Constants
base_url = 'http://localhost:9500'

# Page title
st.set_page_config(page_title="Chatbot Multilingue", layout="centered")
st.title("💬 Multilanguage Chatbot with Sentiment & Translation")

# Init chat list
if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("Write your message in french :", "")

# Send button
if st.button("Send") and user_input:

    try: 
        response = requests.post(base_url+"/process/", json={"text": user_input})
        data = response.json()

        st.session_state.messages.append({
            "user": user_input,
            "translation": data["translation"],
            "sentiment": data["sentiment"],
            "bot": data["response"]
        })

    except req.exceptions.RequestException as e:
        st.error(f"Api connection error : {e}")
    except Exception as e :
        st.error(f"An error occured: {e}")

# History
for chat in reversed(st.session_state.messages):
    st.markdown("**🧑 User :** " + chat["user"])
    st.markdown("**🔁 Translation :** " + chat["translation"])
    st.markdown("**💡 Sentiment :** " + chat["sentiment"])
    st.markdown("**🤖 Chatbot :** " + chat["bot"])
    st.markdown("---")
