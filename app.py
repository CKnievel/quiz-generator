import streamlit as st
import os
import yaml
import base64
from streamlit_chat import message
from utils.ChatTooling import ChatTooling

def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))

def ingest_lecture_summary(file_path):
    try:
        with open(file_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        return "No lecture summary found."
    except Exception as e:
        return f"An error occurred: {e}"

def get_txt_files(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith('.txt')]

def save_chat_history(messages):
    content = ""
    for msg, is_user in messages:
        sender = "User" if is_user else "Assistant"
        content += f"{sender}: {msg}\n\n"
    return content

def process_input(user_input, num_questions):
    if user_input and len(user_input.strip()) > 0:
        st.session_state["messages"].append((user_input, True))
        
        with st.spinner("Thinking..."):
            agent_text = st.session_state["assistant"].query(user_input)                
        
        if st.session_state["assistant"].get_question_count() == num_questions + 1: # number of questions + 1 completion message
            completion_message = "You have completed the quiz. We're now analyzing your answers. This may take a moment..."
            st.session_state["messages"].append((completion_message, False))
            st.session_state["quiz_completed"] = True
        else:
            st.session_state["messages"].append((agent_text, False))

def generate_final_rating():
    final_rating = st.session_state["assistant"]._generate_final_rating()
    st.session_state["messages"].append((final_rating, False))
    st.session_state["rating_complete"] = True

def initialize_session(lec_summary_file):
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "assistant" not in st.session_state:
        st.session_state["assistant"] = ChatTooling()
        lecture_summary = ingest_lecture_summary(lec_summary_file)
        st.session_state["assistant"].set_system_prompt(lecture_summary)
        agent_text = st.session_state["assistant"].query("Please start the quiz.")
        st.session_state["messages"].append((agent_text, False))
    if "quiz_completed" not in st.session_state:
        st.session_state["quiz_completed"] = False
    if "rating_complete" not in st.session_state:
        st.session_state["rating_complete"] = False

def page(config):  
    
    st.set_page_config(page_title=config['app_title'], layout='wide', initial_sidebar_state='collapsed')
    st.title(config['app_title'])
    txt_files = get_txt_files("lectures")
    txt_files = [os.path.join("lectures", f) for f in txt_files]
    st.write("Please select a lecture summary to start the quiz. (Note: You can save the chat history at the end, see left sidebar.)")
    selected_file = st.selectbox("Choose a file", txt_files)
   
    if selected_file:
        initialize_session(selected_file)
   
    display_messages()
   
    user_input = st.chat_input("Type your message here...")
    if user_input:
        process_input(user_input, config['num_questions'])
        st.rerun()

    if st.session_state["quiz_completed"] and not st.session_state["rating_complete"]:
        with st.spinner("Generating final rating..."):
            generate_final_rating()
        st.rerun()

    # Add download button for chat history
    st.sidebar.header("Save Chat History")
    if st.sidebar.button("Download Chat History"):
        chat_content = save_chat_history(st.session_state["messages"])
        b64 = base64.b64encode(chat_content.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="chat_history.txt">Download chat history</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
        st.sidebar.success("Chat history ready for download")
    
if __name__ == "__main__":
    with open("config.yaml") as file: 
            config = yaml.safe_load(file)
    page(config)