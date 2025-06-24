import streamlit as st
import os
import yaml
import base64
from streamlit_chat import message
from utils.ChatTooling import ChatTooling
import pandas as pd
from streamlit_cookies_controller import CookieController
import time

# Constants remain the same...
COOKIE_PREFIX = "quiz_"
COOKIE_ACRONYM = f"{COOKIE_PREFIX}acronym"
COOKIE_SAVE_SCORE = f"{COOKIE_PREFIX}save_score"

# UserPreferences class remains the same...
class UserPreferences:
    def __init__(self, cookie_controller):
        self.controller = cookie_controller
        self.acronym = ""
        self.save_score = False

    def load_from_cookies(self):
        try:
            self.acronym = self.controller.get(COOKIE_ACRONYM) or ""
            save_score_str = self.controller.get(COOKIE_SAVE_SCORE)            
            self.save_score = str(save_score_str).lower() == "true" if save_score_str else False
        except Exception as e:
            st.error(f"Error loading preferences: {e}")
            st.write(f"Error details: {e}")

    def save_to_cookies(self):
        try:
            self.controller.set(COOKIE_ACRONYM, self.acronym)
            self.controller.set(COOKIE_SAVE_SCORE, str(self.save_score).lower())
        except Exception as e:
            st.error(f"Error saving preferences: {e}")

def display_messages():
    if "messages" in st.session_state and st.session_state["messages"]:
        st.subheader("Chat")
        for i, (msg, is_user) in enumerate(st.session_state["messages"]):
            message(msg, is_user=is_user, key=f"msg_{i}")

# Utility functions remain the same...
def ingest_lecture_summary(file_path):
    try:
        with open(file_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        return "No lecture summary found."
    except Exception as e:
        return f"An error occurred: {e}"

def get_txt_files(folder_path):
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    full_paths = [os.path.join(folder_path, f) for f in txt_files]
    sorted_files = sorted(full_paths, key=os.path.getmtime, reverse=True)
    return [os.path.basename(f) for f in sorted_files]

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
        
        if st.session_state["assistant"].get_question_count() == num_questions + 1:
            completion_message = "You have completed the quiz. We're now analyzing your answers. This may take a moment..."
            st.session_state["messages"].append((completion_message, False))
            st.session_state["quiz_completed"] = True
        else:
            st.session_state["messages"].append((agent_text, False))

def generate_final_rating():
    final_rating = st.session_state["assistant"]._generate_final_rating()
    st.session_state["messages"].append((final_rating, False))
    st.session_state["rating_complete"] = True

def initialize_chat(file_path, chapter, preferences):
    """Initialize chat with the first message"""
    assistant = ChatTooling()
    assistant.set_chapter(chapter)
    
    # Apply saved preferences
    if preferences.acronym:
        assistant.set_user_acronym(preferences.acronym)
    assistant.activate_submission_to_leaderbord(preferences.save_score)
    
    # Set up the system prompt
    lecture_summary = ingest_lecture_summary(file_path)
    assistant.set_system_prompt(lecture_summary)
    
    # Initialize the first message
    first_message = assistant.query("Please start the quiz.")
    
    return assistant, [(first_message, False)]

def initialize_session(cookie_controller):
    if "initialized" not in st.session_state:
        st.session_state["initialized"] = True
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "selected_file" not in st.session_state:
        st.session_state["selected_file"] = None
    if "selected_chapter" not in st.session_state:
        st.session_state["selected_chapter"] = None
    if "saved_settings" not in st.session_state:
        st.session_state["saved_settings"] = False
    if "assistant" not in st.session_state:
        st.session_state["assistant"] = None
    if "quiz_completed" not in st.session_state:
        st.session_state["quiz_completed"] = False
    if "rating_complete" not in st.session_state:
        st.session_state["rating_complete"] = False
    if "user_preferences" not in st.session_state:
        # wait for streamlit to get initialized
        time.sleep(0.5)
        st.session_state["user_preferences"] = UserPreferences(cookie_controller)
        # Load preferences from cookies
        st.session_state["user_preferences"].load_from_cookies()
     
def render_sidebar(assistant, cookie_controller):
    st.sidebar.header("Leaderboard")
    
    preferences = st.session_state.user_preferences
    
    user_acronym = st.sidebar.text_input(
        "Enter your 3-letter acronym:",
        max_chars=3,
        value=preferences.acronym
    )
       
    checked = st.sidebar.checkbox("Submit score", value=preferences.save_score)
    
    if st.sidebar.button("Save settings"):
        if checked and not user_acronym:
            st.sidebar.error("Please enter your 3-letter acronym.")
        else:
            preferences.acronym = user_acronym
            preferences.save_score = checked
            preferences.save_to_cookies()
            
            if checked:
                assistant.set_user_acronym(user_acronym)
            assistant.activate_submission_to_leaderbord(checked)
            st.session_state["saved_settings"] = checked
            st.sidebar.success("Settings saved.")
    
    df = assistant.get_leaderboard()
    st.sidebar.data_editor(df, hide_index=True)
    
    st.sidebar.header("Save Chat History")
    if st.sidebar.button("Download Chat History"):
        chat_content = save_chat_history(st.session_state["messages"])
        b64 = base64.b64encode(chat_content.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="chat_history.txt">Download chat history</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
        st.sidebar.success("Chat history ready for download")

def format_func(tuple_item):
    return tuple_item[1]

def page(config):
    st.set_page_config(
        page_title=config['app_title'],
        layout='wide',
        initial_sidebar_state='auto'
    )
    
    cookie_controller = CookieController()
    st.title(config['app_title'])
    
    initialize_session(cookie_controller)
    
    txt_files = get_txt_files("lectures")
    txt_files = [os.path.join("lectures", f) for f in txt_files]
    st.write("Please select a lecture summary to start the quiz. (Note: You can save the chat history at the end, see left sidebar.)")
    
    numbered_files = list(enumerate(txt_files))
    selected = st.selectbox(
        "Choose a file",
        options=numbered_files,
        format_func=format_func,
        key="file_selector"
    )
    
    selected_file = selected[1]
    
    # Handle file selection and chat initialization
    if not st.session_state["selected_file"] or selected_file != st.session_state["selected_file"]:
        time.sleep(0.5)
        st.session_state["selected_file"] = selected_file
        st.session_state["selected_chapter"] = len(numbered_files) - selected[0]
        
        # Reset quiz states when selecting a new file
        st.session_state["quiz_completed"] = False
        st.session_state["rating_complete"] = False
        
        # Initialize chat and store results
        assistant, messages = initialize_chat(
            selected_file, 
            st.session_state["selected_chapter"],
            st.session_state["user_preferences"]
        )

        st.session_state["assistant"] = assistant
        st.session_state["messages"] = messages
        st.rerun()
    
    # Display chat interface
    display_messages()
    
    # Show sidebar if assistant is initialized
    if st.session_state["assistant"]:
        render_sidebar(st.session_state["assistant"], cookie_controller)
    
    # Handle user input
    user_input = st.chat_input("Type your message here...")
    if user_input:
        process_input(user_input, config['num_questions'])
        st.rerun()
    
    # Handle quiz completion
    if st.session_state.get("quiz_completed", False) and not st.session_state.get("rating_complete", False):
        with st.spinner("Generating final rating..."):
            generate_final_rating()
        st.rerun()

if __name__ == "__main__":
    with open("config.yaml") as file:
        config = yaml.safe_load(file)
    page(config)