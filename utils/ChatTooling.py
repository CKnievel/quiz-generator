from langchain_community.chat_models import ChatOllama
from langchain.prompts import (ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate)
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langsmith import traceable
import yaml
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
import random
import string
from dotenv import load_dotenv
import os 
class ChatTooling:
    def __init__(self) -> None:
        self.config = self.load_config(Path("utils/config_chat.yaml"))
        self.prompts = self.load_prompts("prompts")

        if self.config['llm_model'] == 'anthropic':
            load_dotenv()
            MODEL = "claude-3-5-sonnet-20240620"
            ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
            self.model = ChatAnthropic(
                anthropic_api_key=ANTHROPIC_API_KEY, 
                model_name=MODEL, 
                temperature=self.config.get('temperature', 0.5), 
                max_tokens=3000
            )

        else:
            self.model = ChatOllama(
                model=self.config['llm_model'],
                temperature=self.config.get('temperature', 0.5),
                top_k=self.config.get('top_k', 5), 
                top_p=self.config.get('top_p', 0.8),
                keep_alive = 3600,
            )
        
        self.system_prompt = self.prompts['system_prompt']
        
        self.conversation_buffer = ConversationBufferMemory(
            memory_key="history",
            return_messages=True
        )

        self._initialize_chain()

        self.question_count = 0

        # leaderboard variables
        self.score_per_session = 0.0
        self.bsubmit_score_to_leaderboard = False
        self.user_acronym = ''.join(random.choices(string.ascii_uppercase, k=3))
        self.chapter = None

    def load_config(self, config_path):
        """ Helper function to load the configuration file. """        
        with open(config_path) as file: 
            return yaml.safe_load(file)

    def activate_submission_to_leaderbord(self, bsubmit : bool)-> None:
        self.bsubmit_score_to_leaderboard = bsubmit

    def set_user_acronym(self, user_acronym):
        self.user_acronym = user_acronym

    def set_chapter(self, chapter):
        self.chapter = chapter

    def load_prompts(self, prompts_path):
        """ Helper function to load the prompt files. """
        prompts = {}
        prompts_dir = Path(prompts_path)       
        for prompt_file in prompts_dir.glob('*.txt'):
            with open(prompt_file, 'r') as file:
                prompts[prompt_file.stem] = file.read().strip()
        return prompts

    @traceable
    def _initialize_chain(self):
        """ Initializes or reinitializes the LLMChain with the current system prompt. """
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
        
        self.chain = self.prompt_template | self.model            

    def set_system_prompt(self, context):
        self.system_prompt = f"""{self.system_prompt}\n\nContext: {context}"""
        self._initialize_chain()

    @traceable
    def _serialize_messages(self, messages):
        """Helper function to serialize HumanMessage and AIMessage objects."""
        serialized_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                serialized_messages.append({"type": "human", "content": message.content})
            elif isinstance(message, AIMessage):
                serialized_messages.append({"type": "ai", "content": message.content})
            
        return serialized_messages

    @traceable
    def query(self, user_input):                
        # Serialize the conversation history
        serialized_history = self._serialize_messages(self.conversation_buffer.chat_memory.messages)
        
        # Increment the question count
        self.question_count += 1

        # Run the chain to get the response
        response = self.chain.invoke({
            "history": serialized_history,
            "input": user_input
        })
        
        # Manually add the human input to the memory
        self.conversation_buffer.chat_memory.add_message(HumanMessage(content=user_input))

        # Manually add the AI response to the memory
        self.conversation_buffer.chat_memory.add_message(response)

        return response.content

    def get_question_count(self):
        return self.question_count
    
    @traceable
    def _generate_final_rating(self):

        # Serialize the conversation history        
        serialized_history = self._serialize_messages(self.conversation_buffer.chat_memory.messages)

        # Define reference examples for rating user responses
        with open('prompts/example_ratings.json','r') as f:
            examples = json.load(f)

         # Create a string of examples
        examples_str = "\n\n".join([
            f"Example {i+1}:\nQuestion: {ex['question']}\nAnswer: {ex['answer']}\nRating: {ex['rating']}\nExplanation: {ex['explanation']}"
            for i, ex in enumerate(examples)
        ])

        # Define the prompt template
        system_template = SystemMessagePromptTemplate.from_template(self.prompts['system_prompt_rating'])
        system_template = system_template.format(examples=examples_str)

        user_template = self.prompts['user_prompt_rating']

        prompt = ChatPromptTemplate.from_messages([
            system_template,
            HumanMessagePromptTemplate.from_template(user_template)
        ])


        # Iterate over the serialized history to pair AI questions with human answers
        message_answer_pairs = []
        rating_per_qa = []
        score_per_qa = []
        for i, message in enumerate(serialized_history):

            # skip the first message
            if i == 0:
                continue

            # Process AI messages
            if message["type"] == "ai":
                message_answer_pairs.append((message["content"], None))  # Placeholder for human answer

            # Process human messages
            elif message["type"] == "human" and message_answer_pairs:
                # Update the last AI message's tuple with the human answer
                message_answer_pairs[-1] = (message_answer_pairs[-1][0], message["content"])               
                    
                # Construct the input text for the final prompt
                question = message_answer_pairs[-1][0]
                answer = message_answer_pairs[-1][1]
                
                # Format the messages
                formatted_messages = prompt.format_messages(
                    examples=examples_str,
                    question=question,
                    answer=answer
                )

                rating = self.model.invoke(formatted_messages)

                # Store rating for this question-answer pair
                rating_per_qa.append(rating.content)

                bfoundScore, score = self.extract_and_validate_rating(rating.content)

                if bfoundScore:
                    score_per_qa.append(score)

                # Clear the list after processing a full AI-human pair
                message_answer_pairs.clear()

        # if rating and score are of equal length, then all scores were found
        if len(rating_per_qa) == len(score_per_qa):
            if self.bsubmit_score_to_leaderboard:
                self.score_per_session = self.calculate_score(score_per_qa)
                self.add_score_to_leaderboard(self.user_acronym, self.chapter)
   

        # Format the array into a string
        rating_per_qa = "\n".join([f"Question {i+1}: {rating} \n\n" for i, rating in enumerate(rating_per_qa)])

        return rating_per_qa

    def extract_and_validate_rating(self, text):
        # Convert to lowercase to make it case-insensitive
        text_lower = text.lower()
        if 'rating' not in text_lower:
            return False, None
        
        # Find the position of 'rating'
        rating_pos = text_lower.find('rating')
        
        # Get the substring starting from 'rating'
        remaining_text = text[rating_pos:]
        
        # Find first number after 'rating'
        # Skip non-digit characters until we find a digit or negative sign
        i = 6  # length of 'rating'
        while i < len(remaining_text):
            if remaining_text[i].isdigit() or remaining_text[i] == '-':
                # Found start of number, extract until non-digit
                num_start = i
                i += 1
                while i < len(remaining_text) and (remaining_text[i].isdigit() or remaining_text[i] == '.'):
                    i += 1
                try:
                    score = float(remaining_text[num_start:i])
                    return True, score
                except ValueError:
                    return False, None
            i += 1
        
        return False, None

    def reset_conversation(self):
        self.conversation_buffer.clear()
        self.question_count = 0

    # get leaderboard
    def get_leaderboard(self, csv_file = 'utils/leaderboard.csv') -> pd.DataFrame:
        try:
            df_leaderboard = pd.read_csv(csv_file, header = None)
            df_leaderboard.columns = ['Username', 'Score', 'Chapter', 'Submission Time']
            df_leaderboard = df_leaderboard.sort_values(by = 'Chapter', ascending = False)
        except pd.errors.EmptyDataError:
            df_leaderboard = pd.DataFrame(columns = ['Username', 'Score', 'Chapter', 'Submission Time'])
        return df_leaderboard

    # add score to leaderboard
    def add_score_to_leaderboard(self, username, chapter, csv_file = 'utils/leaderboard.csv') -> None:
        try:
            df_ldb = self.get_leaderboard(csv_file)
            #check for empty dataframe
            if df_ldb.empty:
                # raise EmptyDataError to handle empty dataframe
                raise pd.errors.EmptyDataError
            
            # check if lowercase username already exists
            if df_ldb['Username'].str.lower().str.contains(username.lower()).any():
                # check if chapter of this user is already in the leaderboard
                if df_ldb[(df_ldb['Username'].str.lower() == username.lower()) & (df_ldb['Chapter'] == chapter)].empty:
                    # add new row
                    new_row = {'Username': username, 'Score': self.score_per_session, 'Chapter': chapter, 'Submission Time': datetime.now().strftime("%Y-%m-%d")}                    
                    df_ldb = pd.concat([df_ldb, pd.DataFrame([new_row])], ignore_index=True)
                    df_ldb.to_csv(csv_file, index = False, header = False)
                else:
                    # update score only if new score is higher
                    new_score = self.score_per_session
                    if new_score > df_ldb[(df_ldb['Username'].str.lower() == username.lower()) & (df_ldb['Chapter'] == chapter)]['Score'].values[0]:
                        df_ldb.loc[(df_ldb['Username'].str.lower() == username.lower()) & (df_ldb['Chapter'] == chapter), 'Score'] = new_score
                        df_ldb.loc[(df_ldb['Username'].str.lower() == username.lower()) & (df_ldb['Chapter'] == chapter), 'Submission Time'] = datetime.now().strftime("%Y-%m-%d")
                        df_ldb.to_csv(csv_file, index = False, header = False)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns = ['Username', 'Score', 'Chapter', 'Submission Time'])
            new_row = {
                    'Username': username, 
                    'Score': self.score_per_session, 
                    'Chapter': chapter, 
                    'Submission Time': datetime.now().strftime("%Y-%m-%d")
                    }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            df.to_csv(csv_file, index = False, header = False)

    # calculate score out of list of question scores
    def calculate_score(self, question_scores):
        return round(sum(question_scores) / len(question_scores),2)
