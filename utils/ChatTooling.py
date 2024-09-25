from langchain_community.chat_models import ChatOllama
from langchain.prompts import (ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate)
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langsmith import traceable
import yaml
import json
from pathlib import Path

#export LANGCHAIN_TRACING_V2="true"
#export LANGCHAIN_API_KEY="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"


class ChatTooling:
    def __init__(self) -> None:
        self.config = self.load_config(Path("utils/config_chat.yaml"))
        self.prompts = self.load_prompts("prompts")
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

    def load_config(self, config_path):
        """ Helper function to load the configuration file. """        
        with open(config_path) as file: 
            return yaml.safe_load(file)

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
            elif isinstance(message, SystemMessage):
                serialized_messages.append({"type": "system", "content": message.content})
        return serialized_messages

    @traceable
    def query(self, user_input):
        # Manually add the human input to the memory
        self.conversation_buffer.chat_memory.add_message(HumanMessage(content=user_input))
        
        # Serialize the conversation history
        serialized_history = self._serialize_messages(self.conversation_buffer.chat_memory.messages)
        
        # Increment the question count
        self.question_count += 1

        # Run the chain to get the response
        response = self.chain.invoke({
            "history": serialized_history,
            "input": user_input
        })

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

                # Clear the list after processing a full AI-human pair
                message_answer_pairs.clear()

        # Format the array into a string
        rating_per_qa = "\n".join([f"Question {i+1}: {rating} \n\n" for i, rating in enumerate(rating_per_qa)])

        return rating_per_qa

    def reset_conversation(self):
        self.conversation_buffer.clear()
        self.question_count = 0
