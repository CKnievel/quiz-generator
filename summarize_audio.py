# Description: This script transcribes an audio file and summarizes the text.

import argparse
import whisper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain.prompts import (PromptTemplate)

from dotenv import load_dotenv
import os 

# Create the argument parser
parser = argparse.ArgumentParser(description="Convert speech to text and summarize the text.")

# Add the arguments
parser.add_argument("-i", "--input", type=str, help="The path to the audio file to transcribe.")
parser.add_argument("-o", "--output", type=str, help="The path to save the summarized lecture content.")
# Add the optional arguments
parser.add_argument("-is", "--input_s", type=str, help="The path to the summary file. If summary file is given, audio input file will be ignored.")
parser.add_argument("-mt", "--model_t", type=str, default="large", help="The model to use for transcription.")
parser.add_argument("-st", "--save_t", type=str, help="Save the transcribed audio file to this path. (Output of whisper)")
parser.add_argument("-sst", "--save_st", type=str, default="summary.txt", help="Save the summarized transcription to this path.")
parser.add_argument("-ms", "--model_s", type=str, default="gemma2:27b", help="The model to use for summarization.")

# Parse the arguments
args = parser.parse_args()

# Check for given summary file
if not args.input_s:
    # Load the model for transcription
    print("Loading model for transcription...", end="", flush=True)
    model = whisper.load_model(args.model_t)
    print("done")

    # Transcribe the audio file
    print("Transcribing audio file...", end="", flush=True)
    result = model.transcribe(args.input)
    print("done")

    # Saving the transcribed text to a file
    if args.save_t:
        with open(args.save_t, "w", encoding="utf-8") as file:
            file.write(result["text"])

    # Split the text into chunks. Split at sentences or line breaks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
        separators=[
            "\n\n",
            "\n",
            ".",
        ],)
    sentences = splitter.split_text(result["text"])

    # Summarize the text per chunk
    print("Loading model for summarization...", end="", flush=True)
    if args.model_s == "anthropic":
        load_dotenv()
        MODEL = "claude-3-5-sonnet-20240620"
        ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
        model = ChatAnthropic(
            anthropic_api_key=ANTHROPIC_API_KEY, 
            model_name=MODEL, 
            temperature=0.5, 
            max_tokens=3000
        )

    else:
        model = ChatOllama(
            model=args.model_s,
            temperature=0.5,
            top_k=10, 
            top_p=0.7,
            keep_alive = -1,
        )
    
    print("done")


    query = """Please summarize the text. Stay as close to the original text as possible. Output only the summary.
        # Text
        {content}
        # Summary
        """
    prompt_template = PromptTemplate.from_template(template=query)

    summaries = []
    print(f"Summarizing {len(sentences)} sentences...")
    for sentence in sentences:
        print(f"Summary for sentence #{len(summaries)+1}...", end="", flush=True)
        chain = prompt_template | model
        summary = chain.invoke({"content": sentence})
        
        summaries.append(summary.content)
        print("done")

    # Save the summaries to output file   
    with open(args.save_st, "w", encoding="utf-8") as file:
        for summary in summaries:
            file.write(summary + "\n")
else:
    # Summarize the text per chunk
    print("Loading model for summarization...", end="", flush=True)
    if args.model_s == "anthropic":
        load_dotenv()
        MODEL = "claude-3-5-sonnet-20240620"
        ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
        model = ChatAnthropic(
            anthropic_api_key=ANTHROPIC_API_KEY, 
            model_name=MODEL, 
            temperature=0.5, 
            max_tokens=3000
        )

    else:
        model = ChatOllama(
            model=args.model_s,
            temperature=0.5,
            top_k=10, 
            top_p=0.7,
            keep_alive = -1,
        )
    
    print("done")

    # Transcript file is given, read the file
    with open(args.input_s, "r", encoding="utf-8") as file:
        summary = file.read()

# Now provide a second summary of the entire text
with open("prompts/summary_prompt.txt", "r") as file:
    query = file.read()

chain = PromptTemplate.from_template(template=query) | model
summary = chain.invoke({"content": summary})

# save summary to output file
with open(args.output , "w", encoding="utf-8") as file:
    file.write(summary.content)
