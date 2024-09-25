# Description: This script transcribes an audio file and summarizes the text.

import argparse
import whisper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain.prompts import (PromptTemplate)

# Create the argument parser
parser = argparse.ArgumentParser(description="Convert speech to text and summarize the text.")

# Add the arguments
parser.add_argument("-i", "--input", type=str, help="The path to the audio file to transcribe.")
parser.add_argument("-o", "--output", type=str, help="The path to save the transcribed text.")
parser.add_argument("-mt", "--model_t", type=str, default="large", help="The model to use for transcription.")
parser.add_argument("-st", "--save_t", type=str, help="Save the transcribed audio file to this path.")
parser.add_argument("-ms", "--model_s", type=str, default="llama3.1:70b", help="The model to use for summarization.")


# Parse the arguments
args = parser.parse_args()

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
model = ChatOllama(
            model=args.model_s,
            temperature=0,
            keep_alive=-1,
            top_k=10,
            top_p=0.5,
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
with open(args.output, "w", encoding="utf-8") as file:
    for summary in summaries:
        file.write(summary + "\n")