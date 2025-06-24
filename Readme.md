# QuizGenerator: AI-Assisted Formative Assessment Tool 

## Overview 

QuizGenerator is a tool designed to create quizzes from audio content, aiming to support formative assessment in educational settings. By using language models and audio processing techniques, QuizGenerator converts lectures, podcasts, or other educational audio content into interactive quizzes.

## Key Features 

- Audio Processing: Transcribes and summarizes audio files, extracting key information.
- Quiz Generation: Uses language models (Llama 3.1) to create questions based on the audio content.
- Adaptive Questioning: Adjusts questions based on user responses for a more personalized experience.
- Feedback: Provides feedback on user answers to support learning.
- Formative Assessment: Designed to assist with ongoing assessment, helping identify areas for improvement.
- Customizable Prompts: Allows customization of system prompts and evaluation criteria.

## How It Works 

1. Audio Input: Upload an audio file (lecture, podcast, etc.) to the system.
2. Transcription & Summarization: The audio is transcribed and summarized.
3. Quiz Generation: Based on the summary, questions are generated.
4. Interactive Q&A: Students answer the generated questions and receive feedback.
5. Performance Analysis: The system evaluates responses to provide insights into student understanding.

## Potential Use Cases 

- Classroom Support: Generate review quizzes from lecture content.
- Self-Study: Offer students a way to test understanding of audio-based materials.
- Remote Learning: Create assessments from recorded lectures for distance education.
- Ongoing Assessment: Provide a tool for continuous evaluation throughout a course.

## Configuration

The text-to-speech transcription and the summary of the transcipt is only configured via command-line parameters or directly in the code.

The configuration of the quiz-generating model is done in `utils/config_chat.yaml`. 
    
The configuration of the quiz is done in the `config.yaml` file.

## Usage

### Installing dependencies

To install the dependencies, run the following command:

    ```bash
    pip install -r requirements.txt
    ```
Pull LLM models from Ollama:

    ```bash
    ollama pull gemma2:27b
    ```

### Creating summaries of audio files

To create a summary of an audio file, run the following command:

    ```bash
    python3 summarize_audio.py --input_file <path_to_audio_file> --output_file <path_to_output_file> --model_t <model_size for transcription> --model_s <model for the summary> --save_t <path_to_transcribed_audio>
    ```
    
Where:
- <path_to_audio_file> is the path to the audio file to be summarized
- <path_to_output_file> is the path to the output file where the summary will be saved
- <model_t> is the size of the model to be used for summarization. The available options are 'small', 'medium', and 'large'. Default is 'large'
- <save_t> is the path to the output file where the transcription will be saved. Optional, if empty, the transcription will not be saved
- <model_s> is the size of the model to be used for transcription. Use a model compatible with ChatOllama, e.g. llama3.1:70b

### Creating quizzes from summaries

Copy the summary file to the `lectures` directory. It will be automatically shown in the dropdown menu in the GUI.

### Running the GUI

To run the GUI, run the following command:

    ```bash
    streamlit run app.py
    ```
## Contributing
We welcome contributions to the QuizGenerator project. If you encounter issues, have suggestions, or would like to contribute code, please feel free to submit a pull request or reach out.

## Dependencies

This project uses the following third-party libraries:
- [Whisper](https://github.com/openai/whisper) - Licensed under the MIT License
- [Langchain](https://github.com/langchain-ai/langchain) - Licensed under the Apache License 2.0
