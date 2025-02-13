# Import necessary libraries
import os
import gradio as gr
from pydub import AudioSegment

# Importing AI processing functions
from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_elevenlabs

# System prompt for the AI doctor
system_prompt = """You have to act as a professional doctor, I know you are not but this is for learning purposes. 
With what I see, I think you have .... Do you find anything wrong with it medically? 
If you make a differential, suggest some remedies for them. Do not add any numbers or special characters in 
your response. Your response should be in one long paragraph. Always answer as if you are answering a real person.
Do not respond as an AI model in markdown. Keep your answer concise (max 2 sentences). No preamble, start your answer right away please."""


# Function to process inputs
def process_inputs(audio_filepath, image_filepath):
    """Handles audio transcription, image analysis, and text-to-speech generation."""

    print(f"DEBUG: Received audio file path: {audio_filepath}")

    # Ensure audio file exists before processing
    if not audio_filepath or not os.path.exists(audio_filepath):
        return "Error: No valid audio file provided.", "No response generated.", None

    try:
        # Convert speech to text using Groq API
        speech_to_text_output = transcribe_with_groq(
            GROQ_API_KEY=os.getenv("GROQ_API_KEY"),
            audio_filepath=audio_filepath,
            stt_model="whisper-large-v3",
        )
    except Exception as e:
        return f"Error transcribing audio: {e}", "No response generated.", None

    # Handle image analysis
    if image_filepath and os.path.exists(image_filepath):
        try:
            encoded_img = encode_image(image_filepath)
            doctor_response = analyze_image_with_query(
                query=system_prompt + speech_to_text_output,
                encoded_image=encoded_img,
                model="llama-3.2-11b-vision-preview",
            )
        except Exception as e:
            doctor_response = f"Error analyzing image: {e}"
    else:
        doctor_response = "No image provided for analysis."

    # Convert doctor's response to speech using ElevenLabs
    output_wav = "final.wav"
    try:
        text_to_speech_with_elevenlabs(
            input_text=doctor_response,
            output_filepath="final.mp3",  # Generate MP3 first
        )

        # Convert MP3 to WAV
        if os.path.exists("final.mp3"):
            audio = AudioSegment.from_mp3("final.mp3")
            audio.export(output_wav, format="wav")
        else:
            return (
                speech_to_text_output,
                doctor_response,
                "Error: Failed to generate audio.",
            )
    except Exception as e:
        return speech_to_text_output, doctor_response, f"Error generating speech: {e}"

    return speech_to_text_output, doctor_response, output_wav


# Create Gradio Interface
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath"),
        gr.Image(type="filepath"),
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Doctor's Response"),
        gr.Audio(label="Doctor's Voice Response"),
    ],
    title="AI Doctor with Vision and Voice",
    description="Upload an image and speak into the microphone. The AI doctor will analyze the image, transcribe your speech, and respond in both text and voice.",
)

# Launch
iface.launch()
