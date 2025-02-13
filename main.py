import os
import streamlit as st
from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_elevenlabs
from pydub import AudioSegment

# Load API keys (Make sure you have these in your environment)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

system_prompt = """You have to act as a professional doctor. I know you are not, but this is for learning purposes. 
            What's in this image? Do you find anything wrong with it medically? 
            If you make a differential, suggest some remedies for them. 
            Keep your answer concise (max 2 sentences)."""

st.title("🩺 AI Doctor with Vision and Voice")

# Upload Audio
audio_file = st.file_uploader("Upload your audio file", type=["wav", "mp3", "ogg"])

# Upload Image
image_file = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png"])

if st.button("Analyze"):
    if audio_file:
        # Save uploaded file temporarily
        audio_path = "temp_audio.mp3"
        with open(audio_path, "wb") as f:
            f.write(audio_file.read())

        # Transcribe Audio
        speech_to_text_output = transcribe_with_groq(
            GROQ_API_KEY=GROQ_API_KEY, 
            audio_filepath=audio_path,
            stt_model="whisper-large-v3"
        )
        st.text("🎙️ Transcription: " + speech_to_text_output)

        # Process Image (if provided)
        if image_file:
            encoded_image = encode_image(image_file)
            doctor_response = analyze_image_with_query(
                query=system_prompt + speech_to_text_output,
                encoded_image=encoded_image,
                model="llama-3.2-11b-vision-preview"
            )
        else:
            doctor_response = "No image provided for analysis."
        
        st.text("🩺 Doctor's Response: " + doctor_response)

        # Text-to-Speech
        audio_output_path = "final.mp3"
        text_to_speech_with_elevenlabs(
            input_text=doctor_response, 
            output_filepath=audio_output_path
        )

        # Convert MP3 to WAV (For Streamlit Playback)
        audio = AudioSegment.from_mp3(audio_output_path)
        wav_output_path = "final.wav"
        audio.export(wav_output_path, format="wav")

        # Play the generated voice response
        st.audio(wav_output_path)

    else:
        st.warning("⚠️ Please upload an audio file first.")

