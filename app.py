import gradio as gr
from pydub import AudioSegment
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import whisper

# Load the audio model
audio_model = whisper.load_model("large", device='cpu')

# Load the sentence model
sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# DataFrame containing chunk paths and transcriptions
df_mapping = pd.read_csv('audio_chunk_mapping_with_transcription_embeddings.csv')

# Function to process the input audio and retrieve the most similar audio chunk
def process_and_find_audio(audio_file):
    # Load audio from the file path
    audio_path = "./temp_audio.wav"  # Path to temporarily save audio if needed
    sample_rate,audio_np  = audio_file
    # Save the numpy array as an audio file if you need to pass it to the Whisper model
    audio_segment = AudioSegment(
        audio_np.tobytes(), 
        frame_rate=sample_rate,  # Set the frame rate as appropriate
        sample_width=2,  # Assuming 16-bit samples (adjust if necessary)
        channels=1  # Assuming mono channel (adjust if necessary)
    )
    # Save the audio to a temporary file
    audio_segment.export(audio_path, format="wav")        

    # audio_path = audio_file.name
    transcription = audio_model.transcribe(audio_path, task="translate")['text']
    
    # Compute embeddings for database transcriptions and user transcription
    embeddings = df_mapping.iloc[:, 4:].to_numpy().astype('float32')
    embedding_query = sentence_model.encode(transcription)

    # Find the most similar transcription
    similarities = sentence_model.similarity(embeddings, embedding_query)
    index_of_most_similar_item = int(similarities.argmax())

    # Retrieve the matching audio chunk path and transcription
    matched_chunk_path = df_mapping.loc[index_of_most_similar_item, "chunk_path"]
    matched_chunk_text = df_mapping.loc[index_of_most_similar_item, "transcription"]
    print(matched_chunk_path, matched_chunk_text)
    # Return the text and audio data
    return matched_chunk_text, matched_chunk_path

# Set up the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("### Upload/Record an audio file and retrieve the most similar excerpt from the database of audio.")

    # Use gr.File for file upload and define text + audio outputs
    mic = gr.Audio(type="numpy", label="Record Your Audio")
    output_text = gr.Textbox(label="Matched Transcription")
    output_audio = gr.Audio(label="Matched Audio Playback")

    # Link the function to Gradio inputs and outputs
    mic.change(process_and_find_audio, inputs=mic, outputs=[output_text, output_audio])

# Launch the app
demo.launch(share=True)
# demo.launch()












# import gradio as gr
# from pydub import AudioSegment
# import numpy as np
# from io import BytesIO

# # Simulated function to fetch audio from a "database"
# def get_audio_from_database():
#     # Replace with actual database retrieval logic
#     return "/home/shashank/Desktop/ml_fiest/Dataset/SandalWoodNewsStories_2.mp3"  # Example path to an audio file

# # Define the function that takes the user-recorded audio and returns database audio
# def process_audio(user_audio):
#     # Process the user audio if needed
#     # Here we’re just passing it through without saving for demonstration purposes

#     # Get the database audio
#     db_audio_path = get_audio_from_database()
#     db_audio_segment = AudioSegment.from_file(db_audio_path)

#     # Convert db audio to numpy array and sample rate
#     buffer = BytesIO()
#     db_audio_segment.export(buffer, format="wav")
#     buffer.seek(0)
#     db_audio = np.frombuffer(buffer.read(), dtype=np.int16)

#     # Return the database audio as a response
#     return (db_audio_segment.frame_rate, db_audio)

# # Set up the Gradio interface
# with gr.Blocks() as demo:
#     gr.Markdown("### Record your audio and play a sample from the database.")
    
#     # Define microphone input and audio output without `source` argument
#     mic = gr.Audio(type="numpy", label="Record Your Audio")
#     output_audio = gr.Audio(label="Database Audio Response")
    
#     # Connect the function with Gradio components
#     mic.change(process_audio, inputs=mic, outputs=output_audio)

# # Launch the app
# demo.launch(share=True)
