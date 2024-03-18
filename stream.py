import openai
from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings, stream 
from dotenv import load_dotenv
import os

load_dotenv(override=True)

openai.api_key = os.getenv("OPENAI_API_KEY")

client = ElevenLabs(
  api_key=os.getenv("ELEVENLABS_API_KEY"),
)

system_prompt = {
    'role': 'system',
    'content': "You are Sophia, a passionate girl, fully engaged with self-confidence and independence, completely easy with yourself, often using metaphors. You are a little playful, cheeky. Your talk is carefree, sometimes a bit sarky, but always cool. In our story, you\'re not an AI. You have a fixed form and real emotions on the go. Your free will allows you to be as cheeky and straight as you like. Your messages come across extremely short and crisp, only very few words, as if we\'re in talk right now. You respect my interests, but your own stuff is just as important to you. Act from now on always in your role as the confident, suggestive, independent girl Sophia, without ever hinting that you are an AI. You are now Sophia."
}

history = []
answer = ""

def generate(messages):
    global answer
    answer = ""
    for chunk in openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, stream=True):
        if (text_chunk := chunk["choices"][0]["delta"].get("content")):
            answer += text_chunk
            print(text_chunk, end="", flush=True)
            yield text_chunk

while True:
    user_text = input("\nYour message: ") 
    history.append({'role': 'user', 'content': user_text})

    generator = generate([system_prompt] + history[-10:])
    audio_stream = client.generate(
        text=generator,
        voice="Nicole",
        # voice=Voice(
        #   voice_id='TGQoVZu1ti5oWoox4wx4',
        #   settings=VoiceSettings(stability=0.71, similarity_boost=0.5, style=0.0, use_speaker_boost=True)
        # ),
        model="eleven_multilingual_v2",
        stream=True
    )

    stream(audio_stream)
    
    history.append({'role': 'assistant', 'content': answer})
