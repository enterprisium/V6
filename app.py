import re
import os
import io
import json
import uuid
import torch
import random
import requests
import numpy as np
import gradio as gr
from PIL import Image
from TTS.api import TTS
from pprint import pprint
from hercai import Hercai
from elevenlabs import play
from g4f.client import Client
import google.generativeai as genai
from faster_whisper import WhisperModel
from elevenlabs.client import ElevenLabs
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip, concatenate_videoclips, ImageClip



# Define default values
TOPIC = "Success and Achievement"
GOAL = "Inspire people to overcome challenges, achieve success, and celebrate their victories"
LLM = "G4F"
IMAGE_GEN = "Hercai"
HERCAI_MODEL = "v3"
VIDEO_DIMENSIONS = "1080x1920"
FONT_COLOR = "#FFFFFF"
FONT_SIZE = 80
FONT_NAME = "Nimbus-Sans-Bold"
POSITION = "center"
ELEVENLABS_VOICE = "Adam"
TTS_ENGINE = "ElevenLabs"
XTTS_VOICE = "Ana Florence"


# Global variables for API keys
segmind_apikey = ""
elevenlabs_apikey = ""
gemini_apikey = ""

# Available ElevenLabs voices
AVAILABLE_VOICES = ["Adam", "Antoni", "Arnold", "Bella", "Domi", "Elli", "Josh", "Rachel", "Sam"]

# Available XTTS voices
XTTS_VOICES = ["Claribel Dervla", "Daisy Studious", "Gracie Wise", "Tammie Ema", "Alison Dietlinde", "Ana Florence", "Annmarie Nele", "Asya Anara", "Brenda Stern", "Gitta Nikolina", "Henriette Usha", "Sofia Hellen", "Tammy Grit", "Tanja Adelina", "Vjollca Johnnie", "Andrew Chipper", "Badr Odhiambo", "Dionisio Schuyler", "Royston Min", "Viktor Eka", "Abrahan Mack", "Adde Michal", "Baldur Sanjin", "Craig Gutsy", "Damien Black", "Gilberto Mathias", "Ilkin Urbano", "Kazuhiko Atallah", "Ludvig Milivoj", "Suad Qasim", "Torcull Diarmuid", "Viktor Menelaos", "Zacharie Aimilios", "Nova Hogarth", "Maja Ruoho", "Uta Obando", "Lidiya Szekeres", "Chandra MacFarland", "Szofi Granger", "Camilla Holmström", "Lilya Stainthorpe", "Zofija Kendrick", "Narelle Moon", "Barbora MacLean", "Alexandra Hisakawa", "Alma María", "Rosemary Okafor", "Ige Behringer", "Filip Traverse", "Damjan Chapman", "Wulf Carlevaro", "Aaron Dreschner", "Kumar Dahl", "Eugenio Mataracı", "Ferran Simen", "Xavier Hayasaka", "Luis Moray", "Marcos Rudaski"]

# Available languages
AVAILABLE_LANGUAGES = ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Russian", "Chinese", "Japanese", "Korean"]

def fetch_imagedescription_and_script(prompt, llm):
    if llm == "G4F":
        client = Client()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = response.choices[0].message.content.strip()
    elif llm == "Gemini":
        genai.configure(api_key=gemini_apikey)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        response_text = response.text
    else:
        raise ValueError("Invalid LLM selected")

    # Extract JSON from the response
    json_match = re.search(r'\[[\s\S]*\]', response_text)
    if json_match:
        json_str = json_match.group(0)
        output = json.loads(json_str)
    else:
        raise ValueError("No valid JSON found in the response")

    pprint(output)
    image_prompts = [k['image_description'] for k in output]
    texts = [k['text'] for k in output]
    return image_prompts, texts

def generate_images(prompts, active_folder, image_gen, hercai_model=None):
    if not os.path.exists(active_folder):
        os.makedirs(active_folder)

    if image_gen == "Hercai":
        herc = Hercai("")
        for i, prompt in enumerate(prompts):
            final_prompt = f"((perfect quality)), ((cinematic photo:1.3)), ((raw candid)), 4k, {prompt.strip('.')}, no occlusion, Fujifilm XT3, highly detailed, bokeh, cinemascope"
            try:
                image_result = herc.draw_image(
                    model=hercai_model,
                    prompt=final_prompt,
                    negative_prompt="((deformed)), ((limbs cut off)), ((quotes)), ((extra fingers)), ((deformed hands)), extra limbs, disfigured, blurry, bad anatomy, absent limbs, blurred, watermark, disproportionate, grainy, signature, cut off, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, extra eyes, amputation, disconnected limbs"
                )
                image_url = image_result["url"]
                image_response = requests.get(image_url)
                if image_response.status_code == 200:
                    image_data = image_response.content
                    image = Image.open(io.BytesIO(image_data))
                    image_filename = os.path.join(active_folder, f"{i + 1}.jpg")
                    image.save(image_filename)
                    print(f"Image {i + 1}/{len(prompts)} saved as '{image_filename}'")
                else:
                    print(f"Error: Failed to download image {i + 1}")
            except Exception as e:
                print(f"Error generating image {i + 1}: {str(e)}")
    elif image_gen == "Segmind":
        url = "https://api.segmind.com/v1/sdxl1.0-txt2img"
        headers = {'x-api-key': segmind_apikey}

        num_images = len(prompts)
        currentseed = random.randint(1, 1000000)
        print("seed ", currentseed)

        for i, prompt in enumerate(prompts):
            final_prompt = f"((perfect quality)), ((cinematic photo:1.3)), ((raw candid)), 4k, {prompt.strip('.')}, no occlusion, Fujifilm XT3, highly detailed, bokeh, cinemascope"
            data = {
                "prompt": final_prompt,
                "negative_prompt": "((deformed)), ((limbs cut off)), ((quotes)), ((extra fingers)), ((deformed hands)), extra limbs, disfigured, blurry, bad anatomy, absent limbs, blurred, watermark, disproportionate, grainy, signature, cut off, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, extra eyes, amputation, disconnected limbs",
                "style": "hdr",
                "samples": 1,
                "scheduler": "UniPC",
                "num_inference_steps": 30,
                "guidance_scale": 8,
                "strength": 1,
                "seed": currentseed,
                "img_width": 1024,
                "img_height": 1024,
                "refiner": "yes",
                "base64": False
            }

            response = requests.post(url, json=data, headers=headers)

            if response.status_code == 200 and response.headers.get('content-type') == 'image/jpeg':
                image_data = response.content
                image = Image.open(io.BytesIO(image_data))
                image_filename = os.path.join(active_folder, f"{i + 1}.jpg")
                image.save(image_filename)
                print(f"Image {i + 1}/{num_images} saved as '{image_filename}'")
            else:
                print(response.text)
                print(f"Error: Failed to retrieve or save image {i + 1}")

def generate_and_save_audio(text, active_folder, output_filename, tts_engine, elevenlabs_apikey=None, voice_name=None):
    if tts_engine == "ElevenLabs":
        client = ElevenLabs(api_key=elevenlabs_apikey)
        audio_generator = client.generate(text=text, voice=voice_name)
        audio_content = b"".join(audio_generator)  # Convert generator to bytes
        output_path = os.path.join(active_folder, f"{output_filename}.mp3")
        play(audio_content)
        with open(output_path, "wb") as f:
            f.write(audio_content)
        print(f"Audio saved as '{output_path}'")
    elif tts_engine == "XTTS_V2":
        os.environ["COQUI_TOS_AGREED"] = "1"  # Assuming agreement to TOS
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts.to(device)
        output_path = os.path.join(active_folder, f"{output_filename}.mp3")
        tts.tts_to_file(text=text, file_path=output_path, language="en", speaker=voice_name)
        print(f"Audio saved as '{output_path}'")

def create_combined_video_audio(active_folder, output_filename):
    image_files = sorted([f for f in os.listdir(active_folder) if f.endswith('.jpg')])
    audio_files = sorted([f for f in os.listdir(active_folder) if f.endswith('.mp3')])

    clips = []
    for img, aud in zip(image_files, audio_files):
        img_path = os.path.join(active_folder, img)
        aud_path = os.path.join(active_folder, aud)

        image_clip = ImageClip(img_path).set_duration(AudioFileClip(aud_path).duration)
        audio_clip = AudioFileClip(aud_path)
        video_clip = image_clip.set_audio(audio_clip)
        clips.append(video_clip)

    final_clip = concatenate_videoclips(clips)
    output_path = os.path.join(active_folder, output_filename)
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    return output_path

def extract_audio_from_video(outvideo):
    if outvideo is None:
        raise ValueError("Input video is None")
    audiofilename = outvideo.replace(".mp4", '.mp3')
    input_stream = ffmpeg.input(outvideo)
    audio = input_stream.audio
    output_stream = ffmpeg.output(audio, audiofilename)
    output_stream = ffmpeg.overwrite_output(output_stream)
    ffmpeg.run(output_stream)
    return audiofilename

def generate_text_clip(word, start, end, video, font_color, font_size, font_name, text_position):
    txt_clip = (TextClip(word, fontsize=font_size, color=font_color, font=font_name, stroke_width=3, stroke_color='black')
                .set_position(text_position)
                .set_duration(end - start))
    return txt_clip.set_start(start)

def get_word_level_timestamps(model, audioname):
    segments, info = model.transcribe(audioname, word_timestamps=True)
    segments = list(segments)
    wordlevel_info = []
    for segment in segments:
        for word in segment.words:
            wordlevel_info.append({'word': word.word, 'start': word.start, 'end': word.end})
    return wordlevel_info

model_size = "base"
model = WhisperModel(model_size)

def add_captions_to_video(videofilename, wordlevelcaptions, font_color, font_size, font_name, text_position):
    video = VideoFileClip(videofilename)
    clips = [generate_text_clip(item['word'], item['start'], item['end'], video, font_color, font_size, font_name, text_position) for item in wordlevelcaptions]
    final_video = CompositeVideoClip([video] + clips)
    path, old_filename = os.path.split(videofilename)
    finalvideoname = os.path.join(path, "final.mp4")
    final_video.write_videofile(finalvideoname, codec="libx264", audio_codec="aac")
    return finalvideoname

def create_video_with_params(topic, goal, llm, image_gen, hercai_model, video_dimensions, font_color, font_size, font_name, text_position, tts_engine, elevenlabs_voice, xtts_voice):
    prompt_prefix = f"""You are tasked with creating a script for a {topic} video that is about 30 seconds.
Your goal is to {goal}.
Please follow these instructions to create an engaging and impactful video:
1. Begin by setting the scene and capturing the viewer's attention with a captivating visual.
2. Each scene cut should occur every 5-10 seconds, ensuring a smooth flow and transition throughout the video.
3. For each scene cut, provide a detailed description of the stock image being shown.
4. Along with each image description, include a corresponding text that complements and enhances the visual. The text should be concise and powerful.
5. Ensure that the sequence of images and text builds excitement and encourages viewers to take action.
6. Strictly output your response in a JSON list format, adhering to the following sample structure:"""

    sample_output = """
    [
        { "image_description": "Description of the first image here.", "text": "Text accompanying the first scene cut." },
        { "image_description": "Description of the second image here.", "text": "Text accompanying the second scene cut." },
        ...
    ]"""

    prompt_postinstruction = f"""By following these instructions, you will create an impactful {topic} short-form video.
    Output:"""

    prompt = prompt_prefix + sample_output + prompt_postinstruction
    image_prompts, texts = fetch_imagedescription_and_script(prompt, llm)

    current_uuid = uuid.uuid4()
    active_folder = str(current_uuid)

    generate_images(image_prompts, active_folder, image_gen, hercai_model)

    for i, text in enumerate(texts):
        output_filename = str(i + 1)
        if tts_engine == "ElevenLabs":
            generate_and_save_audio(text, active_folder, output_filename, tts_engine, elevenlabs_apikey, elevenlabs_voice)
        elif tts_engine == "XTTS_V2":
            generate_and_save_audio(text, active_folder, output_filename, tts_engine, voice_name=xtts_voice)

    output_filename = "combined_video.mp4"
    output_video_file = create_combined_video_audio(active_folder, output_filename)

    # Extract audio from the combined video
    audiofilename = extract_audio_from_video(output_video_file)

    # Generate word-level timestamps using Faster Whisper
    wordlevelinfo = get_word_level_timestamps(model, audiofilename)

    # Add captions to the video
    final_video_path = add_captions_to_video(output_video_file, wordlevelinfo, font_color, font_size, font_name, text_position)

    return final_video_path

def reset_values():
    return TOPIC, GOAL, LLM, IMAGE_GEN, HERCAI_MODEL, VIDEO_DIMENSIONS, FONT_COLOR, FONT_SIZE, FONT_NAME, POSITION, ELEVENLABS_VOICE, TTS_ENGINE, XTTS_VOICE

def save_api_keys(segmind_key, elevenlabs_key, gemini_key):
    global segmind_apikey, elevenlabs_apikey, gemini_apikey
    segmind_apikey = segmind_key
    elevenlabs_apikey = elevenlabs_key
    gemini_apikey = gemini_key
    return "API keys saved successfully!"
# Gradio UI
with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("# SocialGPT - Generate Short-form Videos")

    with gr.Box(visible=False):
        with gr.Row():
            segmind_key_input = gr.Textbox(label="Segmind API Key", type="password")
            elevenlabs_key_input = gr.Textbox(label="ElevenLabs API Key", type="password")
            gemini_key_input = gr.Textbox(label="Gemini API Key", type="password")
        with gr.Row():
            save_keys_btn = gr.Button("Save API Keys")
            status_output = gr.Textbox(label="Status", interactive=False)

    with gr.Accordion("Script Settings", open=True):
        with gr.Row():
            topic = gr.Textbox(label="Video Topic", value=TOPIC)
            goal = gr.Textbox(label="Video Goal", value=GOAL)
            llm = gr.Dropdown(["G4F", "Gemini"], label="Language Model", value=LLM)

    with gr.Accordion("Video Settings", open=False):
        with gr.Row():
            image_gen = gr.Dropdown(["Hercai", "Segmind"], label="Image Generation Model", value=IMAGE_GEN)
            hercai_model = gr.Dropdown(["v1", "v2", "v3", "lexica", "prodia"], label="Hercai Model", value=HERCAI_MODEL, visible=False)
            video_dimensions = gr.Dropdown(["1080x1920", "1920x1080"], label="Video Dimensions", value=VIDEO_DIMENSIONS)

    with gr.Accordion("Audio Settings", open=False):
        with gr.Row():
            tts_engine = gr.Dropdown(["ElevenLabs", "XTTS_V2"], label="TTS Engine", value=TTS_ENGINE)
            elevenlabs_voice = gr.Dropdown(AVAILABLE_VOICES, label="ElevenLabs Voice", value=ELEVENLABS_VOICE)
            xtts_voice = gr.Dropdown(XTTS_VOICES, label="XTTS Voice", value=XTTS_VOICE, visible=False)

    with gr.Accordion("Subtitles Settings", open=False):
        with gr.Row():
            font_color = gr.ColorPicker(label="Font Color", value=FONT_COLOR)
            font_size = gr.Number(label="Font Size", value=FONT_SIZE)
        with gr.Row():
            font_name = gr.Dropdown(["Nimbus-Sans-Bold", "Arial", "Helvetica", "Times New Roman"], label="Font Name", value=FONT_NAME)
            text_position = gr.Dropdown(["center", "top", "bottom"], label="Text Position", value=POSITION)

    with gr.Row():
        video_output = gr.Video(label="Generated Video", format='mp4')

    with gr.Row():
        generate_btn = gr.Button("Generate", variant="primary")
        reset_btn = gr.Button("Reset")

    def update_hercai_visibility(choice):
        return gr.update(visible=choice == "Hercai")

    def update_voice_visibility(tts_engine):
        return gr.update(visible=tts_engine == "XTTS_V2"), gr.update(visible=tts_engine == "ElevenLabs")

    image_gen.change(fn=update_hercai_visibility, inputs=[image_gen], outputs=[hercai_model])
    tts_engine.change(fn=update_voice_visibility, inputs=[tts_engine], outputs=[xtts_voice, elevenlabs_voice])

    generate_btn.click(
        fn=create_video_with_params,
        inputs=[topic, goal, llm, image_gen, hercai_model, video_dimensions, font_color, font_size, font_name, text_position, tts_engine, elevenlabs_voice, xtts_voice],
        outputs=[video_output]
    )

    reset_btn.click(
        fn=reset_values,
        inputs=[],
        outputs=[topic, goal, llm, image_gen, hercai_model, video_dimensions, font_color, font_size, font_name, text_position, elevenlabs_voice, tts_engine, xtts_voice]
    )

    save_keys_btn.click(
        fn=save_api_keys,
        inputs=[segmind_key_input, elevenlabs_key_input, gemini_key_input],
        outputs=[status_output]
    )

demo.launch(debug=True, enable_queue=True)
