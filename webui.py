import os
import io
import g4f
import cv2
import json
import uuid
import random
import ffmpeg
import requests
import numpy as np
import gradio as gr
from PIL import Image
from hercai import Hercai
from pprint import pprint
from elevenlabs import play
from g4f.client import Client
import google.generativeai as genai
from faster_whisper import WhisperModel
from elevenlabs.client import ElevenLabs
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.editor import AudioFileClip, concatenate_audioclips, concatenate_videoclips, ImageClip

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

# Global variables for API keys
segmind_apikey = ""
elevenlabs_apikey = ""
gemini_apikey = ""


def fetch_imagedescription_and_script(prompt, llm):
    if llm == "G4F":
        client = Client()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        output = json.loads(response.choices[0].message.content.strip())

    elif llm == "Gemini":
        genai.configure(api_key=gemini_apikey)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        output = json.loads(response.text)
    else:
        raise ValueError("Invalid LLM selected")

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
            final_prompt = "((perfect quality)), ((cinematic photo:1.3)), ((raw candid)), 4k, {}, no occlusion, Fujifilm XT3, highly detailed, bokeh, cinemascope".format(prompt.strip('.'))
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

        if not os.path.exists(active_folder):
            os.makedirs(active_folder)

        num_images = len(prompts)
        currentseed = random.randint(1, 1000000)
        print("seed ", currentseed)

        for i, prompt in enumerate(prompts):
            final_prompt = "((perfect quality)), ((cinematic photo:1.3)), ((raw candid)), 4k, {}, no occlusion, Fujifilm XT3, highly detailed, bokeh, cinemascope".format(prompt.strip('.'))
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

def generate_and_save_audio(text, active_folder, output_filename, elevenlabs_apikey, voice_name):
    client = ElevenLabs(api_key=elevenlabs_apikey)
    audio = client.generate(text=text, voice=voice_name)
    output_path = os.path.join(active_folder, f"{output_filename}.mp3")
    play(audio)
    with open(output_path, "wb") as f:
        f.write(audio)
    print(f"Audio saved as '{output_path}'")

def create_combined_video_audio(active_folder, output_filename):
    images = read_images_from_folder(active_folder)
    for i, text in enumerate(texts):
        output_filename = str(i + 1)
        print(output_filename)
        generate_and_save_audio(text, active_folder, output_filename, voice_id, elevenlabsapi)

    output_filename = "combined_video.mp4"
    create_combined_video_audio(active_folder, output_filename)
    output_video_file = os.path.join(active_folder, output_filename)

    return output_video_file

def add_captions_to_video(videofilename, wordlevelcaptions, font_color, font_size, font_name, text_position):
    video = VideoFileClip(videofilename)
    clips = [generate_text_clip(item['word'], item['start'], item['end'], video, font_color, font_size, font_name, text_position) for item in wordlevelcaptions]
    final_video = CompositeVideoClip([video] + clips)
    path, old_filename = os.path.split(videofilename)
    finalvideoname = os.path.join(path, "final.mp4")
    final_video.write_videofile(finalvideoname, codec="libx264", audio_codec="aac")
    return finalvideoname

def add_captions(inputvideo):
    print(inputvideo)
    audiofilename = extract_audio_from_video(inputvideo)
    print(audiofilename)
    wordlevelinfo = get_word_level_timestamps(model, audiofilename)
    print(wordlevelinfo)
    finalvidpath = add_captions_to_video(inputvideo, wordlevelinfo, FONT_COLOR, FONT_SIZE, FONT_NAME, POSITION)
    print(finalvidpath)
    return finalvidpath

def create_video_with_params(topic, goal, llm, image_gen, hercai_model, video_dimensions, font_color, font_size, font_name, text_position, elevenlabs_voice):
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
    image_prompts, texts = handle_errors(fetch_imagedescription_and_script, prompt, llm)

    current_uuid = uuid.uuid4()
    active_folder = str(current_uuid)

    handle_errors(generate_images, image_prompts, active_folder, image_gen, hercai_model)

    for i, text in enumerate(texts):
        output_filename = str(i + 1)
        handle_errors(generate_and_save_audio, text, active_folder, output_filename, elevenlabs_apikey, elevenlabs_voice)

    output_filename = "combined_video.mp4"
    handle_errors(create_combined_video_audio, active_folder, output_filename)

    output_video_file = os.path.join(active_folder, output_filename)
    return output_video_file

def reset_values():
    return TOPIC, GOAL, LLM, IMAGE_GEN, HERCAI_MODEL, VIDEO_DIMENSIONS, FONT_COLOR, FONT_SIZE, FONT_NAME, POSITION, ELEVENLABS_VOICE

def save_api_keys(segmind_key, elevenlabs_key, gemini_key):
    global segmind_apikey, elevenlabs_apikey, gemini_apikey
    segmind_apikey = segmind_key
    elevenlabs_apikey = elevenlabs_key
    gemini_apikey = gemini_key
    return "API keys saved successfully!"

with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("# Generate Short-form Videos for YouTube Shorts or Instagram Reels")

    with gr.Tab("Video Generation"):
        with gr.Row():
            topic = gr.Textbox(label="Topic", placeholder="Enter the video topic", value=TOPIC)
            goal = gr.Textbox(label="Goal", placeholder="Enter the video goal", value=GOAL)

        with gr.Row():
            llm = gr.Dropdown(["G4F", "Gemini"], label="Language Model", value=LLM)
            image_gen = gr.Dropdown(["Hercai", "Segmind"], label="Image Generation Model", value=IMAGE_GEN)
            hercai_model = gr.Dropdown(["v1", "v2", "v3", "lexica", "prodia"], label="Hercai Model", value=HERCAI_MODEL, visible=False)

        with gr.Row():
            video_dimensions = gr.Dropdown(["1080x1920", "1920x1080"], label="Video Dimensions", value=VIDEO_DIMENSIONS)
            font_color = gr.ColorPicker(label="Font Color", value=FONT_COLOR)
            font_size = gr.Slider(minimum=20, maximum=120, step=1, label="Font Size", value=FONT_SIZE)

        with gr.Row():
            font_name = gr.Dropdown(["Nimbus-Sans-Bold", "Arial", "Helvetica", "Times New Roman"], label="Font Name", value=FONT_NAME)
            text_position = gr.Dropdown(["center", "top", "bottom"], label="Text Position", value=POSITION)
            elevenlabs_voice = gr.Dropdown(choices=["Sarah", "Laura", "Charlie", "George", "Callum", "Liam", "Charlotte", "Alice", "Matilda", "Will", "Jessica", "Eric", "Chris", "Brian", "Daniel", "Lily", "Bill"], label="ElevenLabs Voice", value=ELEVENLABS_VOICE)

        with gr.Row():
            btn_create_video = gr.Button('Generate Video')
            btn_reset = gr.Button('Reset to Default')

        with gr.Row():
            video = gr.Video(label="Generated Video", format='mp4', height=720, width=405)
            btn_add_captions = gr.Button('Add Captions')
            final_video = gr.Video(label="Video with Captions", format='mp4', height=720, width=405)

    with gr.Tab("API Keys"):
        segmind_key_input = gr.Textbox(label="Segmind API Key", type="password")
        elevenlabs_key_input = gr.Textbox(label="ElevenLabs API Key", type="password")
        gemini_key_input = gr.Textbox(label="Gemini API Key", type="password")
        save_keys_btn = gr.Button("Save API Keys")
        api_status = gr.Textbox(label="Status", interactive=False)

    def update_hercai_visibility(choice):
        return gr.update(visible=choice == "Hercai")

    image_gen.change(fn=update_hercai_visibility, inputs=[image_gen], outputs=[hercai_model])

    btn_create_video.click(
        fn=create_video_with_params,
        inputs=[topic, goal, llm, image_gen, hercai_model, video_dimensions, font_color, font_size, font_name, text_position, elevenlabs_voice],
        outputs=[video]
    )

    btn_reset.click(
        fn=reset_values,
        inputs=[],
        outputs=[topic, goal, llm, image_gen, hercai_model, video_dimensions, font_color, font_size, font_name, text_position, elevenlabs_voice]
    )

    btn_add_captions.click(
        fn=add_captions,
        inputs=[video],
        outputs=[final_video]
    )

demo.queue().launch(debug=True)
