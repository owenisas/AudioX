import gc
import platform
import os
import subprocess as sp  # For merging audio and video

import numpy as np
import gradio as gr
import json 
import torch
import torchaudio
import torchvision
import decord
from decord import VideoReader
from decord import cpu
import math
import einops
import torchvision.transforms as transforms

from aeiou.viz import audio_spectrogram_image
from einops import rearrange
from safetensors.torch import load_file
from torch.nn import functional as F
from torchaudio import transforms as T

from ..inference.generation import generate_diffusion_cond, generate_diffusion_uncond
from ..models.factory import create_model_from_config
from ..models.pretrained import get_pretrained_model
from ..models.utils import load_ckpt_state_dict
from ..inference.utils import prepare_audio
from ..training.utils import copy_state_dict
from ..data.utils import read_video, merge_video_audio

from PIL import Image

model_configurations = {}
device = torch.device("cpu")

os.environ['TMPDIR'] = '/aifs4su/data/tianzeyue/tmp'

current_model_name = None
current_model = None
current_sample_rate = None
current_sample_size = None
current_model_config = None


_SYNC_SIZE = 224
from torchvision.transforms import v2        
sync_transform = v2.Compose([
    v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
    v2.CenterCrop(_SYNC_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


try:
    from ..models.synchformer.features_utils import FeaturesUtils
    synchformer_ckpt = "model/synchformer_state_dict.pth"
    if os.path.exists(synchformer_ckpt):
        sync_feature_extractor = FeaturesUtils(
            tod_vae_ckpt='vae_path',
            enable_conditions=True,
            bigvgan_vocoder_ckpt='bigvgan_path',
            synchformer_ckpt=synchformer_ckpt,
            mode='44k'
        ).eval()
        if torch.cuda.is_available():
            sync_feature_extractor = sync_feature_extractor.cuda()
    else:
        sync_feature_extractor = None
        print(f"Warning: synchformer checkpoint not found at {synchformer_ckpt}, sync features will be zeros")
except Exception as e:
    sync_feature_extractor = None
    print(f"Warning: Could not initialize sync_feature_extractor: {e}, sync features will be zeros")

def adjust_video_duration(video_tensor, duration, target_fps):
    current_duration = video_tensor.shape[0]
    target_duration = duration * target_fps
    if current_duration > target_duration:
        video_tensor = video_tensor[:target_duration]
    elif current_duration < target_duration:
        last_frame = video_tensor[-1:]
        repeat_times = target_duration - current_duration
        video_tensor = torch.cat((video_tensor, last_frame.repeat(repeat_times, 1, 1, 1)), dim=0)
    return video_tensor



def video_read_local(filepath, seek_time=0., duration=-1, target_fps=2):
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png']:
        resize_transform = transforms.Resize((224, 224))
        image = Image.open(filepath).convert("RGB")
        frame = transforms.ToTensor()(image).unsqueeze(0)
        # Resize the image to 224x224
        frame = resize_transform(frame)
        target_frames = int(duration * target_fps)
        frame = frame.repeat(int(math.ceil(target_frames / frame.shape[0])), 1, 1, 1)[:target_frames]
        assert frame.shape[0] == target_frames, f"The shape of frame is {frame.shape}"
        return frame  # [N, C, H, W]

    vr = VideoReader(filepath, ctx=cpu(0))
    fps = vr.get_avg_fps()
    total_frames = len(vr)

    seek_frame = int(seek_time * fps)
    if duration > 0:
        total_frames_to_read = int(target_fps * duration)
        frame_interval = int(math.ceil(fps / target_fps))
        end_frame = min(seek_frame + total_frames_to_read * frame_interval, total_frames)
        frame_ids = list(range(seek_frame, end_frame, frame_interval))
    else:
        frame_interval = int(math.ceil(fps / target_fps))
        frame_ids = list(range(0, total_frames, frame_interval))

    frames = vr.get_batch(frame_ids).asnumpy()
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

    if frames.shape[2] != 224 or frames.shape[3] != 224:
        print(f'resizing...--->224x224')
        resize_transform = transforms.Resize((224, 224))
        frames = resize_transform(frames)

    video_tensor = adjust_video_duration(frames, duration, target_fps)

    assert video_tensor.shape[0] == duration * target_fps, f"The shape of video_tensor is {video_tensor.shape}"

    return video_tensor


def merge_video_audio(video_path, audio_path, output_path, start_time, duration):
    command = [
        'ffmpeg',
        '-y',                   # Overwrite output files without asking
        '-ss', str(start_time), # Start time
        '-t', str(duration),    # Duration
        '-i', video_path,       # Input video file
        '-i', audio_path,       # Input audio file
        '-c:v', 'copy',         # Copy the video codec (no re-encoding)
        '-c:a', 'aac',          # Use AAC audio codec
        '-map', '0:v:0',        # Map the video from the first input
        '-map', '1:a:0',        # Map the audio from the second input
        '-shortest',            # Stop encoding when the shortest input ends
        '-strict', 'experimental',  # Allow experimental codecs if needed
        output_path             # Output file path
    ]
    
    try:
        sp.run(command, check=True)
        print(f"Successfully merged audio and video into {output_path}")
        return output_path
    except sp.CalledProcessError as e:
        print(f"Error merging audio and video: {e}")
        return None

def load_model(model_name, model_config=None, model_ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, device="cuda", model_half=False):
    global model_configurations
    
    if pretrained_name is not None:
        print(f"Loading pretrained model {pretrained_name}")
        model, model_config = get_pretrained_model(pretrained_name)

    elif model_config is not None and model_ckpt_path is not None:
        print(f"Creating model from config")
        model = create_model_from_config(model_config)

        print(f"Loading model checkpoint from {model_ckpt_path}")
        # Load checkpoint
        copy_state_dict(model, load_ckpt_state_dict(model_ckpt_path))

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    if pretransform_ckpt_path is not None:
        print(f"Loading pretransform checkpoint from {pretransform_ckpt_path}")
        model.pretransform.load_state_dict(load_ckpt_state_dict(pretransform_ckpt_path), strict=False)
        print(f"Done loading pretransform")

    model.to(device).eval().requires_grad_(False)

    if model_half:
        model.to(torch.float16)
        
    print(f"Done loading model {model_name}")

    return model, model_config, sample_rate, sample_size


def load_and_process_audio(audio_path, sample_rate, seconds_start, seconds_total):
    audio_tensor, sr = torchaudio.load(audio_path)
    start_index = int(sample_rate * seconds_start)
    target_length = int(sample_rate * seconds_total)
    end_index = start_index + target_length
    audio_tensor = audio_tensor[:, start_index:end_index]
    if audio_tensor.shape[1] < target_length:
        pad_length = target_length - audio_tensor.shape[1]
        audio_tensor = F.pad(audio_tensor, (0, pad_length))
    return audio_tensor

def generate_cond(
        prompt,
        negative_prompt=None,
        video_file=None,
        video_path=None,
        audio_prompt_file=None,
        audio_prompt_path=None,
        seconds_start=0,
        seconds_total=10,
        cfg_scale=6.0,
        steps=250,
        preview_every=None,
        seed=-1,
        sampler_type="dpmpp-3m-sde",
        sigma_min=0.03,
        sigma_max=1000,
        cfg_rescale=0.0,
        use_init=False,
        init_audio=None,
        init_noise_level=1.0,
        mask_cropfrom=None,
        mask_pastefrom=None,
        mask_pasteto=None,
        mask_maskstart=None,
        mask_maskend=None,
        mask_softnessL=None,
        mask_softnessR=None,
        mask_marination=None,
        batch_size=1
    ):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print(f"Prompt: {prompt}")
    preview_images = []
    if preview_every == 0:
        preview_every = None

    try:
        has_mps = platform.system() == "Darwin" and torch.backends.mps.is_available()
    except Exception:
        has_mps = False
    if has_mps:
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model_name = 'default'
    if model_name not in model_configurations:
        raise ValueError(f"Model {model_name} configuration is not available.")
    
    cfg = model_configurations[model_name]
    model_config_path = cfg.get("model_config")
    ckpt_path = cfg.get("ckpt_path")
    pretrained_name = cfg.get("pretrained_name")
    pretransform_ckpt_path = cfg.get("pretransform_ckpt_path")
    model_type = cfg.get("model_type", "diffusion_cond")
    
    global current_model_name, current_model, current_sample_rate, current_sample_size, current_model_config
    if current_model is None or model_name != current_model_name:
        if model_config_path:
            with open(model_config_path) as f:
                model_config = json.load(f)
        else:
            model_config = None
        
        current_model, model_config, sample_rate, sample_size = load_model(
            model_name=model_name,
            model_config=model_config,
            model_ckpt_path=ckpt_path,
            pretrained_name=pretrained_name,
            pretransform_ckpt_path=pretransform_ckpt_path,
            device=device,
            model_half=False
        )
        current_model_name = model_name
        current_model_config = model_config
        model = current_model
        current_sample_rate = sample_rate
        current_sample_size = sample_size
    else:
        model = current_model
        sample_rate = current_sample_rate
        sample_size = current_sample_size
        model_config = current_model_config
    
    # Get target_fps from model_config after loading
    if model_config is not None:
        target_fps = model_config.get("video_fps", 5)
    else:
        target_fps = 5

    original_video_path = video_path
    if video_file is not None:
        video_path = video_file.name
    elif video_path:
        video_path = video_path.strip()
    else:
        video_path = None

    if audio_prompt_file is not None:
        print(f'audio_prompt_file: {audio_prompt_file}')
        audio_path = audio_prompt_file.name
    elif audio_prompt_path:
        audio_path = audio_prompt_path.strip()
    else:
        audio_path = None

    # target_fps=10
    if video_path is None and audio_path is None:
        mask_type = "mask_video_audio"
        Video_tensors = torch.zeros(int(target_fps * seconds_total), 3, 224, 224)
        audio_tensor = torch.zeros((2, int(sample_rate * seconds_total)))
        sync_features = torch.zeros(1, 240, 768).to(device)

    elif video_path is None:
        mask_type = "mask_video"
        Video_tensors = torch.zeros(int(target_fps * seconds_total), 3, 224, 224)
        sync_features = torch.zeros(1, 240, 768).to(device)
        try:
            audio_tensor = load_and_process_audio(audio_path, sample_rate, seconds_start, seconds_total)
        except Exception as e:
            print("Audio prompt file is empty or invalid, using zero audio tensor.")
            audio_tensor = torch.zeros((2, int(sample_rate * seconds_total)))
    elif audio_path is None:
        mask_type = "mask_audio"
        try:
            Video_tensors = read_video(video_path, seek_time=seconds_start, duration=seconds_total, target_fps=target_fps)

            sync_video_tensor = read_video(video_path, seek_time=seconds_start, duration=seconds_total, target_fps=25)
            if sync_feature_extractor is not None:
                try:
                    sync_video = sync_transform(sync_video_tensor)
                    sync_video = sync_video.unsqueeze(0).to(device)
                    sync_features = sync_feature_extractor.encode_video_with_sync(sync_video)
                except Exception as e:
                    print(f"Error processing sync video: {e}, using zero sync features")
                    sync_features = torch.zeros(1, 240, 768).to(device)
            else:
                sync_features = torch.zeros(1, 240, 768).to(device)
        except Exception as e:
            print("Video file is empty or invalid, using zero video tensor.")
            Video_tensors = torch.zeros((seconds_total * target_fps, 3, 224, 224))   
            sync_features = torch.zeros(1, 240, 768).to(device)         
        audio_tensor = torch.zeros((2, int(sample_rate * seconds_total)))
    else:
        mask_type = None
        try:
            Video_tensors = read_video(video_path, seek_time=seconds_start, duration=seconds_total, target_fps=target_fps)

            sync_video_tensor = read_video(video_path, seek_time=seconds_start, duration=seconds_total, target_fps=25)
            if sync_feature_extractor is not None:
                try:
                    sync_video = sync_transform(sync_video_tensor)
                    sync_video = sync_video.unsqueeze(0).to(device)
                    sync_features = sync_feature_extractor.encode_video_with_sync(sync_video)
                except Exception as e:
                    print(f"Error processing sync video: {e}, using zero sync features")
                    sync_features = torch.zeros(1, 240, 768).to(device)
            else:
                sync_features = torch.zeros(1, 240, 768).to(device) 

        except Exception as e:
            print("Video file is empty or invalid, using zero video tensor.")
            Video_tensors = torch.zeros((seconds_total * target_fps, 3, 224, 224))
            sync_features = torch.zeros(1, 240, 768).to(device)
        try:
            audio_tensor = load_and_process_audio(audio_path, sample_rate, seconds_start, seconds_total)
        except Exception as e:
            print("Audio prompt file is empty or invalid, using zero audio tensor.")
            audio_tensor = torch.zeros((2, int(sample_rate * seconds_total)))
            
    # if sync_feature_path is not None and os.path.exists(sync_feature_path):
    #     sync_features = torch.load(sync_feature_path, weights_only=True, map_location='cpu').to(device)            
    # else:
    #     sync_features = torch.zeros(1, 240, 768).to(device)
    
    # try:
    #     sync_features = torch.load(sync_feature_path, weights_only=True, map_location='cpu').to(device)
    # except:
    #     sync_video_tensor = video_read_local(video_path, seek_time=seconds_start, duration=seconds_total, target_fps=25)
    #     sync_video=sync_transform(sync_video_tensor)
    #     sync_video = sync_video.unsqueeze(0).to(device)
    #     sync_features = sync_feature_extractor.encode_video_with_sync(sync_video) 



    audio_tensor=audio_tensor.to(device)
    seconds_input=sample_size/sample_rate
    print(f'video_path: {video_path}')
    print(f'audio_path: {audio_path}')

    
    # Use default or empty string if prompt is not provided
    if not prompt:
        prompt = ""
    
    conditioning = [{
        # "video_prompt": [Video_tensors.unsqueeze(0)],        
        "video_prompt": {"video_tensors":Video_tensors.unsqueeze(0), "video_sync_frames": sync_features},        
        "text_prompt": prompt,
        "audio_prompt": audio_tensor.unsqueeze(0),
        "seconds_start": seconds_start,
        "seconds_total": seconds_input
    }] * batch_size
    if negative_prompt:
        negative_conditioning = [{
            "video_prompt": [Video_tensors.unsqueeze(0)],        
            "text_prompt": negative_prompt,
            "audio_prompt": audio_tensor.unsqueeze(0),
            "seconds_start": seconds_start,
            "seconds_total": seconds_total
        }] * batch_size
    else:
        negative_conditioning = None

    print(f"Model type: {model_type}")

    try:
        device = next(model.parameters()).device 
    except Exception as e:
        device = next(current_model.parameters()).device

    seed = int(seed)

    if not use_init:
        init_audio = None

    input_sample_size = sample_size

    if init_audio is not None:
        in_sr, init_audio = init_audio
        init_audio = torch.from_numpy(init_audio).float().div(32767)
        
        if init_audio.dim() == 1:
            init_audio = init_audio.unsqueeze(0)  # [1, n]
        elif init_audio.dim() == 2:
            init_audio = init_audio.transpose(0, 1)  # [n, 2] -> [2, n]

        if in_sr != sample_rate:
            resample_tf = T.Resample(in_sr, sample_rate).to(init_audio.device)
            init_audio = resample_tf(init_audio)

        audio_length = init_audio.shape[-1]

        if audio_length > sample_size:
            input_sample_size = audio_length + (model.min_input_length - (audio_length % model.min_input_length)) % model.min_input_length

        init_audio = (sample_rate, init_audio)

    def progress_callback(callback_info):
        nonlocal preview_images
        denoised = callback_info["denoised"]
        current_step = callback_info["i"]
        sigma = callback_info["sigma"]

        if (current_step - 1) % preview_every == 0:
            if model.pretransform is not None:
                denoised = model.pretransform.decode(denoised)
            denoised = rearrange(denoised, "b d n -> d (b n)")
            denoised = denoised.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            audio_spectrogram = audio_spectrogram_image(denoised, sample_rate=sample_rate)
            preview_images.append((audio_spectrogram, f"Step {current_step} sigma={sigma:.3f})"))

    if mask_cropfrom is not None: 
        mask_args = {
            "cropfrom": mask_cropfrom,
            "pastefrom": mask_pastefrom,
            "pasteto": mask_pasteto,
            "maskstart": mask_maskstart,
            "maskend": mask_maskend,
            "softnessL": mask_softnessL,
            "softnessR": mask_softnessR,
            "marination": mask_marination,
        }
    else:
        mask_args = None 

    if model_type == "diffusion_cond":
      
        audio = generate_diffusion_cond(
            model, 
            conditioning=conditioning,
            negative_conditioning=negative_conditioning,
            steps=steps,
            cfg_scale=cfg_scale,
            batch_size=batch_size,
            sample_size=input_sample_size,
            sample_rate=sample_rate,
            seed=seed,
            device=device,
            sampler_type=sampler_type,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            init_audio=init_audio,
            init_noise_level=init_noise_level,
            mask_args=mask_args,
            callback=progress_callback if preview_every is not None else None,
            scale_phi=cfg_rescale
        )
    elif model_type == "diffusion_uncond":
        audio = generate_diffusion_uncond(
            model, 
            steps=steps,
            batch_size=batch_size,
            sample_size=input_sample_size,
            seed=seed,
            device=device,
            sampler_type=sampler_type,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            init_audio=init_audio,
            init_noise_level=init_noise_level,
            callback=progress_callback if preview_every is not None else None
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    audio = rearrange(audio, "b d n -> d (b n)")
    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    
    output_dir = "demo_result"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    audio_path = f"{output_dir}/output.wav"
    torchaudio.save(audio_path, audio, sample_rate)

    file_name = os.path.basename(original_video_path) if original_video_path else "output"
    output_video_path = f"{output_dir}/{file_name}"
        
    if original_video_path:
        merge_video_audio(original_video_path, audio_path, output_video_path, seconds_start, seconds_total)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    gc.collect()

    return (output_video_path, audio_path)

def toggle_custom_model(selected_model):
    return gr.Row.update(visible=(selected_model == "Custom Model"))



def create_sampling_ui(model_options, model_config_map, inpainting=False):
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # 🎧AudioX: A Unified Framework for Anything-to-Audio Generation
            **[Project Page](https://zeyuet.github.io/AudioX/) · [Huggingface](https://huggingface.co/Zeyue7/AudioX) · [GitHub](https://github.com/ZeyueT/AudioX)**
            """
        )

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(show_label=False, placeholder="Enter your prompt")
                negative_prompt = gr.Textbox(show_label=False, placeholder="Negative prompt", visible=False)
                video_path = gr.Textbox(label="Video Path", placeholder="Enter video file path")
                video_file = gr.File(label="Upload Video File")
                audio_prompt_file = gr.File(label="Upload Audio Prompt File", visible=False)
                audio_prompt_path = gr.Textbox(label="Audio Prompt Path", placeholder="Enter audio file path", visible=False)

        with gr.Row():
            with gr.Column(scale=6):
                with gr.Accordion("Video Params", open=False):                
                    seconds_start_slider = gr.Slider(minimum=0, maximum=512, step=1, value=0, label="Video Seconds Start")
                    seconds_total_slider = gr.Slider(minimum=0, maximum=10, step=1, value=10, label="Seconds Total", interactive=False)

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Accordion("Sampler Params", open=False):
                    steps_slider = gr.Slider(minimum=1, maximum=500, step=1, value=100, label="Steps")
                    preview_every_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Preview Every")
                    cfg_scale_slider = gr.Slider(minimum=0.0, maximum=25.0, step=0.1, value=7.0, label="CFG Scale")
                    seed_textbox = gr.Textbox(label="Seed (set to -1 for random seed)", value="-1")
                    sampler_type_dropdown = gr.Dropdown(
                        ["dpmpp-2m-sde", "dpmpp-3m-sde", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast"],
                        label="Sampler Type",
                        value="dpmpp-3m-sde"
                    )
                    sigma_min_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.03, label="Sigma Min")
                    sigma_max_slider = gr.Slider(minimum=0.0, maximum=1000.0, step=0.1, value=500, label="Sigma Max")
                    cfg_rescale_slider = gr.Slider(minimum=0.0, maximum=1, step=0.01, value=0.0, label="CFG Rescale Amount")
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Accordion("Init Audio", open=False, visible=False):
                    init_audio_checkbox = gr.Checkbox(label="Use Init Audio")
                    init_audio_input = gr.Audio(label="Init Audio")
                    init_noise_level_slider = gr.Slider(minimum=0.1, maximum=100.0, step=0.01, value=0.1, label="Init Noise Level")

        if inpainting: 
            with gr.Accordion("Inpainting", open=False):
                mask_cropfrom_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=0, label="Crop From %")
                mask_pastefrom_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=0, label="Paste From %")
                mask_pasteto_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=100, label="Paste To %")
                mask_maskstart_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=50, label="Mask Start %")
                mask_maskend_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=100, label="Mask End %")
                mask_softnessL_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=0, label="Softmask Left Crossfade Length %")
                mask_softnessR_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=0, label="Softmask Right Crossfade Length %")
                mask_marination_slider = gr.Slider(minimum=0.0, maximum=1, step=0.0001, value=0, label="Marination Level", visible=False)

        with gr.Row():
            generate_button = gr.Button("Generate", variant='primary', scale=1)

        with gr.Row():
            with gr.Column(scale=6):
                video_output = gr.Video(label="Output Video", interactive=False)
                audio_output = gr.Audio(label="Output Audio", interactive=False)
                send_to_init_button = gr.Button("Send to Init Audio", scale=1, visible=False)
        send_to_init_button.click(
            fn=lambda audio: audio,
            inputs=[audio_output],
            outputs=[init_audio_input]
        )

        inputs = [
            prompt, 
            negative_prompt,
            video_file,
            video_path,
            audio_prompt_file,
            audio_prompt_path,
            seconds_start_slider, 
            seconds_total_slider, 
            cfg_scale_slider, 
            steps_slider, 
            preview_every_slider, 
            seed_textbox, 
            sampler_type_dropdown, 
            sigma_min_slider, 
            sigma_max_slider,
            cfg_rescale_slider,
            init_audio_checkbox,
            init_audio_input,
            init_noise_level_slider
        ]
        
        if inpainting:
            inputs.extend([
                mask_cropfrom_slider,
                mask_pastefrom_slider,
                mask_pasteto_slider,
                mask_maskstart_slider,
                mask_maskend_slider,
                mask_softnessL_slider,
                mask_softnessR_slider,
                mask_marination_slider
            ])

        generate_button.click(
            fn=generate_cond, 
            inputs=inputs,
            outputs=[
                video_output,
                audio_output
            ], 
            api_name="generate"
        )

        return demo
    
def create_txt2audio_ui(model_options, model_config_map):
    with gr.Blocks() as ui:
        with gr.Tab("Generation"):
            create_sampling_ui(model_options, model_config_map)
        with gr.Tab("Inpainting"):
            create_sampling_ui(model_options, model_config_map, inpainting=True)    
    return ui

def toggle_custom_model(selected_model):
    return gr.Row.update(visible=(selected_model == "Custom Model"))

def create_ui(model_config_path=None, ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, model_half=False):
    global model_configurations
    global device

    try:
        has_mps = platform.system() == "Darwin" and torch.backends.mps.is_available()
    except Exception:
        has_mps = False

    if has_mps:
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    model_configurations = {}
    
    if model_config_path is not None and ckpt_path is not None:
        model_configurations["default"] = {
            "model_config": model_config_path,
            "ckpt_path": ckpt_path,
            "pretrained_name": pretrained_name,
            "pretransform_ckpt_path": pretransform_ckpt_path,
            "model_type": "diffusion_cond"
        }
        model_options = {"default": {}}
    elif pretrained_name is not None:
        model_configurations["default"] = {
            "model_config": None,
            "ckpt_path": None,
            "pretrained_name": pretrained_name,
            "pretransform_ckpt_path": pretransform_ckpt_path,
            "model_type": "diffusion_cond"
        }
        model_options = {"default": {}}
    else:
        model_configurations["default"] = {
            "model_config": model_config_path,
            "ckpt_path": ckpt_path,
            "pretrained_name": pretrained_name,
            "pretransform_ckpt_path": pretransform_ckpt_path,
            "model_type": "diffusion_cond"
        }
        model_options = {"default": {}}

    ui = create_txt2audio_ui(model_options, model_configurations)
    return ui

if __name__ == "__main__":
    ui = create_ui(
        model_config_path='/aifs4su/data/tianzeyue/project/stable/stable-audio-tools/stable-audio-open-1.0/models_config.json',
        share=True
    )
    ui.launch()
