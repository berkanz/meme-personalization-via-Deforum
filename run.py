import json
import sys, time, gc
import shutil
import re
import numpy as np
import os 
from utils import *
sys.path.extend([
    'third_party/deforum-stable-diffusion/',
    'third_party/deforum-stable-diffusion/src',
])
import torch
import random
from IPython import display
from types import SimpleNamespace
from helpers.save_images import get_output_folder
from helpers.settings import load_args
from helpers.render import render_animation, render_input_video, render_image_batch, render_interpolation
from helpers.model_load import load_model, get_model_output_paths
import argparse

DEFAULT_MODEL = "v1-5-pruned-emaonly.ckpt"
DEFAULT_MODEL_CONFIG = "v1-inference.yaml"
DEFAULT_SETTINGS = "./examples/runSettings_StillImages.txt"

template = "michael_scott"
model_path = "myself.ckpt"

def cl_parser():
    parser = argparse.ArgumentParser(description="Animation properties")
    parser.add_argument('--meme_template', default="michael_scott", type=str, help='name of the meme template (e.g. michael_scott)')
    parser.add_argument('--fine_tuned_model_path', default="myself.ckpt", type=str, help='path to the fine-tuned model')
    arguments = parser.parse_args()
    return arguments


def load_file_args(path):
    with open(path, "r") as f:
        loaded_args = json.load(f)#, ensure_ascii=False, indent=4)
    return loaded_args

def Root():
        models_path = "models" #@param {type:"string"}
        configs_path = "third_party/deforum-stable-diffusion/configs/" #@param {type:"string"}
        output_path = "output/" #@param {type:"string"}
        mount_google_drive = False #@param {type:"boolean"}

        #@markdown **Model Setup**
        model_config = DEFAULT_MODEL_CONFIG #@param ["custom","v1-inference.yaml"]
        model_checkpoint =  model_path #@param ["custom","v1-5-pruned.ckpt","v1-5-pruned-emaonly.ckpt","sd-v1-4-full-ema.ckpt","sd-v1-4.ckpt","sd-v1-3-full-ema.ckpt","sd-v1-3.ckpt","sd-v1-2-full-ema.ckpt","sd-v1-2.ckpt","sd-v1-1-full-ema.ckpt","sd-v1-1.ckpt", "robo-diffusion-v1.ckpt","wd-v1-3-float16.ckpt"]
        custom_config_path = "" #@param {type:"string"}
        custom_checkpoint_path = "" #@param {type:"string"}
        half_precision = True
        return locals()
    
def DeforumAnimArgs(master_args, root):
    
    ENABLE_STORY_MODE = master_args["ENABLE_STORY_MODE"] == "True"
    if ENABLE_STORY_MODE == True:
        animation_mode = master_args["animation_mode"] #@param ['None', '2D', '3D', 'Video Input', 'Interpolation'] {type:'string'}
        max_frames = master_args["max_frames"] #@param {type:"number"}
        border = master_args["border"] #@param ['wrap', 'replicate'] {type:'string'}

        #@markdown ####**Motion Parameters:**
        angle = master_args["angle"]#@param {type:"string"}
        zoom = master_args["zoom"] #@param {type:"string"}
        translation_x = master_args["translation_x"] #@param {type:"string"}
        translation_y = master_args["translation_y"] #@param {type:"string"}
        translation_z = master_args["translation_z"] #@param {type:"string"}
        rotation_3d_x = master_args["rotation_3d_x"] #@param {type:"string"}
        rotation_3d_y = master_args["rotation_3d_y"] #@param {type:"string"}
        rotation_3d_z = master_args["rotation_3d_z"] #@param {type:"string"}
        flip_2d_perspective = master_args["flip_2d_perspective"] #@param {type:"boolean"}
        perspective_flip_theta = master_args["perspective_flip_theta"] #@param {type:"string"}
        perspective_flip_phi = master_args["perspective_flip_phi"] #@param {type:"string"}
        perspective_flip_gamma = master_args["perspective_flip_gamma"] #@param {type:"string"}
        perspective_flip_fv = master_args["perspective_flip_fv"] #@param {type:"string"}
        noise_schedule = master_args["noise_schedule"] #@param {type:"string"}
        strength_schedule = master_args["strength_schedule"] #@param {type:"string"}
        contrast_schedule = master_args["contrast_schedule"] #@param {type:"string"}

        hybrid_video_comp_alpha_schedule = "0:(1)" #@param {type:"string"}
        hybrid_video_comp_mask_blend_alpha_schedule = "0:(0.5)" #@param {type:"string"}
        hybrid_video_comp_mask_contrast_schedule = "0:(1)" #@param {type:"string"}
        hybrid_video_comp_mask_auto_contrast_cutoff_high_schedule =  "0:(100)" #@param {type:"string"}
        hybrid_video_comp_mask_auto_contrast_cutoff_low_schedule =  "0:(0)" #@param {type:"string"}

        #@markdown ####**Unsharp mask (anti-blur) Parameters:**
        kernel_schedule = "0: (5)"#@param {type:"string"}
        sigma_schedule = "0: (1.0)"#@param {type:"string"}
        amount_schedule = "0: (0.2)"#@param {type:"string"}
        threshold_schedule = "0: (0.0)"#@param {type:"string"}
        
        #@markdown ####**Coherence:**
        color_coherence = master_args["color_coherence"] #@param ['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB'] {type:'string'}
        diffusion_cadence = master_args["diffusion_cadence"] #@param ['1','2','3','4','5','6','7','8'] {type:'string'}

        #@markdown #### 3D Depth Warping
        use_depth_warping = master_args["use_depth_warping"] #@param {type:"boolean"}
        midas_weight = master_args["midas_weight"] #@param {type:"number"}
        near_plane = master_args["near_plane"]
        far_plane = master_args["far_plane"]
        fov = master_args["fov"] #@param {type:"number"}
        padding_mode = master_args["padding_mode"] #@param ['border', 'reflection', 'zeros'] {type:'string'}
        sampling_mode = master_args["sampling_mode"] #@param ['bicubic', 'bilinear', 'nearest'] {type:'string'}
        save_depth_maps = master_args["save_depth_maps"] #@param {type:"boolean"}

        #@markdown ####**Video Input:**
        video_init_path = master_args["video_init_path"] #@param {type:"string"}
        extract_nth_frame = master_args["extract_nth_frame"] #@param {type:"number"}
        overwrite_extracted_frames = master_args["overwrite_extracted_frames"] #@param {type:"boolean"}
        use_mask_video = master_args["use_mask_video"] #@param {type:"boolean"}
        video_mask_path = master_args["video_mask_path"] #@param {type:"string"}

        #@markdown ####**Hybrid Video for 2D/3D Animation Mode:**
        hybrid_video_generate_inputframes = False #@param {type:"boolean"}
        hybrid_video_use_first_frame_as_init_image = True #@param {type:"boolean"}
        hybrid_video_motion = "None" #@param ['None','Optical Flow','Perspective','Affine']
        hybrid_video_flow_method = "Farneback" #@param ['Farneback','DenseRLOF','SF']
        hybrid_video_composite = False #@param {type:"boolean"}
        hybrid_video_comp_mask_type = "None" #@param ['None', 'Depth', 'Video Depth', 'Blend', 'Difference']
        hybrid_video_comp_mask_inverse = False #@param {type:"boolean"}
        hybrid_video_comp_mask_equalize = "None" #@param  ['None','Before','After','Both']
        hybrid_video_comp_mask_auto_contrast = False #@param {type:"boolean"}
        hybrid_video_comp_save_extra_frames = False #@param {type:"boolean"}
        hybrid_video_use_video_as_mse_image = False #@param {type:"boolean"}

        #@markdown ####**Interpolation:**
        interpolate_key_frames = master_args["interpolate_key_frames"] #@param {type:"boolean"}
        interpolate_x_frames = master_args["interpolate_x_frames"] #@param {type:"number"}
        
        #@markdown ####**Resume Animation:**
        resume_from_timestring = master_args["resume_from_timestring"] #@param {type:"boolean"}
        resume_timestring = master_args["resume_timestring"] #@param {type:"string"}

    else:
        #@markdown ####**Animation:**
        animation_mode = 'None' #@param ['None', '2D', '3D', 'Video Input', 'Interpolation'] {type:'string'}
        max_frames = 100 #@param {type:"number"}
        border = 'replicate' #@param ['wrap', 'replicate'] {type:'string'}

        #@markdown ####**Motion Parameters:**
        angle = "0:(0)"#@param {type:"string"}
        zoom = "0:(1.04)"#@param {type:"string"}
        translation_x = "0:(0)"#@param {type:"string"}
        translation_y = "0:(0)"#@param {type:"string"}
        translation_z = "0:(10)"#@param {type:"string"}
        rotation_3d_x = "0:(0)"#@param {type:"string"}
        rotation_3d_y = "0:(0)"#@param {type:"string"}
        rotation_3d_z = "0:(0)"#@param {type:"string"}
        flip_2d_perspective = False #@param {type:"boolean"}
        perspective_flip_theta = "0:(0)"#@param {type:"string"}
        perspective_flip_phi = "0:(t%15)"#@param {type:"string"}
        perspective_flip_gamma = "0:(0)"#@param {type:"string"}
        perspective_flip_fv = "0:(53)"#@param {type:"string"}
        noise_schedule = "0: (0.02)"#@param {type:"string"}
        strength_schedule = "0: (0.65)"#@param {type:"string"}
        contrast_schedule = "0: (1.0)"#@param {type:"string"}

        #@markdown ####**Coherence:**
        color_coherence = 'Match Frame 0 LAB' #@param ['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB'] {type:'string'}
        diffusion_cadence = '3' #@param ['1','2','3','4','5','6','7','8'] {type:'string'}

        #@markdown ####**3D Depth Warping:**
        use_depth_warping = True #@param {type:"boolean"}
        midas_weight = 0.3#@param {type:"number"}
        near_plane = 200
        far_plane = 10000
        fov = 40#@param {type:"number"}
        padding_mode = 'border'#@param ['border', 'reflection', 'zeros'] {type:'string'}
        sampling_mode = 'bicubic'#@param ['bicubic', 'bilinear', 'nearest'] {type:'string'}
        save_depth_maps = False #@param {type:"boolean"}

        #@markdown ####**Video Input:**
        video_init_path ='/content/video_in.mp4'#@param {type:"string"}
        extract_nth_frame = 1#@param {type:"number"}
        overwrite_extracted_frames = True #@param {type:"boolean"}
        use_mask_video = False #@param {type:"boolean"}
        video_mask_path ='/content/video_in.mp4'#@param {type:"string"}

        #@markdown ####**Interpolation:**
        interpolate_key_frames = False #@param {type:"boolean"}
        interpolate_x_frames = 4 #@param {type:"number"}
        
        #@markdown ####**Resume Animation:**
        resume_from_timestring = False #@param {type:"boolean"}
        resume_timestring = "20220829210106" #@param {type:"string"}

    return locals()

def DeforumArgs(master_args, root):
    ENABLE_STORY_MODE = master_args["ENABLE_STORY_MODE"] == "True"

    #@markdown **Image Settings**
    W = master_args["width"] #@param
    H = master_args["height"] #@param
    W, H = map(lambda x: x - x % 64, (W, H))  # resize to integer multiple of 64
    bit_depth_output = master_args["bit_depth_output"] #@param [8, 16, 32] {type:"raw"}

    #@markdown **Sampling Settings**
    seed = master_args["seed"] #@param
    sampler = master_args["sampler"] #@param ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim", "dpm_fast", "dpm_adaptive", "dpmpp_2s_a", "dpmpp_2m"]
    steps = master_args["steps"] #@param
    scale = master_args["scale"] #@param
    ddim_eta = master_args["ddim_eta"] #@param
    dynamic_threshold = None
    static_threshold = None   
    
    #@markdown **CLIP\Aesthetics Conditional Settings**
    clip_name = master_args["clip_name"] #@param ['ViT-L/14', 'ViT-L/14@336px', 'ViT-B/16', 'ViT-B/32']
    clip_scale = master_args["clip_scale"] #@param {type:"number"}
    aesthetics_scale = master_args["aesthetics_scale"] #@param {type:"number"}
    cutn = master_args["cutn"] #@param {type:"number"}
    cut_pow = master_args["cut_pow"] #@param {type:"number"}

    #@markdown **Save & Display Settings**
    save_samples = True #@param {type:"boolean"}
    save_settings = False #@param {type:"boolean"}
    display_samples = True #@param {type:"boolean"}
    save_sample_per_step = False #@param {type:"boolean"}
    show_sample_per_step = False #@param {type:"boolean"}

    #@markdown **Prompt Settings**
    prompt_weighting = True #@param {type:"boolean"}
    normalize_prompt_weights = True #@param {type:"boolean"}
    log_weighted_subprompts = False #@param {type:"boolean"}

    #@markdown **Batch Settings**
    n_batch = master_args["n_batch"]  #@param
    batch_name = master_args["batch_name"] #@param {type:"string"}
    filename_format = master_args["filename_format"] #@param ["{timestring}_{index}_{seed}.png","{timestring}_{index}_{prompt}.png"]
        
    seed_behavior = master_args["seed_behavior"] #@param ["iter","fixed","random","ladder","alternate"]
    seed_iter_N = 1 #@param {type:'integer'}

    make_grid = False #@param {type:"boolean"}
    grid_rows = 2 #@param 
    outdir = get_output_folder(root.output_path, "")
    
    #@markdown **Init Settings**
    use_init = master_args["use_init"] #@param {type:"boolean"}
    strength = master_args["strength"] #@param {type:"number"}
    strength_0_no_init = True # Set the strength to 0 automatically when no init image is used
    init_image = master_args["init_image"] #@param {type:"string"} 
    # Whiter areas of the mask are areas that change more
    use_mask = master_args["use_mask"] #@param {type:"boolean"}
    use_alpha_as_mask = master_args["use_alpha_as_mask"] # use the alpha channel of the init image as the mask
    mask_file = master_args["mask_file"] #@param {type:"string"}
    invert_mask = master_args["invert_mask"] #@param {type:"boolean"}
    # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
    mask_brightness_adjust = master_args["mask_brightness_adjust"] #@param {type:"number"}
    mask_contrast_adjust = master_args["mask_contrast_adjust"]  #@param {type:"number"}

    # Overlay the masked image at the end of the generation so it does not get degraded by encoding and decoding
    overlay_mask = master_args["overlay_mask"]  # {type:"boolean"}
    # Blur edges of final overlay mask, if used. Minimum = 0 (no blur)
    mask_overlay_blur = master_args["mask_overlay_blur"] # {type:"number"}

    #@markdown **Exposure/Contrast Conditional Settings**
    mean_scale = master_args["mean_scale"] #@param {type:"number"}
    var_scale = master_args["var_scale"] #@param {type:"number"}
    exposure_scale = master_args["exposure_scale"] #@param {type:"number"}
    exposure_target = master_args["exposure_target"] #@param {type:"number"}

    #@markdown **Color Match Conditional Settings**
    colormatch_scale = master_args["colormatch_scale"] #@param {type:"number"}
    colormatch_image = master_args["colormatch_image"] #@param {type:"string"}
    colormatch_n_colors = master_args["colormatch_n_colors"] #@param {type:"number"}
    ignore_sat_weight = master_args["ignore_sat_weight"] #@param {type:"number"}

    cutn = master_args["cutn"] #@param {type:"number"}
    cut_pow = master_args["cut_pow"] #@param {type:"number"}

    #@markdown **Other Conditional Settings**
    init_mse_scale = master_args["init_mse_scale"] #@param {type:"number"}
    init_mse_image = master_args["init_mse_image"] #@param {type:"string"}

    blue_scale = master_args["blue_scale"] #@param {type:"number"}
    
    #@markdown **Conditional Gradient Settings**
    gradient_wrt = master_args["gradient_wrt"] #@param ["x", "x0_pred"]
    gradient_add_to = master_args["gradient_add_to"] #@param ["cond", "uncond", "both"]
    decode_method = master_args["decode_method"] #@param ["autoencoder","linear"]
    grad_threshold_type = master_args["grad_threshold_type"] #@param ["dynamic", "static", "mean", "schedule"]
    clamp_grad_threshold = master_args["clamp_grad_threshold"] #@param {type:"number"}
    clamp_start = master_args["clamp_start"] #@param
    clamp_stop = master_args["clamp_stop"] #@param
    grad_inject_timing = list(range(1,10)) #@param

    #@markdown **Speed vs VRAM Settings**
    cond_uncond_sync = master_args["cond_uncond_sync"] #@param {type:"boolean"}

    n_samples = 1 # doesnt do anything
    precision = 'autocast' 
    C = 4
    f = 8

    prompt = ""
    timestring = ""
    init_latent = None
    init_sample = None
    init_sample_raw = None
    mask_sample = None
    init_c = None
    seed_internal = 0

    return locals()


def personalize_meme(template, model_path):

    root = Root()
    root = SimpleNamespace(**root)

    root.models_path, root.output_path = get_model_output_paths(root)
    root.model, root.device = load_model(root, 
                                        load_on_run_all=True
                                        , 
                                        check_sha256=False
                                        )

    
    master_args = load_file_args("templates/" + template + "/settings.txt")
    template_label = template.strip(".txt")
    prompts = master_args["prompts"]

    if master_args["ENABLE_STORY_MODE"] == "True":
        animation_prompts = master_args["animation_prompts"]
    else:
        animation_prompts = {}

    args_dict = DeforumArgs(master_args, root)
    anim_args_dict = DeforumAnimArgs(master_args,root)
    args = SimpleNamespace(**args_dict)
    anim_args = SimpleNamespace(**anim_args_dict)
    args.timestring = time.strftime('%Y%m%d%H%M%S')
    args.strength = max(0.0, min(1.0, args.strength))

    
    if args.seed == -1:
        args.seed = random.randint(0, 2**32 - 1)
    if not args.use_init:
        args.init_image = None
    if args.sampler == 'plms' and (args.use_init or anim_args.animation_mode != 'None'):
        print(f"Init images aren't supported with PLMS yet, switching to KLMS")
        args.sampler = 'klms'
    if args.sampler != 'ddim':
        args.ddim_eta = 0

    if anim_args.animation_mode == 'None':
        anim_args.max_frames = 1
    elif anim_args.animation_mode == 'Video Input':
        args.use_init = True

    # clean up unused memory
    gc.collect()
    torch.cuda.empty_cache()

    # dispatch to appropriate renderer
    if anim_args.animation_mode == '2D' or anim_args.animation_mode == '3D':
        render_animation(args, anim_args, animation_prompts, root)
    elif anim_args.animation_mode == 'Video Input':
        render_input_video(args, anim_args, animation_prompts, root)
    elif anim_args.animation_mode == 'Interpolation':
        render_interpolation(args, anim_args, animation_prompts)
    else:
        render_image_batch(args, prompts, root)

    skip_video_for_run_all = False #@param {type: 'boolean'}
    fps = master_args["fps"] #@param {type:"number"}
    #reverse_loop = master_args["reverse_loop"]
    #@markdown **Manual Settings**
    use_manual_settings = False #@param {type:"boolean"}
    render_steps = False  #@param {type: 'boolean'}
    path_name_modifier = "x0_pred" #@param ["x0_pred","x"]

    if skip_video_for_run_all == True or args.ENABLE_STORY_MODE == False:
        print('Skipping video creation')
        return root.output_path
    else:
        import os
        import subprocess
        from base64 import b64encode

        if use_manual_settings:
            max_frames = "200" #@param {type:"string"}
        else:
            if render_steps: # render steps from a single image
                fname = f"{path_name_modifier}_%05d.png"
                all_step_dirs = [os.path.join(args.outdir, d) for d in os.listdir(args.outdir) if os.path.isdir(os.path.join(args.outdir,d))]
                newest_dir = max(all_step_dirs, key=os.path.getmtime)
                image_path = os.path.join(newest_dir, fname)
                print(f"Reading images from {image_path}")
                mp4_path = os.path.join(newest_dir, f"{args.timestring}_{path_name_modifier}.mp4")
                max_frames = str(args.steps)
            else: # render images for a video
                
                image_path = os.path.join(args.outdir, f"{args.timestring}_%05d.png")    
                mp4_path = os.path.join(args.outdir, f"{template_label}.mp4")
                max_frames = str(anim_args.max_frames)
                
        # make video
        cmd = [
            'ffmpeg',
            '-y',
            '-vcodec', 'png',
            '-r', str(fps),
            '-start_number', str(0),
            '-i', image_path,
            '-frames:v', max_frames,
            '-c:v', 'libx264',
            '-vf',
            f'fps={fps}',
            '-pix_fmt', 'yuv420p',
            '-crf', '17',
            '-preset', 'veryfast',
            '-pattern_type', 'sequence',
            mp4_path
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(stderr)
            raise RuntimeError(stderr)

        mp4 = open(mp4_path,'rb').read()
        data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
        display.display(display.HTML(f'<video controls loop><source src="{data_url}" type="video/mp4"></video>') )
        
        cleanup_after_render(args.outdir)
     
    return mp4_path

if __name__ == "__main__":
    
    cl_args = cl_parser()
    meme_path = personalize_meme(cl_args.meme_template, cl_args.fine_tuned_model_path)