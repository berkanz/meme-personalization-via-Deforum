import cv2
import matplotlib.pyplot as plt
import numpy as np
import shutil
from tqdm import tqdm
import subprocess
from base64 import b64encode
import os
import argparse
import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
ins = instanceSegmentation()
ins.load_model("models/pointrend_resnet50.pkl")
target_classes = ins.select_target_classes(person=True)

def cl_parser():
    parser = argparse.ArgumentParser(description="Animation properties")
    parser.add_argument('--video_path', default="source.mp4", type=str, help='name of the meme template (e.g. michael_scott)')
    parser.add_argument('--save_path', default="mask.mp4", type=str, help='path to the fine-tuned model')
    arguments = parser.parse_args()
    return arguments
    
def cleanup(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    return

def create_folder(folder):
    isExist = os.path.exists(folder)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(folder)

def extract_frames(video_path, save_path):
    create_folder(save_path)
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    frame_count = 0
    while success:
        cv2.imwrite(f"{save_path}frame{frame_count}.jpg", image)     # save frame as JPEG file      
        success,image = vidcap.read()
        frame_count += 1
    return frame_count

def max_nonzero_channel(arr):
    # Sum across all dimensions except for the channel dimension
    channel_sums = np.sum(arr != 0, axis=tuple(range(arr.ndim-1)))
    # Find the index of the channel with the highest sum
    max_channel_index = np.argmax(channel_sums)
    return max_channel_index

def frame_segmentation(path_to_extracted_frames, frame_count, save_path_for_mask_frames):
    create_folder(save_path_for_mask_frames) # create the folder if not already exists
    for index in tqdm(range(frame_count)):
        # run PointRend for each frame of the video
        r, output = ins.segmentImage(f"{path_to_extracted_frames}frame{index}.jpg", 
                                     show_bboxes=True,segment_target_classes = target_classes, 
                                     save_extracted_objects = False, mask_points_values = False, 
                                     extract_segmented_objects = True, output_image_name=None)
        # pick the channel with highes nonzero pixel count
        if r["masks"].ndim==3:
            picked_object = int(max_nonzero_channel(r["masks"][:,:,:]))
            cv2.imwrite(f"{save_path_for_mask_frames}mask{index:05}.png",(1-r["masks"][:,:,picked_object]).astype(int)*255)
        else:
            cv2.imwrite(f"{save_path_for_mask_frames}mask{index:05}.png",np.ones((output.shape)).astype(int)*255)

def save_mask_video(count, path_to_frames="extracted_masks/", mp4_path="mask.mp4"):
    image_path = os.path.join(path_to_frames, f"mask%05d.png")  
    fps = 30
    # make video
    cmd = [
            'ffmpeg',
            '-y',
            '-vcodec', 'png',
            '-r', str(fps),
            '-start_number', str(0),
            '-i', image_path,
            '-frames:v', str(count),
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
    return 

if __name__ == "__main__":
    
    cl_args = cl_parser()
    frame_count =  extract_frames(cl_args.video_path, "extracted_frames/")
    frame_segmentation("extracted_frames/", frame_count, "extracted_masks/")
    save_mask_video(frame_count, path_to_frames="extracted_masks/", mp4_path=cl_args.save_path)
    cleanup("extracted_frames")
    cleanup("extracted_masks")
    print(f"mask video has been saved as {cl_args.save_path}")