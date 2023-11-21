from mmdet.apis import inference_detector
import argparse
import mmcv
import cv2
import time
import os
import requests
import yaml
import glob as glob
from mmdet.apis import init_detector
def download_weights(url, file_save_name):
    """
    Download weights for any model.
    :param url: Download URL for the weihgt file.
    :param file_save_name: String name to save the file on to disk.
    """
    # Make chekcpoint directory if not presnet.
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    # Download the file if not present.
    if not os.path.exists(os.path.join('checkpoint', file_save_name)):
        file = requests.get(url)
        open(
            os.path.join('checkpoint', file_save_name), 'wb'
        ).write(file.content)
        
def parse_meta_file():
    """
    Function to parse all the model meta files inside `mmdetection/configs`
    and return the download URLs for all available models.
    Returns:
        weights_list: List containing URLs for all the downloadable models.
    """
    root_meta_file_path = 'mmdetection/configs'
    all_metal_file_paths = glob.glob(os.path.join(root_meta_file_path, '*', 'metafile.yml'), recursive=True)
    weights_list = []
    for meta_file_path in all_metal_file_paths:
        with open(meta_file_path,'r') as f:
            yaml_file = yaml.safe_load(f)
            
        for i in range(len(yaml_file['Models'])):
            try:
                weights_list.append(yaml_file['Models'][i]['Weights'])
            except:
                for k, v in yaml_file['Models'][i]['Results'][0]['Metrics'].items():
                    if k == 'Weights':
                        weights_list.append(yaml_file['Models'][i]['Results'][0]['Metrics']['Weights'])
    return weights_list

def get_model(weights_name):
    """
    Either downloads a model or loads one from local path if already 
    downloaded using the weight file name (`weights_name`) provided.
    :param weights_name: Name of the weight file. Most like in the format
        retinanet_ghm_r50_fpn_1x_coco. SEE `weights.txt` to know weight file
        name formats and downloadable URL formats.
    Returns:
        model: The loaded detection model.
    """
    # Get the list containing all the weight file download URLs.
    weights_list = parse_meta_file()
    download_url = None
    for weights in weights_list:
        if weights_name in weights:
            print(f"Founds weights: {weights}\n")
            download_url = weights
            break
    assert download_url != None, f"{weights_name} weight file not found!!!"
    # Download the checkpoint file.
    download_weights(
        url=download_url,
        file_save_name=download_url.split('/')[-1]
    )
    checkpoint_file = os.path.join('checkpoint', download_url.split('/')[-1])
    # Build the model using the configuration file.
    config_file = os.path.join(
        'mmdetection/configs', 
        download_url.split('/')[-3],
        download_url.split('/')[-2]+'.py'
    )
    model = init_detector(config_file, checkpoint_file)
    return model

def write_weights_txt_file():
    """
    Write all the model URLs to `weights.txt` to have a complete list and 
    choose one of them.
    EXECUTE `utils.py` if `weights.txt` not already present.
    `python utils.py` command will generate the latest `weights.txt` 
    file according to the cloned mmdetection repository.
    """
    # Get the list containing all the weight file download URLs.
    weights_list = parse_meta_file()
    with open('weights.txt', 'w') as f:
        for weights in weights_list:
            f.writelines(f"{weights}\n")
    f.close()


# Contruct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', default='input/video_1.mp4',
    help='path to the input file'
)
parser.add_argument(
    '-w', '--weights', default='yolov3_mobilenetv2_320_300e_coco',
    help='weight file name'
)
parser.add_argument(
    '-t', '--threshold', default=0.5, type=float,
    help='detection threshold for bounding box visualization'
)
args = vars(parser.parse_args())

# Build the model.
model = get_model(args['weights'])
cap = mmcv.VideoReader(args['input'])
save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{args['weights']}"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    f"outputs/{save_name}.mp4", fourcc, cap.fps,
    (cap.width, cap.height)
)
frame_count = 0 # To count total frames.
total_fps = 0 # To get the final frames per second.

for frame in mmcv.track_iter_progress(cap):
    # Increment frame count.
    frame_count += 1
    start_time = time.time()# Forward pass start time.
    result = inference_detector(model, frame)
    end_time = time.time() # Forward pass end time.
    # Get the fps.
    fps = 1 / (end_time - start_time)
    # Add fps to total fps.
    total_fps += fps
    show_result = model.show_result(frame, result, score_thr=args['threshold'])
    # Write the FPS on the current frame.
    cv2.putText(
        show_result, f"{fps:.3f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 0, 255), 2, cv2.LINE_AA
    )
    mmcv.imshow(show_result, 'Result', wait_time=1)
    out.write(show_result)
# Release VideoCapture()
out.release()
# Close all frames and video windows
cv2.destroyAllWindows()
# Calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")