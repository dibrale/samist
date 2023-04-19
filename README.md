# SAMIST
Segment Anything Model (SAM) Image Segmentation Tool. A python GUI for image segmentation using SAM by Meta AI. 

## Installation

1. Clone this repository and navigate into its directory
2. Install dependencies, i.e. using `pip install . -r requirements.txt`
3. Download a model checkpoint from Meta AI:
  - `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth), 363M parameters
  - `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth), 308M parameters
  - `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth), 91M parameters
4. Run the script using `python main.py`

## Usage

SAMIST utilizes [PySimpleGui](https://www.pysimplegui.org/en/latest/), and has no command line interface. The GUI looks like this when the program is started:

![samist0](https://user-images.githubusercontent.com/108030031/232980570-dd181966-96cf-4327-8c9f-1980749b60a0.png)

- Use the 'Device' dropdown to select between CUDA and CPU execution. Note that CPU execution is significantly slower.
- Select the 'Model Tyle' using the dropdown. Ensure that the model type you selected corresponds to the checkpoint you are using, or the program will hang.
- Click 'Browse Model' to open up a system dialog and select the model path.
- To the right of the 'Image File' textbox, select 'Browse' to load find an image, then press 'Load Image' to load it for segmentation. The image thumbnail will then appear in the GUI. The GUI looks like this after an image is loaded:

![samist](https://user-images.githubusercontent.com/108030031/232975306-77545c3f-c0f3-4c5d-91eb-5e1e434bbb53.png)

- Use the 'Points per side', 'Stability Score Threshold' and 'Prediction IOU Threshold' to set mask generation parameters. Experiment to find what works best for your application.
- Click 'Segment Loaded Image' to begin image segmentation. This and the 'Load Image' button will be grayed out while segmentation is in progress. The GUI looks like this after image segmentation:

![samist2](https://user-images.githubusercontent.com/108030031/232979333-9e04a45c-0acc-428e-917b-580ffcba2c33.png)

- If one or more masks were generated using your parameter set, the 'Select Masked Image' slider will appear on the GUI. Drag it to select a mask for viewing and export. Position zero will show the original image.
- There are three ways to export a mask using SAMIST, all of which will bring up the system dialog:
  1. 'Save Mask': This will save the full resolution greyscale mask to an image file.
  2. 'Save Masked Image': This will save a copy of the image with the mask applied to it, i.e. a full-resolution version of the masked image as it appears in the GUI.
  3. 'Save Cropped Image': This will save a copy of the image with the mask applied to it and crop it to contents.
  
## Afterword
  
I hope you find this tool both useful and simple to use. Please let me know if you encounter any issues or have any suggestions. The Segment Anything Model was created by Meta AI and is licensed under the Apache 2.0 license.
