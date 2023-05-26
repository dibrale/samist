# SAMIST - Segment Anything Model Image Segmentation Tool
# A python GUI for image segmentation using SAM by Meta AI

# By Alexander Dibrov
# Visit me at https://github.com/dibrale/

import tkinter
from tkinter import filedialog
import os
import PySimpleGUI as sg
import cv2
import numpy as np
from PIL import Image, ImageTk
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

root = tkinter.Tk()
root.withdraw()     # use to hide tkinter window

# Variable declarations
masks = []
masked_images = []
image = []
reference_image = []
out_full_res = []
out_bw = []
out_cropped = []

# Default values
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

points_per_side = 32
min_area = 100
stability_thresh = 0.95
predict_iou = 0.85


# General function for saving masks
def save_mask_pic(mask_array):
    if values["-MASK-"] == 0:
        sg.Popup('Please select a mask')
    else:
        initial_name = f'mask{int(values["-MASK-"])}.png'
        save_filename = sg.tk.filedialog.asksaveasfilename(
            defaultextension='png',
            filetypes=(("png", "*.png"), ("All Files", "*.*")),
            initialdir=os.getcwd(),
            initialfile=initial_name,
            title="Save As"
        )
        if len(save_filename) > 0:
            mask_array[int(values["-MASK-"] - 1)].save(save_filename)


# Segmentation function taking slider parameters and returning masks in various formats
def segmentation(pps, area, st, iou):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=pps,
        pred_iou_thresh=iou,
        stability_score_thresh=st,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=area,  # Requires open-cv to run post-processing
    )

    try:
        masks_raw = mask_generator.generate(image)
    except RuntimeError:
        return [[], []]
    sorted_masks = sorted(masks_raw, key=(lambda x: x['area']), reverse=True)
    return [sorted_masks, prepare_masked(sorted_masks)]


# Non-destructively crop and apply an alpha channel to an image
def crop_alpha(image_in, alpha, bbox):
    out = image_in.copy()
    out.putalpha(alpha.copy())
    out = out.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
    return out


# Take masks, returning area-sorted greyscale masks, cropped full-res images, cropped thumbnails
# and cropped full-res images
def prepare_masked(masks_in):
    out = []
    out_full_res_pil = []
    out_bw_pil = []
    out_cropped_pil = []

    for mask in masks_in:
        result = np.asarray(image).copy()

        bw_mask = np.asarray(image).copy()
        bw_mask[mask["segmentation"] == 1] = (255, 255, 255)
        bw_mask[mask["segmentation"] != 1] = (0, 0, 0)

        bw_mask_pil = Image.fromarray(bw_mask).convert('L')
        result_pil = Image.fromarray(result)
        cropped_pil = crop_alpha(result_pil, bw_mask_pil, mask["bbox"])
        result_pil.putalpha(bw_mask_pil.copy())

        out_full_res_pil.append(result_pil.copy())
        out_bw_pil.append(bw_mask_pil)
        out_cropped_pil.append(cropped_pil)

        result_pil.thumbnail((400, 400))
        photo_img = ImageTk.PhotoImage(result_pil)
        out.append(photo_img)

    return out_bw_pil, out_full_res_pil, out, out_cropped_pil


# Main function
if __name__ == '__main__':
    print('Starting Segment Anything Model Image Segmentation Tool (SAMIST)')

    layout = [
        [
            sg.Text('Device'),
            sg.Drop(values=('cpu', 'cuda'),
                    default_value='cuda',
                    auto_size_text=True,
                    key='-DEVICE-',
                    readonly=True,
                    enable_events=True,
                    tooltip="Choose whether to run image segmentation on CPU, or a CUDA-capable device. "
                            + "Segmentation is slow on CPU."),
            sg.Text('Model Type'),
            sg.Drop(values=('vit_h', 'vit_l', 'vit_b'),
                    default_value='vit_h',
                    auto_size_text=True,
                    key='-MODEL_TYPE-',
                    readonly=True,
                    enable_events=True,
                    tooltip="Three model versions of the model are available with different backbone sizes: "
                            + "The 636M ViT-H, 308M ViT-L and 91M ViT-B ")
        ],
        [
            sg.Text('Model Path'),
            sg.Input(size=(25, 1), default_text=sam_checkpoint, key='-MODEL_TEXT-', enable_events=True, readonly=True),
            sg.FileBrowse(button_text="Browse Model", key='-MODEL_BROWSE-',
                          file_types=[("PTH Files", '*.pth'), ("All Files", '.')], initial_folder=os.getcwd(),
                          tooltip="Download vial links here: https://github.com/facebookresearch/segment-anything"),
        ],
        [sg.HSeparator()],
        [sg.Image(key="-IMAGE-")],
        [sg.Text(key="-IMAGETEXT-", text='Ready for segmentation', visible=False)],
        [
            sg.Text("Image File"),
            sg.Input(size=(25, 1), key="-FILE-"),
            sg.FileBrowse(),
            sg.Button("Load Image")
        ],
        [
            sg.Text(text='Minimum Area', visible=False),
            sg.Push(),
            sg.Slider((500, 10000), 1000, 500, expand_x=True, orientation='horizontal', enable_events=True, key='-MA-',
                      visible=False)
        ],
        [
            sg.Text('Points per side'),
            sg.Push(),
            sg.Slider((4, 64), 32, 1, expand_x=True, orientation='horizontal', enable_events=True, key='-PPS-')
        ],
        [
            sg.Text('Stability Score Threshold'),
            sg.Push(),
            sg.Slider((0.5, 1), 0.95, 0.01, expand_x=True, orientation='horizontal', enable_events=True, key='-ST-')
        ],
        [
            sg.Text('Prediction IOU Threshold'),
            sg.Push(),
            sg.Slider((0.5, 1.5), 0.95, 0.01, expand_x=True, orientation='horizontal', enable_events=True, key='-IOU-')
        ],
        [sg.HSeparator()],
        [
            sg.Text(key='-SMI-', text='Select Masked Image', visible=False),
            sg.Push(),
            sg.Slider((1, 1), 1, 1,
                      expand_x=True,
                      orientation='horizontal',
                      enable_events=True,
                      key='-MASK-',
                      visible=False)
        ],
        [
            sg.Button(button_text="Segment Loaded Image", visible=False, key='-SEG-')
        ],
        [
            sg.Button(button_text="Save Mask", visible=False, key='-SAVE_BW-',
                      tooltip="Save this mask as a greyscale image."),
            sg.Button(button_text="Save Masked Image", visible=False, key='-SAVE_MASKED-',
                      tooltip="Save the image with the selected mask applied."),
            sg.Button(button_text="Save Cropped Image", visible=False, key='-SAVE_CROPPED-',
                      tooltip="Save a cropped version of the masked image.")
        ],
    ]

    # Declare the GUI window
    window = sg.Window("Image Segmentation Tool", layout)

    # Event loop
    while True:
        event, values = window.read()

        # Exit
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        # Select CPU or CUDA
        if event == "-DEVICE-":
            device = values['-DEVICE-']

        # Select model type
        if event == "-MODEL_TYPE-":
            model_type = values['-MODEL_TYPE-']

        # Browse for a model
        if event == "-MODEL_TEXT-":
            sam_path = values["-MODEL_TEXT-"]
            if os.path.exists(sam_path):
                sam_checkpoint = sam_path
                window['-MODEL_TEXT-'].update(value=sam_path)
            else:
                window['-MODEL_TEXT-'].update('File not found')

        # Load an image
        if event == "Load Image":
            filename = values["-FILE-"]
            if os.path.exists(filename):
                im_open = Image.open(filename)
                im_open.thumbnail((400, 400))
                reference_image = ImageTk.PhotoImage(im_open)
                window["-IMAGE-"].update(data=reference_image)
                image = cv2.imread(filename)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                window["-IMAGETEXT-"].update(visible=True, value="Ready for segmentation")
                window["-MASK-"].update(visible=False)
                window["-SEG-"].update(visible=True)

        # Points-per-side slider
        if event == "-PPS-":
            points_per_side = int(values['-PPS-'])

        # Minimum area slider - hidden by default because not super useful
        if event == "-MA-":
            min_area = int(values['-MA-'])

        # Stability threshold slider
        if event == "-ST-":
            stability_thresh = float(values['-ST-'])

        # Stability IOU slider
        if event == "-IOU-":
            predict_iou = float(values['-IOU-'])

        # Segmentation button
        if event == "-SEG-":
            window["-IMAGETEXT-"].update(visible=True, value='Image segmentation started')
            window.perform_long_operation(
                lambda: segmentation(points_per_side, min_area, stability_thresh, predict_iou),
                '-SEG_DONE-')
            window["-SEG-"].update(disabled=True)
            window["Load Image"].update(disabled=True)

        # Segmentation complete
        if event == "-SEG_DONE-":
            [masks, [out_bw, out_full_res, masked_images, out_cropped]] = values[event]
            window["-SEG-"].update(disabled=False)
            window["Load Image"].update(disabled=False)

            if len(masks) > 0:
                image_text = 'Image segmentation complete'
                window["-SMI-"].update(visible=True)
                window["-SAVE_BW-"].update(visible=True)
                window["-SAVE_MASKED-"].update(visible=True)
                window["-SAVE_CROPPED-"].update(visible=True)
                window["-MASK-"].update(visible=True, range=(0, len(masks)), value=0)

            else:
                window["-SMI-"].update(visible=False)
                window["-MASK-"].update(visible=False, range=(0, len(masks)), value=0)
                window["-SAVE_BW-"].update(visible=False)
                window["-SAVE_MASKED-"].update(visible=False)
                window["-SAVE_CROPPED-"].update(visible=False)
                image_text = 'No masks were generated with this set of parameters'

            window["-IMAGETEXT-"].update(visible=True, value=image_text)

        # Mask slider
        if event == "-MASK-":
            if values["-MASK-"] == 0:
                image_text = "Loaded image"
                try:
                    window["-IMAGE-"].update(data=reference_image)
                except TypeError:
                    pass
            else:
                mask_index = int(values["-MASK-"]-1)
                image_text = f'Mask: {int(values["-MASK-"])}, Stability: ' \
                             + str(np.round(masks[mask_index]['stability_score'], 3)) \
                             + ', IOU: ' \
                             + str(np.round(masks[mask_index]['predicted_iou'], 3))
                window["-IMAGE-"].update(data=masked_images[mask_index])
            window["-IMAGETEXT-"].update(visible=True, value=image_text)

        # Save greyscale mask
        if event == "-SAVE_BW-":
            save_mask_pic(out_bw)

        # Save masked image with input image dimensions
        if event == "-SAVE_MASKED-":
            save_mask_pic(out_full_res)

        # Save masked image cropped to contents
        if event == "-SAVE_CROPPED-":
            save_mask_pic(out_cropped)

    window.close()
