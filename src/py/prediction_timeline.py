import pandas as pd
import pdb
from nets.classification import ResNetLSTM3, ResNetLSTM_p2
from loaders.hyst_dataset import TestTransforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse 
import matplotlib.cm as cm

import numpy as np 
import os
import cv2
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def sort_by_timesteps(df):

    l_start, l_end = [], []

    for idx, row in df.iterrows():
        vid_path = row['vid_path']
        name, ext = os.path.splitext(vid_path)

        min_f = int(name.split('_')[-2])
        max_f = int(name.split('_')[-1])

        l_start.append(min_f)
        l_end.append(max_f)

    df['start'] = l_start
    df['end'] = l_end


    df = df.sort_values(by=['start'])

    return df

def read_and_select_mp4_frames_cv2(vid_path, frame_idx, frame_length, mount_point):
    cap = cv2.VideoCapture(os.path.join(mount_point, vid_path))

    transform = TestTransforms(num_frames=frame_length)

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)


    for i in range(frame_length):
        success, frame = cap.read()

        frame = np.zeros((256,256,3))
        frames.append(frame)
        
    video = np.stack(frames, axis=0)
    cap.release()
    cv2.destroyAllWindows()

    video = transform(video)

    return video


def plot_timeline(ax, labels, y_pos, height, color_mapping):
    # Plot each segment in the timeline
    for i, label in enumerate(labels):
        color = color_mapping.get(label, '#000000')  # Default to black if label not found
        ax.add_patch(patches.Rectangle((i, y_pos), 1, height, color=color))


def main(args):

    # load model for prediction
    model = ResNetLSTM_p2.load_from_checkpoint(args.model, strict=True)
    model.eval()
    model.cuda()
    model.memory_model.eval()
    model.memory_model.cuda()


    df_test = pd.read_csv(args.csv)
    df_test = sort_by_timesteps(df_test)


    predicted_labels, true_labels = [], []
    batch_size = 1
    idx_reading = 0

    with torch.no_grad():
        for idx, row in df_test.iterrows():
            vid_path = row['vid_path']
            start_frame = row['start']
            end_frame = row['end']-30
            
            y_true = row['tag']
            
            if end_frame - start_frame >= 0:
                print(f"Reading video {idx_reading} / {len(df_test)}")

                for frame_idx in range(start_frame, end_frame):

                    X = read_and_select_mp4_frames_cv2(vid_path, frame_idx, args.frame_length, args.mount_point)
                    X = X.view(batch_size, args.frame_length, 3, 256, 256)
                    X = X.cuda().contiguous()
                    
                    long_features= model.memory_model(X) # (BS* num_frames, 512)
                    long_features = long_features.view(-1, args.frame_length, 512)

                    ## take only 10 frames in x
                    X = X[:,:10,:,:,:]

                    pred = model.forward(X, long_features)

                    pred = torch.argmax(pred, dim=1).cpu().numpy()
                    predicted_labels.append(pred)
                    true_labels.append(y_true)
            else:
                print(f"file {idx_reading} / {len(df_test)} - skipping video, not enough frames")
            idx+=1

    pdb.set_trace()
    classes_names = np.unique(df_test['class_label'])
    classes_tag = np.unique(df_test['tag'])


    cmap = cm['Paired']
    # Create a color mapping for the labels
    label_colors = {0:cmap(0), 1:cmap(2), 2:cmap(4), 3:cmap(6), 
                    4:cmap(8), 5:cmap(10), 6:cmap(11) }

    # Create a dataframe from the labels
    df = pd.DataFrame({'index': range(len(true_labels)),
                       'true_label': true_labels,
                       'predicted_label': predicted_labels})

    # Prepare the figure
    fig, ax = plt.subplots(figsize=(15, 2))  # Adjust the size as needed
    ax.set_xlim(0, len(true_labels))  # Set the x-axis limits to the number of labels
    ax.set_ylim(-0.5, 1)  # Set the y-axis limits to accommodate the true and predicted labels

    # Plot the true/predicted labels
    plot_timeline(ax, df['true_label'], y_pos=0, height=0.4, color_mapping=label_colors)
    plot_timeline(ax, df['predicted_label'], y_pos=0.5, height=0.4, color_mapping=label_colors)


    ax.plot([0, len(true_labels)], [1, 1], color='black')  # Line for the video timeline
    ax.text(len(true_labels) / 2, 1.1, 'Video timeline', ha='center')  # Title for the video timeline
    ax.text(0, 1.1, 'start', ha='left')  # Title for the video timeline
    ax.text(len(true_labels), 1.1, 'end', ha='right')  # Title for the video timeline


    # Customize the y-axis
    ax.set_yticks([0.2, 0.7])
    ax.set_yticklabels(['True', 'Predicted'])

    # Turn off the x-axis visibility as we are not showing real time
    ax.get_xaxis().set_visible(False)

    patch_list = []
    for idx in classes_tag.tolist():
        patch = patches.Patch(color=cmap.colors[idx], label=classes_names[idx])
        patch_list.append(patch)

    ax.legend(handles=patch_list, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Remove spines --> box lines
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(args.outfile, bbox_inches='tight')


if __name__ == "__main__":
  
    # video_path = 'DPM_1.24.23.csv'

    parser = argparse.ArgumentParser(description='Classification predict')
    parser.add_argument('--csv', type=str, help='csv files containing frames number and labels', required=True)
    parser.add_argument('--model', help='Model checkpoint', type=str, required=True)

    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    parser.add_argument('--frame_length', type=int, help='frame length used for training', default=30)

    parser.add_argument('--outfile', help='Output file', type=str, default="./out.png")
    args = parser.parse_args()


    main()