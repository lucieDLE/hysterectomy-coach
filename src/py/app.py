import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import nrrd
import numpy as np
import os
import SimpleITK as sitk
# st.set_page_config(layout="wide")
import cv2
import matplotlib.colors as mcolors


# start from 23470
@st.cache_data
def load_csv(path):
    df = pd.read_csv(path)
    df["frame_id"] = df["Video Name"].astype(str) + "_" + df["Frame"].astype(str)
    return df

# df = df.iloc[]

mnt = '/CMF/data/lumargot/hysterectomy/mnt/surgery_tracking/'
csv_path = os.path.join(mnt,"csv/dataset_train_train.csv" )
df = load_csv(csv_path)

# Unique frames
frame_ids = df["frame_id"].unique()
# frame_ids = frame_ids.iloc[13700: ].reset_index()
frame_index = st.number_input("Frame Index", min_value=0, max_value=len(frame_ids)-1, value=0, step=1)

# Get all rows for the current frame
frame_id = frame_ids[frame_index]
frame_df = df[df["frame_id"] == frame_id]

# Use first row to load shared info
row = frame_df.iloc[0]
img_path = os.path.join(mnt,row['img_path'])

try:
    # img, _ = nrrd.read(img_path)
    img = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(img_path)).copy())
    h1, w1 = img.shape[:2]

except Exception as e:
    st.error(f"Error reading image file `{img_path}`: {e}")
    st.stop()

# Setup figure
fig, ax = plt.subplots()
img = img[::3, ::3]
ax.imshow(img, cmap='gray')
H, W = img.shape[:2]

# Define distinct colors
colors = list(mcolors.TABLEAU_COLORS.values())
# np.random.seed(42)  # for consistent coloring

# Create composite RGBA overlay for masks
overlay = np.zeros((H, W, 4), dtype=np.float32)  # RGBA

# Display all masks and bounding boxes
for i, r in frame_df.iterrows():
    try:
      seg_path = os.path.join(mnt,r["seg_path"])
      class_idx = r['class']
      seg = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(seg_path)).copy())
      seg = seg[::3, ::3]
      
      color = mcolors.to_rgba(colors[i % len(colors)], alpha=0.8)
      color = np.array(color).reshape(1, 1, 4)
      mask_bool = seg > 0
      overlay[mask_bool] = color

    except Exception as e:
        st.warning(f"Could not read mask {r['seg_path']}")

ax.imshow(overlay)  # RGBA mask overlay

for _, r in frame_df.iterrows():
    x, y, w, h = float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])
    rect = patches.Rectangle((x*W, y*H), w*W, h*H,
                             linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(x*W, y*H - 5, f"{r['Instrument Name']} (class {r['class']})", color='red', fontsize=8)

ax.set_title(f"Frame: {frame_id}")
st.pyplot(fig)

# Tagging interface
if "bad_frames" not in st.session_state:
    st.session_state.bad_frames = set()

flagged = st.checkbox("Mark this frame as bad")
if flagged:
    st.session_state.bad_frames.add(frame_id)
else:
    st.session_state.bad_frames.discard(frame_id)

st.write(f"Total bad frames flagged: {len(st.session_state.bad_frames)}")

# Export flagged frames to CSV
if st.button("Export flagged frames"):
    bad_df = df[df["frame_id"].isin(st.session_state.bad_frames)]
    big_df = pd.read_csv('flagged_bad_frames_cat.csv')
    big_df = pd.concat([big_df, bad_df])
    big_df.to_csv(f"flagged_bad_frames_cat.csv", index=False)
    st.success("Exported to `flagged_bad_frames.csv`")