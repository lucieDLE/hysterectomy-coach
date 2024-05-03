# hysterectomy-coach

### 1. training end to end 

```bash
python classification_train.py --csv_train ds_train_train.csv --csv_valid ds_train_test.csv --csv_test ds_test.csv --class_column tag --epochs 250 --out /output/ --mount_point /data/ --num_frames 30 --batch_size 10
```

### 3. prediction
```bash
python classification_predict.py --csv ds_test.csv --class_column tag --model end_end.ckpt --out output/ --mount_point /data/ --num_frames 30 
```

### 4. evaluation
```bash

python eval_classification.py --csv test_prediction.csv --csv_tag_column class
```

### 5. Grad-Cam prediction 

``` bash
python grad_cam_fullvid.py --csv test_prediction.csv --model end_end.ckpt  --num_frames 30  --mount_point /data/ --class_column tag --out  /out/ --batch_size 1 --fps 30
```