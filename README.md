 # yolov8 training dogs breed

### data images from kaggle

### python version 3.10.9

### pip install -r requirements.txt

### python main.py
1. create images/labels
2. create train files 70% of data about images & labels
3. create valid files 30% of data about images & labels
4. create train.txt
5. create valid.txt
6. create classess.txt
7. create data.yaml

### after data.yaml
change main.py yaml_path: absolute path of data.yaml
open #xml2yolo_converter.yolo_training()
training with cpu on mac os
you can reset all information (epochs=50, imgsz=640, device='cpu', workers=4)

### python detect.py
this best.pt is already a train model can use
find the runs/train model best.pt you can copy current directory

### before detect
![img](https://github.com/leolee1204/yolov8/blob/3684fbdfb14da8f030fff065f22e7356cd4f7010/dog3.JPG)

### after detect
![img](https://github.com/leolee1204/yolov8/blob/3684fbdfb14da8f030fff065f22e7356cd4f7010/dog3_finish.JPG)
