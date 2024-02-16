import os
import xml.etree.ElementTree as ET
import shutil
import yaml
from ultralytics import YOLO

class YoloTrainingData:
    def __init__(self, annotations_path='annotations/Annotation', images_path='images/images', labels_path='images/labels', yaml_path='input your absolute path of data.yaml'):
            self.annotations_path = annotations_path
            self.images_path = images_path
            self.labels_path = labels_path
            self.yaml_path = yaml_path
            self.class_mapping = self.get_labels()

    def get_labels(self):
        result = []
        for x in os.listdir('images/images'):
            result.append(x[10:])
        result = {name: i for i, name in enumerate(result)}
        return result

    def convert_coordinates(self, width, height, x, y, w, h):
        x_center = (x + w / 2.0) / width
        y_center = (y + h / 2.0) / height
        w_normalized = w / width
        h_normalized = h / height
        return x_center, y_center, w_normalized, h_normalized

    def xml_to_yolo(self, xml_content, yolo_file):
        root = ET.fromstring(xml_content)
        image_width = float(root.find('size/width').text)
        image_height = float(root.find('size/height').text)

        for obj in root.findall('object'):
            class_label = obj.find('name').text
            class_index = self.class_mapping.get(class_label, -1)

            if class_index != -1:
                yolo_file_dir = "/".join(yolo_file.split('/')[:-1])
                if not os.path.exists(yolo_file_dir):
                    os.makedirs(yolo_file_dir)

                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)

                x_center, y_center, w_normalized, h_normalized = self.convert_coordinates(
                    image_width, image_height, xmin, ymin, xmax - xmin, ymax - ymin
                )

                with open(f"{yolo_file}.txt", 'w') as f:
                    f.write(f"{class_index} {x_center:.6f} {y_center:.6f} {w_normalized:.6f} {h_normalized:.6f}\n")

                if x_center >= 1 or y_center >= 1 or w_normalized >= 1 or h_normalized >= 1:
                    os.remove(f"{yolo_file}.txt")
                    file = "/".join(yolo_file.split('/')[2:])
                    os.remove(f"images/images/{file}.jpg")

    def xml2yolo(self):
        if not os.path.exists(self.labels_path):
            for current_dir, subdirectories, files in os.walk(self.annotations_path):
                for file in files:
                    xml_path = os.path.join(current_dir, file)
                    dir = f"{self.labels_path}/{current_dir.split('/')[-1]}"
                    file_path = os.path.join(dir, file)

                    with open(xml_path, 'r') as file:
                        try:
                            xml_content = file.read()
                            self.xml_to_yolo(xml_content, file_path)
                        except Exception as e:
                            print(f"Error processing {xml_path}: {e}")
    
    def split_train_valid(self, train_ratio=0.7, random_state=42):
        train_dir = 'train'
        val_dir = 'valid'
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(os.path.join(train_dir,'images'), exist_ok=True)
        os.makedirs(os.path.join(train_dir,'labels'), exist_ok=True)
        os.makedirs(os.path.join(val_dir,'images'), exist_ok=True)
        os.makedirs(os.path.join(val_dir,'labels'), exist_ok=True)

        if not os.listdir("train/images"):
            # Get the list of files in the directory
            for dir in os.listdir(self.images_path):
                try:
                    dir_img_path = os.path.join(self.images_path, dir)
                    dir_label_path = os.path.join(self.labels_path, dir)

                    images_length = len(os.listdir(dir_img_path))
                    labels_length = len(os.listdir(dir_label_path))
                    sevenry_index = int(images_length * train_ratio)

                    images = "_".join(dir_img_path.split('/')[-1].split('-')[1:])
                    labels = "_".join(dir_label_path.split('/')[-1].split('-')[1:])

                    train_img_path = os.path.join('train/images',images)
                    train_label_path = os.path.join('train/labels',labels)
                    val_img_path = os.path.join('valid/images',images)
                    val_label_path = os.path.join('valid/labels',labels)
                    
                    os.makedirs(train_img_path, exist_ok=True)
                    os.makedirs(train_label_path, exist_ok=True)
                    os.makedirs(val_img_path, exist_ok=True)
                    os.makedirs(val_label_path, exist_ok=True)
                    
                    train_images = sorted(os.listdir(dir_img_path))[:sevenry_index+1]
                    train_labels = sorted(os.listdir(dir_label_path))[:sevenry_index+1]
                    for img,label in zip(train_images,train_labels):
                        source_path_img = os.path.join(dir_img_path, img)
                        destination_path_img = os.path.join(train_img_path,"".join(img.split('_')[-1:]))
                        shutil.copyfile(source_path_img, destination_path_img)

                        source_path_label = os.path.join(dir_label_path, label)
                        destination_path_label = os.path.join(train_label_path,"".join(label.split('_')[-1:]))
                        shutil.copyfile(source_path_label, destination_path_label)

                    val_images = sorted(os.listdir(dir_img_path))[sevenry_index+1:]
                    val_labels = sorted(os.listdir(dir_label_path))[sevenry_index+1:]
                    for img,label in zip(val_images,val_labels):
                        source_path_img = os.path.join(dir_img_path, img)
                        destination_path_img = os.path.join(val_img_path,"".join(img.split('_')[-1:]))
                        shutil.copyfile(source_path_img, destination_path_img)

                        source_path_label = os.path.join(dir_label_path, label)
                        destination_path_label = os.path.join(val_label_path,"".join(label.split('_')[-1:]))
                        shutil.copyfile(source_path_label, destination_path_label)

                except Exception as e:
                    pass
    
    def create_yolo_train_files(self):
        if not os.path.exists('train.txt'):
            with open('train.txt','w') as f:
                for root, dirs, files  in os.walk('train/images'):
                    for file in files:
                        img_path = os.path.join(root,file)
                        f.write(f"./{img_path}\n")

        if not os.path.exists('valid.txt'):
            with open('valid.txt','w') as f:
                for root, dirs, files  in os.walk('valid/images'):
                    for file in files:
                        img_path = os.path.join(root,file)
                        f.write(f"./{img_path}\n")

        if not os.path.exists('classes.txt'):
            with open('classes.txt','w')as f:
                for class_ in os.listdir('train/images'):
                    f.write(f"{class_}\n")

        if not os.path.exists('data.yaml'):
            with open('data.yaml', 'w') as yaml_file:
                result = dict()
                for dir in os.listdir('train/labels'):
                    path_ = os.path.join('train/labels',dir)
                    try:
                        with open(os.path.join(path_, os.listdir(path_)[0]),'r')as f:
                            data = f.readline()
                            result[int(data.split(' ')[0])] = dir
                    except :
                        pass
                data = {
                    "train": "train.txt",
                    "val": "valid.txt",
                    "nc": len(result),
                    "names": result
                }
                yaml.dump(data, yaml_file, default_flow_style=False)

          
    def yolo_training(self):
        # Load a model
        model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

        # Train the model with 2 GPUs
        # model.train(data=self.yaml_path, epochs=100, imgsz=240, device='mps')
        model.train(data=self.yaml_path, epochs=50, imgsz=640, device='cpu', workers=4)
        model.val() 

        model.export(format="onnx")

def main():
    xml2yolo_converter = YoloTrainingData()
    xml2yolo_converter.xml2yolo()
    xml2yolo_converter.split_train_valid()
    xml2yolo_converter.create_yolo_train_files()
    #xml2yolo_converter.yolo_training()

if __name__ == "__main__":
    main()