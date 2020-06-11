> 📋A template README.md for code accompanying a Machine Learning paper

A Target Guided Generative Clustering Model for Spatial Object Detection from Weak Annotations

This repository is the official implementation of A Target Guided Generative Clustering Model for Spatial Object Detection from Weak Annotations. 


## Requirements

To install requirements:

```setup
sudo nvidia-docker run -t -i -v /folder/:/folder/ -p 8888:8888 spatialcomputing/map_text_recognition_gpu /bin/bash
```

> 📋Run above command to set up the environment
## Training

To train the model(s) in the paper, run this command:

```data generation
python data_gen.py --training_data_folder <path_to_data> --image <path_to_image> --mask <path_to_annotation>
```

```train
python train.py --target_samples_folder <path_to_data> --image <path_to_image> --mask <path_to_annotation> --weight 1000
```

> 📋Use data_gen.py to get 30% data as training data, and the user choose a target sample and put it to the target sample folder.
> 📋Use train.py to train TOD.

## Evaluation

To detect target objects, run:

```eval
python test.py --model-file mymodel.pth --image <path_to_image>
```

> 📋Use the trained model to detect objects on the entire annotated area.


## Results

Our model achieves the following performance on cars detection in parking lots:

|       | Parking lot 1      | Parking lot 2      | Parking lot 3      |
| ------|------------------- |------------------- | -------------------|
|       |precision | recall  |precision | recall  |precision | recall  |
| ------|------------------- |------------------- | -------------------|
| TOD   |95.00%.   | 88.68%  |78.57%.   |92.86%.  |93.20%.    |93.70%. |  



## Contributing
Copyright (c) year copyright holders

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
