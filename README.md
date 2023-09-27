<!-- PROJECT LOGO -->
<br />
<p align="center">
    <img src="./images/impartial-logo.png" width="50%">
    <h1 align="center"><strong>Interactive deep learning whole-cell segmentation and thresholding using partial 
		annotations</strong></h1>
    <p align="center">
    <a href="https://doi.org/10.1101/2021.01.20.427458">Read Link</a> |
    <a href="https://github.com/nadeemlab/ImPartial/issues">Report Bug</a> |
    <a href="https://github.com/nadeemlab/ImPartial/issues">Request Feature</a>
  </p>
</p>

Segmenting noisy multiplex spatial tissue images is a challenging task, since the characteristics of both the noise and 
the biology being imaged differs significantly across tissues and modalities; this is compounded by the high monetary 
and time costs associated with manual annotations. It is therefore important to create algorithms that can accurately 
segment the noisy images based on a small number of annotations. *With **ImPartial**, we have developed an algorithm to 
perform segmentation using as few as 2-3 training images with some user-provided scribbles. ImPartial augments the 
segmentation objective via self-supervised multi-channel quantized imputation, meaning that each class of the 
segmentation objective can be characterized by a mixture of distributions. This is based on the observation that perfect 
pixel-wise reconstruction or denoising of the image is not needed for accurate segmentation, and hence a self-supervised 
classification objective that better aligns with the overall segmentation goal suffices. We demonstrate the superior 
performance of our approach for a variety of datasets acquired with different highly-multiplexed imaging platform.*

## Pipeline

![unet_arch](./images/unet_arch.png)
*(A) Overview of the ImPartial pipeline. (B) Each image patch is separated into an imputation patch and a blind spot 
patch. The blind spot patch is fed through the U-Net to recover the component mixture and the component statistics. The 
latter statistics are averaged across the entire patch to enforce component consistency. Both the component statistics 
and component mixture are used to compute the mixture loss for the patch. Simultaneously, a scribble containing a small 
number of ground truth segmentations for the patch is used to compute the scribble loss. Both losses propagate gradients 
back to the U-Net architecture on the backward pass.*

## Impartial, Monai Label & Fiji Intergration 

We  transitioned from a research workflow into a production ready environment,  where in the user can upload images, provide scribbles and also run deep learning based model training and inference. We utilized following three components to provide an end to end service:

1. ImageJ/Fiji - Which acts as a client. Here user can add, delete or modify annotations to the uploaded image dataset. 
We chose ImageJ because it is one of the most extensively used labeling and vewing tool used by expert annotators. 
 
2. Monai label - For backend, we used Monai label which is a Pytorch based open source framework for deep learning healthcare imaging. It provides an out of the box inferfaces to plugin ImPartial DL pipeline via Restful API that ties together training, inference and selection strategy. 
Active learning approach: Monai label suports an active learning based approch for user to actively train and give feedback to fine tune the model. 

3. AWS - We deployed impartial using the AWS cloud pltform with Monai label to support Multi-user and  deploy impartial as a service.

![pipline_impartial_fig](./images/pipline_impartial_fig.png)

*This workflow diagram illustrates the interactive and iterative nature of the impartial pipeline, allowing users to actively contribute to the segmentation model's improvement through annotation and fine-tuning. The combination of user input and deep learning enables more accurate and adaptive whole cell image segmentation. `(1.) Setup:` The workflow begins with the user interacting with the Impartial plugin through the Fiji app to connect to an Impartial endpoint or a local server which runs MONAI label as its core backend service.  User uploads images to the tool which are stored into cloud storage system, such as Amazon S3, and a backend MONAI datastore.   `(2.) Scribbles:` For each uploaded image, the user utilizes Fiji's draw tool feature to manually mark cell boundaries for a small number of whole cells. This annotation process allows the user to provide initial guidance to the segmentation algorithm.  `(3.) Submit Scribbles:` Once the cell boundaries are marked, the user submits the annotations (scribbles) to the system. `(3.1)` These scribbles are linked and stored alongside original images.  `(3.2.)` Training configuration:  The user can configure the machine learning training job by tuning hyper parameters such as the number of epochs, learning rate, and other relevant parameters.  `(4.) Initiate Training Job:` With the training parameters set, the user initiates an asynchronous training job which will utilize the annotated data alongside image denoising to train a segmentation model. The progress of the training can be monitored in real-time via the plugin.   `(4.1) Model Update:` During training multiple image segmentation metrics are logged and the newly trained, better performing model is stored.   `(4.2) Model Inference:` Since, the impartial workflow is asynchronous, model inference can be run any time during and after the training  to obtain predictions for cell segmentation on new, unlabeled data.   `(5.) Visualization of Results:`  The user can visualize the results of the segmentation model. This includes viewing the provided images, scribbles, model predictions, and entropy maps simultaneously on a single canvas. This visualization aids in understanding the model's performance and identifying areas of high uncertainty in the segmentation. `(6.) Iterative Refinement:` Finally, users can further add additional scribbles or annotations based on the visualization results. With the new annotations, the training is  re-initiated triggering fine-tuning of the existing model with the new data.*

## Impartial Installation: 

### MONAI Label

Pre-requisites
* Python 3

Install Python dependencies in a virtual environment using **pip**
```
python3 -m venv venv
source venv/bin/activate
pip install -U pip && pip install -r requirements.txt
```
Run MONAI Label app
```
cd impartial
monailabel start_server -a api -s <data-dir>
```

and navigate to http://localhost:8000 to access the [Swagger UI](https://github.com/swagger-api/swagger-ui)
for interactive API exploration.

## MONAI Label in Docker

Build the docker image
```shell
docker build -t monailabel/impartial .
```

run the image built above
```shell
docker run -d --name impartial -p 8000:8000 monailabel/impartial monailabel start_server -a api -s /opt/monai/data
```

and navigate to http://localhost:8000

## ImageJ/Fiji Plugin

Pre-requisites
* [Fiji](https://imagej.net/software/fiji/downloads) (version >= 2.3.0)
* [Apache Maven](https://maven.apache.org/install.html) 
  (or use [brew](https://formulae.brew.sh/formula/maven) on macOS)

First, package the plugin. From the repo root directory
```shell
cd imagej-plugin
mvn clean package
```
and copy the `.jar` file into Fiji's plugins directory. For example, if you're using macOS
```shell
cp target/impartial_imagej-0.1.jar /Applications/Fiji.app/plugins
```

then restart **Fiji** and open `ImPartial` from the `Plugins` menu bar.

## No-Code Cloud Deployment
For ready to use plugin, User can get a pre-compiled .jar file [here](imagej-plugin/impartial_imagej-0.1.jar). With this option, you can skip setting up Maven and compiling the package. Copy the .jar file directly to the Fiji plugin folder mentioned above.

User can request our cloud deployed Monai server to readily annotate and segment the data without needing to compile and run any code locally. 

<img src="./images/fiji_impartial_plugin.png" height=500px> <img align="top" src="./images/fiji_app.png" width = 400px />

A detailed guide for the Fiji plugin can be found [here](imagej-plugin/README.md). 

## Issues
Please report all issues on the public forum.


## License
© [Nadeem Lab](https://nadeemlab.org/) - ImPartial code is distributed under **Apache 2.0 with Commons Clause** license, and is available for non-commercial academic purposes. 


## Reference
If you find our work useful in your research or if you use parts of this code, please cite our paper:
```
@article {Martinez2021.01.20.427458,
	author = {Martinez, Natalia and Sapiro, Guillermo and Tannenbaum, Allen and Hollmann, Travis J. and Nadeem, Saad},
	title = {ImPartial: Partial Annotations for Cell Instance Segmentation},
	elocation-id = {2021.01.20.427458},
	year = {2021},
	doi = {10.1101/2021.01.20.427458},
	publisher = {Cold Spring Harbor Laboratory}
}
```
