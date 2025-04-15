<!-- PROJECT LOGO -->
<br />
<p align="center">
    <img src="./images/impartial-logo.png" width="50%">
    <h1 align="center"><strong>Interactive instance segmentation using partial annotations</strong></h1>
    <p align="center">
    <a href="https://monai.io/wg_human_ai_interaction.html">MONAI Human-AI Interaction Working Group</a> |
    <a href="https://github.com/nadeemlab/ImPartial/issues">Report Bug</a> |
    <a href="https://github.com/nadeemlab/ImPartial/issues">Request Feature</a>
  </p>
</p>

Interactive segmentation is a crucial area of research in medical image analysis aiming to boost efficiency of costly annotations by incorporating human feedback. Two use cases where this is especially important are: (1) newer biological imaging modalities that do not have large amounts of expert-generated ground truth data and (2) refining and correcting under-/over-segmentation issues from state-of-the-art (SOTA) deep learning algorithms. In this paper, we present a new interactive deep learning segmentation algorithm, **ImPartial**, that incorporates human feedback during the training phase to create optimal models for images with small/thin repeatable objects (cells, neurons, vessels, etc). Specifically, ImPartial augments the segmentation objective via self-supervised multi-channel quantized imputation. This approach leverages the observation that perfect pixel-wise reconstruction or denoising of the image is not needed for accurate segmentation, and thus, introduces a self-supervised classification objective that better aligns with the overall segmentation goal. We demonstrate the effectiveness of our approach in (1) efficiently generating high-quality annotations from scratch for cells/vessels in multiplexed images with variable number of channels and (2) correction/refinement of results from SOTA automated algorithms. To make the results from our pipeline more reproducible and easy to benchmark, we also release new benchmark datasets and an opensource pipeline with a user-friendly ImageJ/FIJI interface and MONAI Label API server that can be run locally, on HPC or cloud.  

*With **MONAI-Label integration**, **cloud (Amazon Web Services) deployment**, and a user-friendly **ImageJ/Fiji plugin/interface**, **ImPartial** can be run iteratively on a new user-uploaded dataset in an active learning human-in-the-loop framework with a no-code execution. A new **multi-user support scheme** is deployed as well to allow users to sign-up/authenticate and simultaneously use our cloud resources with capabilities to end/restore user sessions and develop/share new models with the wider community as needed (hopefully resulting in an ImPartial-driven marketplace in the future where users can share their models with the wider community while being properly credited for their hard work).*

## Pipeline

![unet_arch](./images/impartial_pipeline_v3.png)
*Each image patch is separated into an imputation patch and a blind spot patch. The blind spot patch is fed through the U-Net to recover the component mixture and the component statistics. The 
latter statistics are averaged across the entire patch to enforce component consistency. Both the component statistics and component mixture are used to compute the mixture loss for the patch. Simultaneously, a scribble containing a small number of ground truth segmentations for the patch is used to compute the scribble loss. Both losses propagate gradients back to the U-Net architecture on the backward pass. Additional scribbles can be added to fine-tune the model trained in the previous iteration. Uncertainty maps are computed and shown to guide the user to provide additional scribblles in high uncertainty regions.*


## Iterative Pipeline
![vectra_iterative](./images/vectra_iterative_v3.png)

*Iterative analysis on our public [VECTRA 2-channel images](./data). In the first iteration 20% scribbles were provided on image (0, 1, 3, 4, 5, 6, 7, 9). Based on the entropy visualization, an additional 20% scribbles were provided on images (1, 3, 9) as shown. The model was fine-tuned from iteration 1 using the original and additional scribbles. Note that no scribbles were provided for images (2, 8) during this iterative process. Here, we show the improvement through entropy-aware / guided human-in-the-loop approach. We also show the F1-score improvement across all images. The last three columns show the results from the state-of-the-art [Cellpose](https://www.nature.com/articles/s41592-022-01663-4), [Mesmer](https://www.nature.com/articles/s41587-021-01094-0), and [Cellotype](https://www.nature.com/articles/s41592-024-02513-1) pre-trained models.*

## Results on the CPDMFCI Dataset
![CPDMFCI_results](./images/cpdmi_results_v2.png)

*Quantitative results on the public [multi-channel CPDMFCI spatial proteomics dataset](https://www.nature.com/articles/s41597-023-02108-z) with ground truth masks, under (red) and over (blue) segmentation predictions, and entropy. First row shows lymph node/normal tissue from the CODEX platform with 6 channels: CD8 (Red), CD20 (Magenta), CD21 (Cyan), CD31 (Green), CD45RO (Yellow), DAPI (Blue). Second row shows lymph node/Hodgkin’s lymphoma tissue from the VECTRA platform with 7 channels: CD8 (Red), CD20 (Magenta), CD21 (Cyan), CD31 (Green), CD45RO (Yellow), DAPI (Blue). Third row shows skin/cutaneous T-cell lymphome from the Zeiss platform with 5 channels: PanCK (Red), PD-L1 (Green), CD3 (Cyan), Foxp3 (Magenta), DAPI (Blue).*


## ImPartial, MONAI-Label & Fiji Intergration 

We have transitioned from research to a production-ready environment/workflow, where in the user can upload images, provide scribbles, and run deep learning based model training and inference. We have utilized the following three components to provide an end-to-end service:

1. ImageJ/Fiji - This acts as a client with a user-friendly interface. User can add, delete, or modify annotations to the uploaded image dataset. We opted for ImageJ/Fiji interface due to its easy extensibility and large user-base (100,000+ active users).
 
2. MONAI-Label - For backend, we used [MONAI-Label](https://github.com/Project-MONAI/MONAILabel) which is a Pytorch based open-source framework for deep learning in medical imaging. It provided out-of-the-box inferface to plug in ImPartial deep learning pipeline via Restful API that ties together training, inference and active learning iterative sample-selection strategy. 
Active learning approach: MONAI-Label suports an active learning based approch for users to iteratively train and fine-tune models. We use uncertainty maps to show users the quality of the results every few epochs. 

3. Amazon Web Services (AWS) Cloud Deployment - We deployed ImPartial using the AWS cloud platform with MONAI-Label backend to deploy ImPartial as a service and support multiple users simultaneously.

![pipline_impartial_fig](./images/pipline_impartial_fig.png)

*This workflow diagram illustrates the interactive and iterative nature of the ImPartial pipeline, allowing users to actively contribute to the segmentation model's improvement through annotation and fine-tuning. The combination of user input and deep learning enables more accurate and adaptive whole cell image segmentation. `(1.) Setup:` The workflow begins with the user interacting with the ImPartial plugin through the Fiji app to connect to an ImPartial endpoint or a local server which runs MONAI label as its core backend service. User uploads images to the tool which are stored on our cloud storage system, such as Amazon S3, and a backend MONAI datastore. `(2.) Scribbles:` For each uploaded image, the user utilizes Fiji's draw tool feature to manually mark cell boundaries for a small number of cells. This annotation process allows the user to provide initial guidance to the segmentation algorithm. `(3.) Submit Scribbles:` Once the cell boundaries are marked, the user submits the annotations (scribbles) to the system. `(3.1)` These scribbles are linked and stored alongside original images. `(3.2.)` Training configuration: The user can configure the machine learning training job by tuning hyper-parameters such as the number of epochs, learning rate, and other relevant parameters. `(4.) Initiate Training Job:` With the training parameters set, the user initiates an asynchronous training job which will utilize the annotated data alongside image denoising to train a segmentation model. The progress of the training can be monitored in real-time via the plugin. `(4.1) Model Update:` During training, multiple image segmentation metrics are logged and the newly trained, better performing model is stored. `(4.2) Model Inference:` Since, the ImPartial workflow is asynchronous, model inference can be run any time during and after the training to obtain predictions for cell segmentation on new, unlabeled data.   `(5.) Visualization of Results:` The user can visualize the results of the segmentation model. This includes viewing the provided images, scribbles, model predictions, and entropy (uncertainty) maps simultaneously on a single canvas. This visualization aids in understanding the model's performance and identifying areas of high uncertainty in the segmentation. `(6.) Iterative Refinement:` Finally, users can add additional scribbles or annotations based on the visualization results. With the new annotations, the training is re-initiated triggering fine-tuning of the existing model with the new data.*


## Quickstart - No Code Execution
For a fast, no-code way to get started using the pre-compiled binaries, check out the [**Quickstart Guide**](QUICK_START.md).


## ImPartial Installation: 

### ImPartial library
```
git clone https://github.com/nadeemlab/ImPartial.git
pip install -e .
```

### MONAI-Label

Pre-requisites
* python >=3.10

Install Python dependencies in a virtual environment using **pip**
```
python3 -m venv venv
source venv/bin/activate
pip install -U pip && pip install -r requirements.txt
```
Run MONAI-Label app
```
cd monailabel-app
monailabel start_server -a api -s <data-dir>
```

and navigate to http://localhost:8000 to access the [Swagger UI](https://github.com/swagger-api/swagger-ui)
for interactive API exploration.

#### MONAI-Label in Docker

Build the docker image
```shell
docker build -t monailabel/impartial .
```

run the image built above
```shell
docker run -d --name impartial -p 8000:8000 monailabel/impartial monailabel start_server -a api -s /opt/monai/data
```

and navigate to http://localhost:8000

### ImageJ/Fiji Plugin

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

A detailed guide for the Fiji plugin can be found [here](imagej-plugin/README.md). 

## Issues
Please report all issues on the public forum.


## License
© [Nadeem Lab](https://nadeemlab.org/) - ImPartial code is distributed under **Apache 2.0 with Commons Clause** license, and is available for non-commercial academic purposes. 

## Funding
This work is funded by the 7-year NIH/NCI R37 MERIT Award ([R37CA295658](https://reporter.nih.gov/search/5dgSOlHosEKepkZEAS5_kQ/project-details/11018883#description)).
