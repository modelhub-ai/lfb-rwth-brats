# lfb-rwth-brats
This repository hosts the contributor source files for the lfb-rwth-brats model. ModelHub integrates these files into an engine and controlled runtime environment. A unified API allows for out-of-the-box reproducible implementations of published models. For more information, please visit [www.modelhub.ai](http://modelhub.ai/) or contact us [info@modelhub.ai](mailto:info@modelhub.ai).

#### Please note that this model has a very long runtime for inference, about 70 minutes per case

## meta
| | |
|-|-|
| id | 0314b397-3d19-446b-a1ec-2832e6444f67 |
| application_area | Medical Imaging, Segmentation |
| task | Brain Tumor Segmentation |
| task_extended | Brain tumor segmentation for the BraTS 18 challenge |
| data_type | Nifti-1 volumes |
| data_source | www.braintumorsegmentation.org |
## publication
| | |
|-|-|
| title | Segmentation of Brain Tumors and Patient Survival Prediction: Methods for the BraTS 2018 Challenge |
| source | International MICCAI Brainlesion Workshop |
| url | https://link.springer.com/chapter/10.1007/978-3-030-11726-9_1 |
| year | 2018 |
| authors | Weninger, Leon and Rippel, Oliver and Koppers, Simon and Merhof, Dorit |
| abstract | Brain tumor localization and segmentation is an important step in the treatment of brain tumor patients. It is the base for later clinical steps, e.g., a possible resection of the tumor. Hence, an automatic segmentation algorithm would be preferable, as it does not suffer from inter-rater variability. On top, results could be available immediately after the brain imaging procedure. Using this automatic tumor segmentation, it could also be possible to predict the survival of patients. The BraTS 2018 challenge consists of these two tasks: tumor segmentation in 3D-MRI images of brain tumor patients and survival prediction based on these images. For the tumor segmentation, we utilize a two-step approach: First, the tumor is located using a 3D U-net. Second, another 3D U-net more complex, but with a smaller output size detects subtle differences in the tumor volume, i.e., it segments the located tumor into tumor core, enhanced tumor, and peritumoral edema. The survival prediction of the patients is done with a rather simple, yet accurate algorithm which outperformed other tested approaches on the train set when thoroughly cross-validated. This finding is consistent with our performance on the test set - we achieved 3rd place in the survival prediction task of the BraTS Challenge 2018. |
| google_scholar | https://scholar.google.com/scholar?hl=en&as_sdt=0%2C22&q=Segmentation+of+Brain+Tumors+and+Patient+Survival+Prediction%3A+Methods+for+the+BraTS+2018+Challenge&btnG= |
| bibtex | @inproceedings{weninger2018segmentation, title={Segmentation of Brain Tumors and Patient Survival Prediction: Methods for the BraTS 2018 Challenge}, author={Weninger, Leon and Rippel, Oliver and Koppers, Simon and Merhof, Dorit}, booktitle={International MICCAI Brainlesion Workshop}, pages={3--12}, year={2018}, organization={Springer}} |
## model
| | |
|-|-|
| description | Ensemble of two 3D U-Nets |
| provenance |  |
| architecture | CNN |
| learning_type | Supervised |
| format | pth |
| I/O | model I/O can be viewed [here](contrib_src/model/config.json) |
| license | model license can be viewed [here](contrib_src/license/model) |
## run
To run this model and view others in the collection, view the instructions on [ModelHub](http://app.modelhub.ai/).
## contribute
To contribute models, visit the [ModelHub docs](https://modelhub.readthedocs.io/en/latest/).
