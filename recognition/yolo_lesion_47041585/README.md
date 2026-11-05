# YOLO11 for Lesion Detection and Triage on ISIC2018 Challenge Dataset

<div align="center">

**COMP3710 - Pattern Recognition and Analysis**

| Task 3 | Lesion Detection: Dermoscopic Screening and Triage |
| :----- | :------------------------------------------------- |
| Author | Alex Janevik (47041585)                            |

**put a training visualisation here, perhaps a gif if i can**

[Installation](./docs/INSTALLATION.md) • [Usage](./docs/USAGE.md)

</div>

---

This project trains a YOLO11 object detection model on the ISIC2018 Challenge Dataset to detect and classify skin lesions as either benign or malignant. The primary motivation is to assist clinicians in the triaging of cases by flagging potentially malignant lesions for review while minimising any unnecessary escalated of benign cases. Great care is taken to ensure the model is trained and evaluated on patient-independent splits to avoid data leakage and ensure that the model's performance is representative of real-world clinical scenarios.

Training YOLO11 on the ISIC2018 dataset follows the 3 core stages of the YOLO11 architecture [\[4\]](#references):

1. **Backbone**: Extract features from dermoscopic images at multiple resolutions to produce 3 multi-scale feature maps.
2. **Neck**: Features across the scales are fused to provide context for lesions of different sizes.
3. **Head**: Bounding boxes and classifications are predicted from the fused features.

| \#    | Section                                                                               |
| ----: | :-------------------------------------------------------------------------------------|
| 1     | [Project Structure](#project-structure)                                               |
| 2     | [Initial Feasibility Review](#initial-feasibility-review)                             |
| 3     | [Data Pre-processing](#data-pre-processing)                                           |
| 4     | [Benchmarks and Results](#benchmarks-and-results)                                     |
| 4.1   |   [Baseline model](#baseline-model)                                                   |
| 4.2   |   [Config Tuning and Candidate Model](#config-tuning-and-candidate-model)            |
| 4.3   |   [Comparison](#comparison)                                                           |
| 5     | [Resource Profiling](#resource-profiling)                                             |
| 6     | [Qualitative Error Autopsy](#qualitative-error-autopsy)                               |
| 7     | [Trade-Off Synthesis and Recommendation](#trade-off-synthesis-and-recommendation)     |
| 8     | [Artificial Intelligence Usage Disclosure](#artificial-intelligence-usage-disclosure) |
| 9     | [References](#references)                                                             |
| 10    | [Citation](#citation)                                                                 |

## Project Structure

```ansi
yolo_lesion_47041585/
├── README.md
├── pyproject.toml
├── uv.lock
├── configs/
│   ├── a0.yaml
│   ├── b0.yaml
│   ├── base.yaml
│   ├── mps.yaml
│   ├── o35.yaml
│   └── o50.yaml
├── dataset/
│   ├── metadata_raw.csv
│   └── {ISIC2018_...}
├── runs/
├── scripts/
│   ├── fetch_data.sh
│   └── train.slurm
├── src/
│   └── yolo_lesion_47041585/
│       ├── audit.py
│       ├── dataset.py
│       ├── modules.py
│       ├── predict.py
│       ├── train.py
│       └── helpers/
│           ├── evaluation.py
│           ├── policy.py
│           └── visualisation.py
└── weights/
```

## Initial Feasibility Review

### User need, scope and acceptance criteria

This project aims to investigate whether an image detection model can localise annotated lesions in ISIC2018 dermoscopic images, flagging malignant cases for clinical review whilst limiting unnecessary escalation of benign cases. This project will evaluate this trade-off within the supplied dataset rather than establish clinical readiness. The target audience of such a model is dermatologists and other clinicians who may use it to assist in triaging cases for review.

To consider this project a success, the following acceptance criteria must be met:

| Acceptance criterion | Proposed target | Evaluation method |
| :--- | :--- | :--- |
| Malignant lesion detection sensitivity $m$. | $m \geq 90\%$, with a stretch goal of $m \geq 95\%$. | Calculate the percentage of all malignant lesions that are correctly localised and labelled malignant. A bounding box is correct at IoU $\geq 0.5$. Missing or misclassifying a malignant lesion counts as a failure. |
| Unnecessarily escalated benign cases $b$. | $b \leq 30\%$ when measured at the same operating point that achieves target $m$. | Benign escalation will be reported separately for monitoring cases and urgent referrals. Both monitoring and urgent referrals count towards $b$ - absent, conflicting or low-confidence detections are assigned for monitoring. |
| Quality of localisation. | Mean matched-box IoU $\geq 0.8$ per class. | Calculate the average IoU of correctly matched boxes separately for benign and malignant lesions. The number of lesions that were successfully and unsuccessfully matched will be included in this result. |
| Mean processing latency $t$ (seconds). | $t \leq 1$ per image at a batch size of 1. This includes preprocessing, CPU-GPU transfers, inference and post-processing. This does not include file reading/writing. | Measure on UQ Rangpur hardware after a warmup run. GPU model, input resolution and number of timed images will be included in this result. |

> Baseline and candidate models will be compared on the same test split using $\text{mAP}_{50}$, $\text{mAP}_{50:95}$, precision, recall and per-class localisation IoU. Inference throughput (FPS) will also be reported. Thresholds will be selected using validation data to assess malignant sensitivity against benign monitoring and urgent-referral rates. 3-5 representative failure cases in the validation data will also be visualised and analysed.

### Model choice and course concepts

The chosen detector will be YOLO11 due to its ability to predict bounding boxes and class labels within a single detection pass. Bounding boxes will be derived by training with the supplied segmentation masks in the ISIC2018 dataset. The corresponding image metadata will be used to train the classifier to make binary predictions - either "benign" or "malignant". A COCO-pretrained YOLO11n detector tuned on the training split will be used as the baseline detector. It will have a fixed configuration and no further tuning. This will be used for testing the project's runtime and VRAM constraints. Individually motivated tweaks will be made to tune the candidate model based on early failure cases. If tuning YOLO11n does not meet the performance targets, YOLO11s or YOLO11m will be considered within the computing budget.

This task requires class prediction, independent evaluation, patient-level splitting and memory/runtime trade-offs - all concepts covered in COMP3710. Using the YOLO11 detector qualifies this project as **Normal Difficulty**.

### Preliminary feasibility evidence

Metadata records were downloaded via the `isic-cli` tool and ISIC2018 test set images, training set images and training set segmentation masks are stored on Rangpur. Segmentation masks for the test set were manually downloaded from the ISIC Archive [\[1\]](#references)-[\[3\]](#references).

The preliminary audit identified 3,594 image-mask pairs. Excluding the 22 records with indeterminate diagnoses left 3,572 records, with 2,313 of these lacking patient identifiers. ISIC makes no concrete claims about records without patient IDs so it cannot be assumed that a record without a patient ID is independent of other records (including those that do have a patient ID). This means that to avoid data leakage and preserve patient-independence across the testing and validation sets, the *entire* dataset is sanitised by removing all records without patient IDs - even those in the training set. This leaves 1,259 binary-labelled images from 999 identified patients, including a total of 241 malignant samples. The audit also identified 59 repeated lesion IDs across 124 images before patient ID filtering, with 97 images sharing 47 lesion IDs after filtering.

The retained records come from the original training and test sets - 390 and 869 images respectively. These sets will be merged and re-split in a 70/15/15 training-validation-test split prior to training. It is imperative to keep all images from each patient together and check for any overlapping lesion IDs. Records with missing lesion IDs will be retained where patient IDs are available, as overlapping lesion IDs indicate a different photo of the same patient's lesion. Metadata pre-sanitising and filtering, mask pairing and bounding-box extraction have also been completed. An example with candidate bounding boxes is shown below.

![Example of a candidate bounding box derived from a segmentation mask](assets/audit-box-overlays.jpg)

### Risks, budget and fallback

The primary risk in this project was the missing patient identifiers in the data, but the solution to this leads to a secondary risk of having too few training records to train a model. The dataset is now very imbalanced, with only 241 of the 1,259 usable images being classified as malignant. To address the class imbalance, oversampling of malignant training images with bounding box-aware augmentation will be compared against the baseline. Validation and test sets will not be oversampled or augmented. Improvement will be assessed by using malignant detection recall at the same benign-escalation limit. Per-class detection metrics will also be reported.

Experiments will be run using a single NVIDIA A100 GPU via the COMP3710 Slurm partition on UQ Rangpur. There is no account limit on Rangpur, but a provisional budget of 12 GPU hours is allocated for this project:

- 1 hour for data-loading and training smoke tests
- 3 hours for training the baseline YOLO11n model.
- 3 hours for the oversampling experiments.
- 3 hours for one YOLO11s or YOLO11m candidate experiment.
- 2 hours for evaluation and analysis.

The smoke test will measure epoch time and peak VRAM usage to set an epoch limit for the controlled comparisons. If the projected runtime exceeds this budget, the number of experiments will be reduced. YOLO11s or YOLO11m will only be tested if there are resources remaining.

## Data Pre-processing

## Benchmarks and Results

### Baseline Model

### Config Tuning and Candidate Model

### Comparison

## Resource Profiling

## Qualitative Error Autopsy

## Trade-Off Synthesis and Recommendation

## Artificial Intelligence Usage Disclosure

In the development of this project, Claude Sonnet 5.5 was used to assist in several areas, including by not strictly limited to:

- Docstring generation and formatting (prompted to follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)) and bulk comments for YOLO configurations.
- In-depth explanations of specific concepts and outputs, such as:
  - YOLO11 architecture
  - `json` and `csv` dumps of results
  - Potential effects of tuning different hyperparameters on model performance. Due to the set budget and time requirements, many different configurations of the model could not be tested. Many of my suggested configurations were expected by Claude to yield low value, and these were unfortunately excluded from testing. All tested hyperparameter choices were made by myself and can be justified by my previous ML experience and understanding of the topics.
- Planning review. I produced a very in-depth project plan prior to development based on my preliminary data audit and self-determined requirements in the [initial feasibility review](#initial-feasibility-review). This plan included basic requirements for each module and the expected flow of data. It was then grilled and reviewed by Claude to ensure it was consistent and left no room for ambiguity.
- Code boilerplating. Based on the project plan, modules were prefilled with boilerplate functions and classes to hold me to a strict implementation plan. It should be noted that it would not be feasible to expect the foreseeing of all implementation details at the planning stage, so some boilerplate was discarded or modified as the project progressed.
- Code review and refactoring suggestions. Specifically:
  - My code was refactored a number of times during development and testing to improve readability.
  - Changes were suggested in the planning stages of the project, such as using the `dataclasses` module to simplify the writing and reading of data outputs in `dataset.py` and `audit.py`. This greatly improved type safety and was scaled across other applicable modules.
  - The use of `yaml` configuration files for hyperparameter tuning was an early suggestion when tuning the candidate model. This allowed for much faster experimentation and makes the specific changes much easier to track and read.
- Generation of output scripts such as `profile_latency.py`, `autopsy.py`, `score.py` and `final_test.py`. These were largely written by Claude as they primarily consist of interpreting bulk output data to produce a readable result for myself to verify. This code contains very critical information and was proof read and grilled by myself to ensure understanding and correctness.
- Complete refactoring and rewriting of visualisation scripts. After building the first visualisation script, it was obvious that it could be cleaned up and automated to make it easier for myself to customise and generate figures for the report.

> [!IMPORTANT]
> It is important to note that while AI assistance is seen in many areas of this project, the final decisions and implementation were all made and can be justified by myself. No decisions regarding data handling, model architecture, hyperparameter tuning, evaluation, etc. were blindly made by an LLM. Great care was taken to ensure that all decisions and assumptions were made based on my own testing and understanding of the project's requirements and provided data.

## References

\[1\] P. Tschandl, C. Rosendahl, and H. Kittler, “The HAM10000 dataset,
a large collection of multi-source dermatoscopic images of common
pigmented skin lesions,” *Sci Data*, vol. 5, no. 1, p. 180161, Aug.
2018, doi:
[10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161).

\[2\] N. C. F. Codella *et al.*, “Skin Lesion Analysis Toward Melanoma
Detection: A Challenge at the 2017 International Symposium on Biomedical
Imaging (ISBI), Hosted by the International Skin Imaging Collaboration
(ISIC),” Jan. 08, 2018, *arXiv*: arXiv:1710.05006. doi:
[10.48550/arXiv.1710.05006](https://doi.org/10.48550/arXiv.1710.05006).

\[3\] N. Codella *et al.*, “Skin Lesion Analysis Toward Melanoma
Detection 2018: A Challenge Hosted by the International Skin Imaging
Collaboration (ISIC),” Mar. 29, 2019, *arXiv*: arXiv:1902.03368. doi:
[10.48550/arXiv.1902.03368](https://doi.org/10.48550/arXiv.1902.03368).

\[4\] M. Raimbekov and G. Jocher, “YOLO Architecture Explained: From
YOLOv3 to YOLO26,” Ultralytics Docs. \[Online\]. Available:
<https://docs.ultralytics.com/guides/yolo-architecture>

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{janevik2026yolo11,
    title={YOLO11 for Lesion Detection and Triage on ISIC2018 Dataset},
    author={Alex Janevik},
    year={2026},
    url={https://github.com/alexjanevik/PatternAnalysis-2026/tree/topic-recognition/recognition/yolo_lesion_47041585}
```
