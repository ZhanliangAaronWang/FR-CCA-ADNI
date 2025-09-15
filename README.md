# Fair CCA for Fair Representation Learning: An ADNI Study

This repository contains the implementation of **FR-CCA (Fair Representation Canonical Correlation Analysis)**, a novel method for fair representation learning that ensures projected features are independent of sensitive attributes while maintaining high correlation analysis performance.

## 📋 Overview

Traditional Canonical Correlation Analysis (CCA) methods often overlook fairness considerations, potentially leading to biased representations that amplify societal biases. Our FR-CCA method addresses this limitation by:

- Ensuring independence between learned representations and sensitive attributes
- Maintaining high correlation between different data modalities
- Optimizing for both fairness and classification performance
- Providing computational efficiency comparable to standard CCA

## 🧠 Method

FR-CCA extends traditional CCA by adding fairness constraints that prevent any classifier from predicting sensitive attributes (e.g., sex, race, age) based solely on the projected data. The method solves:

```
max trace(U^T X^T Y V)
subject to: U^T X^T X U = V^T Y^T Y V = I_R
           U^T X^T ẑ = V^T Y^T ẑ = 0
```

Where ẑ represents the centered sensitive attribute vector.

## 📁 Repository Structure

```
├── FRCCA_AV14_MRI_Update_X.ipynb    # Implementation for MRI modality
├── FRCCA_AV14_MRI_Update_Y.ipynb  # Implementation for AV1451 (Tau PET) modality
├── data/                          # Data directory (ADNI datasets)
├── utils/                         # Utility functions
└── README.md                      # This file
```

## 🔧 Requirements

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
pip install scipy shap jupyter
```

## 📊 Datasets

### ADNI (Alzheimer's Disease Neuroimaging Initiative)
- **MRI**: 66 cortical thickness features from structural MRI scans
- **AV1451**: 68 SUVR features from Tau PET scans
- **Participants**: 375 subjects (182 males, 193 females)
- **Diagnoses**: Cognitively Normal (CN), Mild Cognitive Impairment (MCI), Alzheimer's Disease (AD)

### Synthetic Data
- Multivariate Gaussian distributions with embedded sensitive attribute bias
- 500 samples with controllable correlation structure
- Used for method validation and parameter tuning

## 🚀 Usage
### For Synthetic X Modality:
```python
# Run the synthetic X notebook
jupyter notebook FRCCA_Synthetic_X.ipynb
```

### For Synthetic Y Modality:
```python
# Run the synthetic T notebook
jupyter notebook FRCCA_Synthetic_Y.ipynb
```

*Need access to ADNI data when running MRI and AV14 notebook*
### For MRI Modality:
```python
# Run the MRI notebook
jupyter notebook FRCCA_AV14_MRI_Update_X.ipynb
```

### For AV1451 (Tau PET) Modality:
```python
# Run the AV1451 notebook
jupyter notebook FRCCA_AV14_MRI_Update_Y.ipynb
```

Both notebooks include:
1. Data preprocessing and standardization
2. FR-CCA model training
3. Fairness evaluation (DPG, EOG, GSG metrics)
4. Classification performance assessment

## 📈 Key Results

Our FR-CCA method demonstrates:

- **Superior Fairness**: Significant reduction in demographic parity gap (DPG), equalized odds gap (EOG), and group sufficiency gap (GSG)
- **Maintained Accuracy**: Competitive classification performance compared to traditional methods
- **Computational Efficiency**: Time complexity similar to standard CCA: O((Dx + Dy)³ + N·Dx² + N·Dy² + N·Dx·Dy)
- **Clinical Relevance**: Identifies important brain regions consistent with Alzheimer's disease pathology

### Performance Comparison

| Method | GSG ↓ | DPG ↓ | EOG ↓ | Accuracy |
|--------|-------|-------|-------|----------|
| SVM | 0.329±0.042 | 0.081±0.040 | 0.084±0.037 | 0.612±0.033 |
| CCA | 0.239±0.208 | 0.028±0.024 | 0.015±0.017 | 0.635±0.026 |
| SF-CCA | 0.286±0.033 | 0.036±0.024 | 0.033±0.031 | 0.619±0.026 |
| MF-CCA | 0.242±0.042 | 0.015±0.013 | 0.026±0.022 | 0.621±0.015 |
| **FR-CCA** | **0.215±0.041** | **0.008±0.006** | **0.010±0.005** | **0.612±0.009** |

## 🔬 Key Features

### Fairness Metrics
- **Demographic Parity Gap (DPG)**: Measures score consistency across sensitive groups
- **Equalized Odds Gap (EOG)**: Ensures equal treatment across all label-attribute combinations
- **Group Sufficiency Gap (GSG)**: Validates prediction relevance regardless of sensitive attributes

### Brain Region Analysis
- **MRI Focus**: Memory, language, and visual processing regions (entorhinal cortex, temporal gyri, fusiform gyrus)
- **AV1451 Focus**: Tau pathology in sensory processing and emotional regulation areas (postcentral gyrus, temporal pole, anterior cingulate)

## 📝 Citation
```bibtex
@misc{hou2025fairccafairrepresentation,
      title={Fair CCA for Fair Representation Learning: An ADNI Study}, 
      author={Bojian Hou and Zhanliang Wang and Zhuoping Zhou and Boning Tong and Zexuan Wang and Jingxuan Bao and Duy Duong-Tran and Qi Long and Li Shen},
      year={2025},
      eprint={2507.09382},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.09382}, 
}
```
## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Alzheimer's Disease Neuroimaging Initiative (ADNI) for providing the neuroimaging data
- The research community for foundational work in fair machine learning and canonical correlation analysis

## 📞 Contact

For questions or collaboration opportunities, please open an issue in this repository.

---

**Note**: This implementation is for research purposes. Clinical applications should undergo appropriate validation and regulatory approval.
