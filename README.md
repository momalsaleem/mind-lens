# Mental Health Narrator
> Explaining AI mental health risk predictions in plain English using local LLMs.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![SHAP](https://img.shields.io/badge/XAI-SHAP-green)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)

## Problem
Mental health classifiers are black boxes — they predict risk but 
don't explain why, making them hard to trust in real-world settings.

## Solution
A 3-stage pipeline that classifies, explains, and narrates:
```
Text Input
    │
    ▼
Emotion Classifier (DistilRoBERTa)
    │
    ▼
SHAP — identifies top influential words
    │
    ▼
Mistral 7B — generates plain-English explanation
    │
    ▼
Gradio UI — clean interface with examples
```

## Results

| Metric | Score |
|--------|-------|
| Faithfulness rate | 6/8 (75%) |
| Avg confidence | ~80% |
| Explanation quality | 6/8 clear |

### Evaluation Results (Test Set — 8 samples)

| Text | Prediction | Confidence | Faithful |
|------|------------|------------|----------|
| I feel completely empty...         | fear    | 88.0% | Yes |
| Can't stop worrying...             | fear    | 81.0% | Yes |
| Had a great day with family...     | joy     | 88.0% | Yes |
| The flashbacks keep coming...      | joy     | 53.0% | Yes |
| Lost interest in everything...     | sadness | —     | Yes |
| Feeling anxious about tomorrow...  | fear    | —     | Yes |
| Don't see the point...             | sadness | 86.0% | Yes |
| Went for a walk today...           | neutral | 83.0% | No  |

### Summary
- **Faithfulness rate**: 6/8 (75%)
- **Average confidence**: ~80%
- **Strongest predictions**: fear (88%), joy (88%), neutral (83%)
- **Notable finding**: "The flashbacks keep coming" was predicted as joy (53%) — lowest confidence in the set, suggesting the model struggles with trauma-related text that lacks explicit negative emotion words
- **Failure case**: "Went for a walk today" explanation did not reference SHAP top words — LLM defaulted to generic response when confidence signal was weak

### Key Insight
The model correctly identifies distress signals in high-confidence cases but misclassifies trauma/PTSD-style text — a meaningful limitation for real-world mental health applications.

## Limitations
- Not a diagnostic tool — research purposes only
- LLM explanations can be vague when SHAP values are close to zero
- Model trained on general emotion data, not clinical text

## Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK_HERE)

1. Open the notebook link above
2. Runtime → Change runtime type → T4 GPU
3. Run all cells in order

## Tech Stack
- **Classifier:** j-hartmann/emotion-english-distilroberta-base
- **Explainer:** SHAP (Partition Explainer)
- **LLM:** Mistral 7B Instruct v0.2
- **UI:** Gradio
