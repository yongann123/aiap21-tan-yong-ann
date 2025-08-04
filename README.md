## ✅ `README.md` – AIAP 21 Technical Assessment

# AIAP 21 Technical Assessment – ElderGuard Analytics

**Full Name**: Tan Yong Ann  
**Email**: yongann.tan@gmail.com

---

## 🔧 Project Structure Overview

```

.
├── data/
│   ├── gas\_monitoring.db             # Not committed to GitHub
│   ├── loaded\_data.csv               # Output of data\_loader.py
│   ├── cleaned\_data.csv              # Output of preprocessing.py
│   └── featured\_data.csv             # Output of feature\_engineering.py
├── models/
│   ├── logistic\_regression.pkl
│   ├── random\_forest.pkl
│   ├── xgboost.pkl
│   ├── activity\_level\_mapping.pkl
│   ├── \*\_best\_params.json
├── evaluation/
│   ├── \*.json / \*.csv / \*.png        # Evaluation metrics and plots
├── src/
│   ├── data\_loader.py
│   ├── preprocessing.py
│   ├── feature\_engineering.py
│   ├── train\_model.py
│   ├── evaluate\_model.py
├── eda.ipynb
├── run.sh
├── requirements.txt
└── README.md

````

---

## ▶️ How to Run the Pipeline

1. Place `gas_monitoring.db` inside the `data/` folder.
2. From the project root, execute:
   ```bash
   bash run.sh
    ```

This will execute the full pipeline:

1. Load data →
2. Preprocess →
3. Feature engineer →
4. Train & evaluate models

---

## ⚙️ Configurability

The following settings are easily adjustable:

* Train/test split (`train_model.py`)
* Model list: enable or comment in/out models in `train_model.py`
* Hyperparameter tuning ranges (`get_best_model()` function)

---

## 📊 Pipeline Flow Description

1. **Data Loading**:

   * Uses `sqlite3` to extract the full table from `gas_monitoring.db`

2. **Preprocessing** (`preprocessing.py`):

   * Normalization of text
   * Duplicate removal
   * Missing value handling (drop/impute)
   * Outlier removal (Z-score + IQR)
   * Ordinal encoding of categorical features

3. **Feature Engineering** (`feature_engineering.py`):

   * Created features:

     * `metal_oxide_mean`, `metal_oxide_std`
     * `CO2_mean`, `CO2_diff`

4. **Model Training** (`train_model.py`):

   * Encodes labels
   * Trains 3 models:

     * Logistic Regression (baseline)
     * Random Forest
     * XGBoost
   * Uses cross-validation + hyperparameter tuning
   * Saves best model + parameters

> **Note**: During XGBoost training, you may see the warning  
> `Parameters: { "use_label_encoder" } are not used.`  
> This is expected and safe to ignore — it's due to recent changes in the XGBoost library.  

5. **Model Evaluation** (`evaluate_model.py`):

   * Evaluates on a fresh stratified 20% split
   * Metrics:

     * Accuracy, F1 Score
     * Confusion matrix
     * Classification report
   * Visualizes confusion matrix with `seaborn`

---

## 🧪 Summary of EDA Key Findings

* **Outliers & Invalids** were found in `Temperature`, `Humidity`, `CO2_InfraredSensor`, and removed.
* **HVAC Operation Mode** and **Ambient Light Level** showed little correlation with `Activity Level` → dropped.
* **CO\_GasSensor** and **Time of Day** encoded ordinally.
* New engineered features revealed stronger correlations.

---

## 🧠 Model Comparison

| Model               | Accuracy | Weighted F1 | High Activity F1 |
| ------------------- | -------- | ----------- | ---------------- |
| Logistic Regression | \~0.63   | \~0.57      | 0.00             |
| Random Forest       | \~0.94   | \~0.94      | \~0.91           |
| XGBoost             | \~0.94   | \~0.94      | \~0.91           |

* Logistic regression failed to capture non-linear patterns. In particular, it consistently failed to classify minority classes such as "High Activity", even after applying class weighting and stratified sampling. This is likely due to the linear decision boundary assumption, which is unsuitable for the non-linear nature of the sensor data. Therefore, Random Forest and XGBoost were preferred due to their ability to model complex relationships and handle class imbalance more effectively.
* Random Forest and XGBoost significantly outperformed, handling class imbalance better and showing strong recall for "High Activity".

---

## 📦 Feature Processing Summary

| Feature Name                                 | Processing                            |
| -------------------------------------------- | ------------------------------------- |
| `Temperature`, `Humidity`                    | Outlier detection, normalized         |
| `CO2_*`, `MetalOxideSensor_*`                | Imputed, outliers removed, aggregated |
| `Time of Day`, `CO_GasSensor`                | Ordinal encoding                      |
| `HVAC Operation Mode`, `Ambient Light Level` | Dropped (uninformative)               |
| `metal_oxide_mean/std`, `CO2_mean/diff`      | Engineered features                   |

---

## 🤖 Explanation of Model Choices

To tackle the multi-class classification task of predicting resident activity level, we implemented and compared the following three models:

---

#### 1. Logistic Regression – *Baseline Model (Linear Classifier)*

- **Why it was chosen**:  
  Logistic regression is a fast, interpretable, and widely-used model that serves as a good baseline for multi-class classification tasks.

- **Strengths**:
  - Simple and efficient
  - Easy to interpret feature importance via coefficients
  - Handles moderately sized datasets well

- **Limitations in this task**:
  - Performance is limited due to its **linear decision boundary**, which cannot capture complex non-linear relationships in environmental sensor data.
  - Despite `class_weight='balanced'` and stratified sampling, it failed to detect the "High Activity" class — likely due to class imbalance.

---

#### 2. Random Forest – *Non-linear Tree-Based Ensemble*

- **Why it was chosen**:  
  Random Forest is an ensemble of decision trees that can model complex relationships and interactions between features, making it well-suited for this real-world sensor data.

- **Strengths**:
  - Handles **non-linearity and feature interactions** well
  - Naturally robust to outliers and noise
  - Provides **feature importance** insights
  - Performs well with minimal preprocessing

- **Performance in this task**:
  - Significantly improved detection of all classes, including "High Activity"
  - Achieved >93% accuracy and strong recall across the board

---

#### 3. XGBoost – *Gradient Boosted Tree Ensemble*

- **Why it was chosen**:  
  XGBoost is a high-performance boosting algorithm known for its accuracy and speed. It improves upon Random Forest by building trees sequentially to correct previous errors.

- **Strengths**:
  - Regularization prevents overfitting
  - Fast and optimized for large datasets
  - Excellent handling of **class imbalance**
  - Typically **state-of-the-art** in tabular data competitions

- **Performance in this task**:
  - Matched Random Forest’s performance (~94% accuracy)
  - Slightly more compact model and competitive recall for "High Activity"
  - May offer better generalization under future deployments

---

### 🧠 Summary of Strategy

- **Logistic Regression**: establishes a performance baseline  
- **Random Forest**: strong general-purpose learner with minimal tuning  
- **XGBoost**: pushes performance further using boosting and fine-tuned hyperparameters

---

### 📈 Evaluation of Models and Metrics Used

The models were evaluated using a stratified 80/20 train-test split to preserve class distribution, followed by standard classification metrics. Evaluation was conducted **after saving the best model** from cross-validation to simulate realistic deployment conditions.

---

#### 📏 Evaluation Metrics

| Metric              | Description |
|---------------------|-------------|
| **Accuracy**        | Proportion of correct predictions out of total predictions. While useful, it can be misleading in imbalanced datasets. |
| **F1 Score (Weighted)** | Harmonic mean of precision and recall, averaged across classes with weighting by support (class frequency). Better reflects performance in imbalanced scenarios. |
| **Precision**       | For a given class, the proportion of predicted positives that are actually correct. Important when **false positives** are costly. |
| **Recall**          | For a given class, the proportion of actual positives that were correctly predicted. Important when **false negatives** are risky (e.g., missing a "High Activity" case). |
| **Confusion Matrix**| Shows how predicted labels compare to actual labels, allowing us to identify which classes are being confused with each other. |

All metrics were calculated using scikit-learn's `classification_report`, `confusion_matrix`, and `f1_score` functions.

---

#### 📊 Observed Model Performance

| Model              | Accuracy | Weighted F1 | High Activity Recall | Comments |
|--------------------|----------|-------------|-----------------------|----------|
| Logistic Regression | ~0.63   | ~0.57       | 0.00                  | Linear model underfit the data, unable to predict "High Activity" at all. |
| Random Forest       | ~0.94   | ~0.94       | ~0.91                 | Strong recall and precision across all classes. Slightly better interpretability due to feature importances. |
| XGBoost             | ~0.94   | ~0.94       | ~0.91                 | Competitive with Random Forest. Slight edge in generalization potential and lower overfitting risk. |

---

#### 🧠 Interpretation and Final Thoughts

- **Logistic Regression** is not suitable for this problem due to its linear nature and inability to handle non-linearity or subtle inter-feature interactions.
- **Random Forest** offers robust performance with little tuning, effectively balancing precision and recall across all classes.
- **XGBoost** matches Random Forest performance but adds regularization and more control over model complexity, potentially making it more stable for real-world deployment.

Given these results, **Random Forest and XGBoost** are the two strongest candidates for production use, with XGBoost being slightly more versatile in terms of customization and robustness.


---

## 🚀 Deployment Considerations

* Models are serialized with `joblib`
* Label mapping stored separately for decoding predictions
* XGBoost or Random Forest recommended for deployment
* Future enhancements:

  * Calibrated probabilities
  * Real-time sensor integration
  * Class imbalance monitoring in production