# -Detecting-Cyber-Attacks-in-Network-Logs-Using-Large-Language-Models
🚀 Cyber Threat Detection using Large Language Models (LLMs)
This project investigates how Large Language Models (LLMs), including GPT-4 and LLaMA2, can be utilized to enhance the accuracy and speed of detecting cybersecurity threats. Traditional Intrusion Detection Systems (IDS) often fail against complex threats like DDoS, SQL Injection, and Ransomware. Our solution leverages prompt engineering techniques with real-world network traffic data to build a proactive AI-based security layer.



Datasets include labeled traffic samples with attack and non-attack instances. We evaluate model output using precision, recall, and accuracy metrics, visualized using graphs and plots. The findings show that LLMs can offer superior threat detection and interpretability compared to traditional methods.

📁 Dataset
Sources: Real-world traffic datasets as referenced in:

De Jesus Coelho & Westphall (2024)

Ferrag et al. (2024)

Types of threats detected:

Distributed Denial of Service (DDoS)

SQL Injection

Ransomware

Other complex attacks

✅ Steps Followed in the Project


<h1>🔹 Step 1: Import Required Libraries </h1>
python

Edit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openai  # or HuggingFace for LLaMA2
Necessary libraries for data handling, visualization, and API interaction are imported.


<h1>🔹 Step 2: Mount Google Drive </h1>
python

Edit
from google.colab import drive
drive.mount('/content/drive')
Google Drive is mounted to access datasets and save outputs for later analysis.


<h1>🔹 Step 3: Connect to LLM API using API Key</h1>
```python

Edit
openai.api_key = "YOUR_API_KEY"
You can use APIs like OpenAI (GPT-4) or HuggingFace (LLaMA2).
Set environment variables or use Colab secrets for safe API key storage.
```

<h1>🔹 Step 4: Give LLM Access to Data and Perform Basic Operations</h1>
Loop through dataset rows and feed traffic logs to the LLM.


<h1>🔹 Step 5: Classify Files </h1>
Each row of the dataset is classified into Attack or Non-Attack.

Store the results in a structured format (.csv or DataFrame).

python
Copy
Edit
results = []
for index, row in df.iterrows():
    prompt = f"Classify this traffic data: {row['log']}"
    response = openai.ChatCompletion.create(...)
    results.append(response)

![App Screenshot 1](app1.png)

    
<h1>🔹 Step 6: Evaluate Results </h1>
Evaluate results using traditional metrics:

Precision

Recall

F1 Score

Accuracy

Compare performance of LLM models vs traditional methods.

python
Copy
Edit
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))



<h1>🔹 Step 7: Evaluate Results with Graphs </h1>
Use matplotlib or seaborn to create:

Confusion matrix

Precision-recall curves

Model comparison bar charts

python
Copy
Edit
import seaborn as sns
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True)


<h1>here is Datasets link https://drive.google.com/drive/folders/1unaf_mw7BHYIj0EBPFwXoNsV9PFOqTgY?usp=sharing  </h1>






