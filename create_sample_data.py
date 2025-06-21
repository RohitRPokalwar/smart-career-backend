import pandas as pd
import os

data = {
    "skills": [
        "python, pandas, numpy, matplotlib",
        "html, css, javascript",
        "python, machine learning, deep learning",
        "linux, networking, firewall, wireshark",
        "react, node.js, mongodb, express",
        "excel, data cleaning, matplotlib",
        "cybersecurity, kali linux, wireshark",
        "tensorflow, keras, pandas, sklearn"
    ],
    "domain": [
        "Data Analyst",
        "Web Development",
        "AI/ML",
        "Cybersecurity",
        "Web Development",
        "Data Analyst",
        "Cybersecurity",
        "AI/ML"
    ]
}

os.makedirs("data", exist_ok=True)

df = pd.DataFrame(data)
df.to_csv("data/career_dataset.csv", index=False)
print("✅ Sample data written to 'data/career_dataset.csv'")
