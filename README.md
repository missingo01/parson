\# PARSON – Explainable Book Recommendation Engine



PARSON is a hybrid AI-based book recommendation system that combines:



• Semantic similarity retrieval  

• Intent classification  

• Book form inference  

• Hybrid ranking  

• Explainability  

• Monitoring \& drift detection  



---



\## Features



\- Natural language query input

\- Semantic vector search using embeddings

\- ML-based intent classification

\- Book form inference

\- Hybrid ranking (semantic + intent)

\- Explainable recommendations

\- Decision trace logging

\- Drift detection

\- Web UI + API



---



\## Project Structure



backend/

&nbsp; recommender.py

&nbsp; api.py

&nbsp; train\_intent\_model.py

&nbsp; evaluate.py

&nbsp; drift\_detector.py

&nbsp; analyze\_logs.py



frontend/

&nbsp; index.html

&nbsp; style.css

&nbsp; script.js



data/

&nbsp; books.csv

&nbsp; intent\_dataset.csv

&nbsp; evaluation\_queries.csv



models/

&nbsp; embeddings.npy

&nbsp; books.index

&nbsp; intent\_classifier.pkl



logs/

&nbsp; decision\_traces.jsonl

&nbsp; baseline\_queries.json



requirements.txt



---



\## Installation



1\. Create environment



python -m venv .venv  

.venv\\Scripts\\activate



2\. Install dependencies



pip install -r requirements.txt



---



\## Train Intent Classifier



python backend/train\_intent\_model.py



---



\## Run Backend API



uvicorn backend.api:app --reload



Open:



http://127.0.0.1:8000/docs



---



\## Run Frontend



Open:



frontend/index.html



---



\## Example Query



I want books about wizards and fantasy



---



\## Output



\- Book title

\- Author

\- Preview link

\- Explanation

\- Semantic Match %

\- Intent Match %

\- Final Score %



---



\## Evaluation



python backend/evaluate.py



---



\## Drift Detection



python backend/drift\_detector.py



---



\## License



Educational Project



