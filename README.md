# 🧊 Synthetic Data Generator (with Gemini AI)

> **Day 3 of my GenAI + Data Engineering Learning Path**

A privacy-conscious, AI-powered synthetic data generator built with **Python**, **Streamlit**, and **Gemini API**. Easily define schemas, generate fake datasets, and preview + download them — all through a clean web UI.

---

## 📌 Features

- 🔧 **Custom Schema Input**  
  Define your dataset structure using:
  - Manual JSON input
  - Upload `.json` schema file
  - Auto-infer schema from `.csv`

- 🧠 **AI-Powered Schema Explanation**  
  Uses Google **Gemini API** to generate plain-English descriptions of complex JSON schemas.

- 📊 **Data Preview + Insights**  
  View the first 10 rows of generated data  
  See type distribution, memory usage, and sample column values

- 📥 **Download as CSV**  
  Export your generated dataset instantly

- 🎨 **Report-like UI**  
  Clean, responsive layout styled with custom CSS

---

## ⚙️ Tech Stack

| Tool       | Purpose                            |
|------------|------------------------------------|
| 🐍 Python   | Core logic                         |
| 📦 Streamlit | Web UI & interactivity            |
| 📊 Pandas   | Data handling & preview            |
| 🧪 Faker    | Generate fake, realistic data      |
| 🤖 Gemini API | AI schema explanation             |
| 🔐 dotenv   | Secure API key management          |
---

Absolutely! Here's the **rest of the README** in clean markdown format — pick up from after the Tech Stack section:

````markdown
---

## 🚀 Getting Started

### 🔧 Install Dependencies

```bash
pip install -r requirements.txt
````

### 📁 Create a `.env` File

```env
GEMINI_API_KEY=your_api_key_here
```

### ▶️ Run the App

```bash
streamlit run main_app.py
```

---

## 🌐 Try It Live

* 🔗 **Live App**: \[your-streamlit-link-here]
* 💻 **GitHub Repo**: \[this repo]

---

## 🛠 Example Schema Input

```json
{
  "name": {"type": "name"},
  "email": {"type": "email"},
  "joined_on": {"type": "date"},
  "age": {"type": "int", "params": {"min": 18, "max": 60}},
  "is_active": {"type": "boolean"}
}
```

---

## 📈 Screenshots

> *(Add screenshots of your UI here – data preview, schema explanation, insights panel, etc.)*

---

## 🧠 My Learnings (Day 3 Highlights)

* Debugged complex schema parsing + JSON edge cases
* Integrated Gemini API with secure `.env`-based handling
* Created modular AI logic with fallback handling
* Improved UX using `st.tabs()`, `st.expander()`, and layout containers
* Learned responsive layout tuning in Streamlit

---

## 🧩 Folder Structure

```
├── main_app.py          # Main Streamlit app
├── gemini_utils.py      # Gemini integration logic
├── .env                 # API key (excluded from repo)
├── requirements.txt     # Project dependencies
└── README.md            # You are here
```

---

## 🙌 Acknowledgements

* [Google Gemini](https://ai.google.dev/)
* [Streamlit](https://streamlit.io/)
* [Faker](https://faker.readthedocs.io/)
* [Pandas](https://pandas.pydata.org/)
* [Python dotenv](https://pypi.org/project/python-dotenv/)

---

## 📬 Connect with Me

> Built as part of my **#100DaysOfGenAI + Data Engineering** learning sprint.

* 🔗 [LinkedIn](linkedin.com/in/abhishek-s-a59942239)
* ✉️ [Email](abhisheksndr@gmail.com)

---

## 🏷 Hashtags

`#SyntheticData` `#Python` `#Streamlit` `#GeminiAPI` `#Faker`
`#DataEngineering` `#OpenToWork` `#DevTools` `#AItools` `#LearningInPublic`

```

✅ Just replace the placeholder links (`your-profile`, `your-streamlit-link-here`, etc.) and you're good to commit this to GitHub.

Want me to bundle it into a `README.md` file for direct upload?
```
