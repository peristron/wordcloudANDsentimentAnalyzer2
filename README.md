
# 🧠 AI-Powered Word Cloud & Network Analyzer

A secure, streaming-based text analysis dashboard built with Streamlit. This tool goes beyond simple word clouds by integrating **Interactive Network Graphs**, **Text Statistics**, and **AI-Powered Insight Generation**.

It is optimized for performance, capable of processing large datasets (CSV, Excel, VTT) efficiently via streaming, without crashing memory.

## 🚀 Key Features

### 📊 Visualization & Analytics
*   **Word Clouds:** customizable visualizations with sentiment-based coloring.
*   **Interactive Knowledge Graph:** Visualize how words connect to one another using a physics-based network graph (based on co-occurrence/bigrams). Drag nodes to explore relationships.
*   **Text Statistics:** Instant dashboard for **Lexical Diversity**, Average Word Length, and Unique Vocabulary counts.
*   **Graph Metrics:** Automatically calculates "Centrality" to identify bridge words and key influencers in the text.

### 🤖 AI & Intelligence
*   **AI Integration:** Authenticated users can send frequency data to **GPT-4o** or **Grok (xAI)** to generate qualitative themes and detect anomalies.
*   **Sentiment Analysis:** Built-in VADER sentiment scoring (Positive/Negative/Neutral) applied to both clouds and graphs.

### ⚡ Performance & Security
*   **Memory Safe:** Uses streaming processing to handle files larger than RAM (1GB+).
*   **Smart Cleaning:** Automatically removes chat artifacts (Zoom timestamps, Slack emojis), HTML tags, and URLs.
*   **Multi-File Support:** Upload and merge data from multiple CSV, Excel, or Transcript files at once.

## 🛠️ Setup & Installation

1.  **Clone the repo:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Install dependencies:**
    *(Note: This now includes `networkx` and `streamlit-agraph`)*
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

## 🔐 Configuration (Secrets)

To use the Generative AI features (GPT/Grok), you must configure secrets.
**Locally:** Create a `.streamlit/secrets.toml` file.
**Streamlit Cloud:** Go to App Settings -> Secrets.

```toml
auth_password = "your_app_password"  # Password to unlock AI features in the UI
openai_api_key = "sk-..."            # Optional: For OpenAI
xai_api_key = "..."                  # Optional: For xAI (Grok)
```

## 📖 Usage Guide

1.  **Upload Data:** Drag and drop your files.
2.  **Input Options:** If using CSV/Excel, use the "🧩 Input Options" expander to select specific text columns (e.g., "Comments", "Transcript").
3.  **Visuals:** The Word Cloud updates automatically. Scroll down and check **"🕸️ Show Network Graph"** to visualize relationships.
4.  **AI Analysis:** Enter the password in the sidebar to unlock the **"✨ Analyze Themes"** button, which sends the top data patterns to the AI for interpretation.
