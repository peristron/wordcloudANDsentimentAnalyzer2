# 🧠 AI-Powered Word Cloud Generator

A secure, streaming-based Word Cloud generator built with Streamlit. It processes large text files (CSV, Excel, VTT) efficiently and integrates AI (OpenAI/xAI) to analyze underlying themes.

## 🚀 Features

*   **Multi-File Support:** Upload multiple CSV, Excel, or Transcript files at once.
*   **Memory Safe:** Uses streaming processing to handle files larger than RAM.
*   **AI Integration:** Authenticated users can use GPT-4o or Grok to generate qualitative insights from word frequency patterns.
*   **Sentiment Analysis:** Built-in VADER sentiment scoring to color-code words (Positive/Negative/Neutral).
*   **Smart Cleaning:** Automatically removes chat artifacts (Zoom timestamps, Slack emojis), HTML tags, and URLs.

## 🛠️ Setup & Installation

1.  **Clone the repo:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    streamlit run app_wordcloud_custom_v25.py
    ```

## 🔐 Configuration (Secrets)

To use the AI features, you must configure secrets.
**Locally:** Create a `.streamlit/secrets.toml` file.
**Streamlit Cloud:** Go to App Settings -> Secrets.

```toml
auth_password = "your_app_password"
openai_api_key = "sk-..."
xai_api_key = "..."
