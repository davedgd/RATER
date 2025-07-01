# RATER

This repository will contain the modeling and app code for the paper *"Leveraging Artificial Intelligence in Content Validity Assessment: Development, Illustration, and Evaluation of Two Classes of AI Models"*.

---
### App

The app is written using [Streamlit](https://streamlit.io). To run the app:
- add a RATER<sub>C</sub> model (e.g., our [RATER-C](https://huggingface.co/dobolyilab/RATER-C) model from Hugging Face) to the app's parent folder
- add an [OpenAI API key](https://platform.openai.com/api-keys) (if using the pre-specified closed-weights model; alternatively, an open-weights model can be specified via an OpenAI compatible API endpoint using [vLLM](https://docs.vllm.ai/en/stable/), etc.)
- run the app via `streamlit run app.py`
