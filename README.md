# RATER

This repository contains the modeling and app code for the paper *"Leveraging Artificial Intelligence in Content Validity Assessment: Development, Illustration, and Evaluation of Two Classes of AI Models"*:

Pillet, J. C., Larsen, K., Dobolyi, D., Queiroz, M., Handler, A., Arnulf, K., Sharma, R. (2025). AI-Augmented Content Validation in Behavioral Research: Development and Evaluation of the RATER System. MIS Quarterly, Forthcoming. https://doi.org/10.25300/MISQ/2025/18946

---
### App

The app is written using [Streamlit](https://streamlit.io). To run the app:
- add a RATER<sub>C</sub> model (e.g., our [RATER-C](https://huggingface.co/dobolyilab/RATER-C) model from Hugging Face) to the app's parent folder
- add an [OpenAI API key](https://platform.openai.com/api-keys) (if using the pre-specified closed-weights model; alternatively, an open-weights model can be specified via an OpenAI compatible API endpoint using [vLLM](https://docs.vllm.ai/en/stable/), etc.)
- (optional) for email and remote FastAPI server functionality, create a secrets.toml file in the .streamlit folder of the app's parent folder and enter various information needed for app.py and process.py (i.e., find all values that depend on `st.secrets`)
- run the app via `streamlit run app.py`

---
### RATER-C and RATER-D

These folders contain the Python code and data necessary to recreate and inference the various RATER<sub>C</sub> and RATER<sub>D</sub> models reported in the paper.

Top models from the paper are available on Hugging Face for both [RATER-C](https://huggingface.co/dobolyilab/RATER-C) and [RATER-D](https://huggingface.co/dobolyilab/RATER-D).
