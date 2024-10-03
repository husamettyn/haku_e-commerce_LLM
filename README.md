---

# HAKU e-Commerce Product Description Generator

This project was realized as part of the e-Commerce Hackathon organized by the Turkish Technology Team and Trendyol. The project aims to balance the inequality of opportunity for sellers living in rural areas by providing inexpensive and accessible solutions to the costs of copywriting and professional product shooting required to sell on e-commerce platforms. In this context, the project provides a solution to this problem by utilizing technologies such as LLM and Computer Vision on the amateur product image and short description received from the user.

In this project, the “Meta-LLaMA 3.2 Vision-Instruct” LLM model is used as the basis. This model has natural language processing as well as computer vision capabilities and was chosen because it is efficient and useful. Distortions caused by perspective shifts in the image were corrected by using various image processing algorithms such as Hough Lines. In order to make the image received from the user look more professional, a background replacement process was applied to the subject in the image, for which ComfyUI, which provides ease of use for the use of various Diffussion models, was preferred. Addiniotally, Gradio is used in this project for both frontend and backend services.


## Features

- **Model:** LLaMA 3.2 - 11B Vision Instruct model from Meta
- **Libraries used:**
  - `transformers` by Hugging Face
  - `torch` for deep learning framework
  - `Gradio` for 
- **Functionality:**
  - Generate text-based descriptions from image inputs & product info that user gave.
  - Generate fancy backgrounds using Diffusing models.
  - Take feedback from user about generated product descriptions and revise it.

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   ```

2. **Install dependencies:**
   
   Ensure you have Python 3.x installed. Then install the required packages with the following command:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Provide your Huggingface Login Token:**
   ```
   from huggingface_hub import login
   login("YOUR_LOGIN_TOKEN_HERE")
   ```
2. **Run the notebook end to end**
   You can do it shortly from "Run" menu in your notebook enviroment

3. **Go to website provided by Gradio**
   After execution of all code blocks, Gradio provides a public URL that you can access to project via web interface. Using this interface you are able to use project easily.

## Notes

- **GPU recommended:** This model requires significant memory. It is highly recommended to run it on a machine with a GPU.
- **Model weights:** Make sure to load the model correctly using Hugging Face's API.
- **Error handling:** If you encounter errors with `tie_weights`, make sure you tie model weights using the respective method before running inference.
