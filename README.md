# One-Shot Face Stylization 🎨

A full-stack deep learning web app I built to turn real faces into different art styles (Anime, Sketch, Cyberpunk, Watercolor) using GAN Inversion. 

Instead of just leaving my PyTorch code in a Jupyter Notebook, I wanted to build a proper end-to-end application to see what it takes to serve heavy ML models to a web browser.

*(Insert your demo.gif here)*

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, StyleGAN2, e4e (Encoder for Editing)
* **Backend:** Python, FastAPI, Uvicorn
* **Frontend:** HTML5, Vanilla JavaScript, Tailwind CSS

## 🧠 How it Works Under the Hood
1. **The Frontend:** Takes an image upload and sends it as `FormData` to the FastAPI server. I built a custom "System Log" UI to track the fetch requests because generating these images takes time!
2. **GAN Inversion (e4e):** The backend takes the uploaded face and projects it into the `W+` latent space of a pre-trained StyleGAN2 model.
3. **Generation:** The latent code is passed into a fine-tuned StyleGAN2 generator based on the style the user selected.
4. **The Response:** The PyTorch tensor is converted back into a JPEG and sent back to the browser as a raw Blob, which is then rendered on the screen without refreshing the page.

## 🚀 How to Run it Locally

### 1. Clone & Install
git clone https://github.com/YOUR_USERNAME/one-shot-face-stylization.git
cd one-shot-face-stylization
pip install -r backend/requirements.txt


### 2. Add the Model Weights
**Note:** I couldn't upload the `.pt` model files here because of GitHub's file size limits. 
To run this yourself, you need to create a `checkpoints/` folder inside the `backend/` directory and drop your StyleGAN2 `.pt` files in there. 

### 3. Start the Backend
cd backend
python -m uvicorn app:app --reload

The FastAPI server will start on `localhost:8000` and load the models into RAM.

### 4. Start the Frontend
Just open `frontend/index.html` in your browser (or use VS Code Live Server). 

## 🐛 What I Learned (and fixed)
Serving heavy deep learning models locally is tricky. One of the biggest challenges I faced was browser timeout and refresh behaviors. Because PyTorch takes about 20-30 seconds to run inference on a CPU, standard HTML `<form>` tags kept trying to refresh the page before the Python server could send the image back. I had to rewrite the frontend to use pure asynchronous JavaScript `fetch()` calls and handle binary Blob data directly to make the connection stable!
