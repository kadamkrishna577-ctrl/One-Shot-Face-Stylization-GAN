import os
import io
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# We assume you copied 'model.py' (Generator class) and 'e4e_projection.py' 
# from the JoJoGAN repo into your backend folder.
from model import Generator
from e4e_projection import projection

app = FastAPI(title="One-Shot Stylization API")

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your frontend URL
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_DIR = "checkpoints"

# Dictionary to hold our pre-loaded models in RAM for instant inference
loaded_models = {}

@app.on_event("startup")
async def load_models():
    """Loads all .pt style checkpoints into memory when the server starts."""
    print(f"Initializing models on {DEVICE}...")
    
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        
    for filename in os.listdir(CHECKPOINT_DIR):
        if filename.endswith(".pt"):
            style_name = filename.split(".")[0]
            print(f"Loading {style_name}...")
            
            gen = Generator(1024, 512, 8, 2).to(DEVICE)
            ckpt = torch.load(f"{CHECKPOINT_DIR}/{filename}", map_location=DEVICE)
            gen.load_state_dict(ckpt, strict=False)
            gen.eval()
            
            loaded_models[style_name] = gen
            
    print(f"Successfully loaded {len(loaded_models)} styles: {list(loaded_models.keys())}")

@app.post("/stylize")
async def stylize_image(
    file: UploadFile = File(...), 
    style: str = Form(...)
):
    if style not in loaded_models:
        raise HTTPException(status_code=400, detail=f"Style '{style}' not found. Available styles: {list(loaded_models.keys())}")
        
    try:
        # 1. Read the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Note: In a production app, you would run the dlib alignment step here
        # before projection. For this demo, we assume the user uploads a cropped face.
        
        # 2. Invert the image to get the latent code (using e4e)
        # (Requires the e4e_ffhq_encode.pt model to be in a 'models/' folder locally)
        latent_code = projection(image, name='api_upload', device=DEVICE)
        if latent_code.dim() == 2:
            latent_code = latent_code.unsqueeze(0)
            
        # 3. Generate the stylized face
        generator = loaded_models[style]
        with torch.no_grad():
            stylized_tensor = generator(latent_code, input_is_latent=True, randomize_noise=False)
            if isinstance(stylized_tensor, tuple) or isinstance(stylized_tensor, list):
                stylized_tensor = stylized_tensor[0]
                
        # 4. Convert back to an image
        tensor_img = (stylized_tensor.squeeze().cpu() + 1.0) / 2.0
        tensor_img = tensor_img.clamp(0, 1)
        final_image = transforms.ToPILImage()(tensor_img)
        
        # 5. Return the image directly as a JPEG response
        img_byte_arr = io.BytesIO()
        final_image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        
        return Response(content=img_byte_arr.getvalue(), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))