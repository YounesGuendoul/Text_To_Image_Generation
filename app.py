import torch
import tkinter as tk
import customtkinter as ctk
from PIL import ImageTk
from torch import autocast
from diffusers import StableDiffusionPipeline

app = tk.Tk()
app.geometry("542x642")
app.title("Image Generation App")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(app, height=40, width=512, text_color="black", fg_color="white")
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(app, height=512, width=512)
lmain.place(x=10, y=110)

modelid = "runwayml/stable-diffusion-v1-5"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid,revision="fp16",torch_dtype=torch.float16,use_auth_token="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")
pipe.to(device)


def generate():
    with autocast(device):
        image = pipe(prompt.get(), guidance_scale=8.5).images[0]
    image.save('generatedimage.png')
    img = ImageTk.PhotoImage(image) 
    lmain.configure(image=img)

trigger = ctk.CTkButton(app, height=40, width=120, text_color="white", fg_color="green",command= generate)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)


app.mainloop()