
!pip install ftfy regex tqdm
!pip install git+https://github.com/openai/CLIP.git
!pip install Pillow

#Image Path Location
image_path = "your_image.jpg" 
image = Image.open(image_path).convert("RGB")
image_input = preprocess(image).unsqueeze(0).to(device)


text_inputs = torch.cat([
    clip.tokenize("graffiti on public objects"),
    clip.tokenize("a photo of trash on the street"),
    clip.tokenize("clean walls and streets"),

]).to(device)


with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

import matplotlib.pyplot as plt

labels = ["graffiti", "trash", "clean background"]
probs = similarity.squeeze().cpu().numpy()
for label, prob in zip(labels, probs):
    print(f"{label}: {prob:.4f}")


plt.imshow(image)
plt.axis('off')
plt.title("Input Image")
plt.show()

