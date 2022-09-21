import torch
import model
from torchvision import transforms
from PIL import Image

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

img = Image.open('./test.png')
img = img.convert('RGB')
input_data = to_tensor(img)

print(input_data.shape)
encoder = model.encoder()
encoder.load_state_dict(torch.load('./model_encoder_weights.pth'))

out_data = encoder(input_data)
print(out_data.shape)
out_img = to_pil(out_data)

out_img.save('out.png')
