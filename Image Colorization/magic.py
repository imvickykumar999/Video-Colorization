
# >>> pip install pyinstaller
# >>> pyinstaller --onefile --noconsole "magic.py"
# >>> cd dist

# put imgs/input.jpg in dist folder and double click 
# on ImgColexe.exe to get output in imgs_out folder.

import argparse, os
import matplotlib.pyplot as plt
from colorizers import *

try:
	os.mkdir('imgs')
except:
	pass

try:
	os.mkdir('imgs_out')
except:
	pass

parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
opt = parser.parse_args()

colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()
if(opt.use_gpu):
	colorizer_eccv16.cuda()
	colorizer_siggraph17.cuda()

img = load_img('imgs/input.jpg')
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
if(opt.use_gpu):
	tens_l_rs = tens_l_rs.cuda()

img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
plt.imsave('imgs_out/eccv16.png', out_img_eccv16)
plt.imsave('imgs_out/siggraph17.png', out_img_eccv16)
