from PIL import Image

import argparse



parser = argparse.ArgumentParser(description='')
parser.add_argument('-f', nargs=1, required=True, help='png file')
args = parser.parse_args()  
img = Image.open(args.f[0])
width,height=100,100
img = img.resize((width,height))
img.save("out.png")

