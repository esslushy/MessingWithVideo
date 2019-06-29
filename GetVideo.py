import cv2
from twitchstream.outputvideo import TwitchOutputStream
import argparse
from utils import get_img
from wct import WCT

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoints', nargs='+', type=str, help='List of checkpoint directories', required=True)
parser.add_argument('--relu-targets', nargs='+', type=str, help='List of reluX_1 layers, corresponding to --checkpoints', required=True)
parser.add_argument('--vgg-path', type=str, help='Path to vgg_normalised.t7', default='models/vgg_normalised.t7')
parser.add_argument('--style-path', type=str, dest='style_path', help='Style image')
parser.add_argument('--out-path', type=str, dest='out_path', help='Output folder path')
parser.add_argument('--keep-colors', action='store_true', help="Preserve the colors of the style image", default=False)
parser.add_argument('--device', type=str, help='Device to perform compute on, e.g. /gpu:0', default='/gpu:0')
parser.add_argument('--style-size', type=int, help="Resize style image to this size before cropping, default 512", default=0)
parser.add_argument('--crop-size', type=int, help="Crop square size, default 256", default=0)
parser.add_argument('--content-size', type=int, help="Resize short side of content image to this", default=0)
parser.add_argument('--passes', type=int, help="# of stylization passes per content image", default=1)
parser.add_argument('-r','--random', type=int, help="Choose # of random subset of images from style folder", default=0)
parser.add_argument('--alpha', type=float, help="Alpha blend value", default=1)
parser.add_argument('--concat', action='store_true', help="Concatenate style image and stylized output", default=False)
parser.add_argument('--adain', action='store_true', help="Use AdaIN instead of WCT", default=False)

## Style swap args
parser.add_argument('--swap5', action='store_true', help="Swap style on layer relu5_1", default=False)
parser.add_argument('--ss-alpha', type=float, help="Style swap alpha blend", default=0.6)
parser.add_argument('--ss-patch-size', type=int, help="Style swap patch size", default=3)
parser.add_argument('--ss-stride', type=int, help="Style swap stride", default=1)

# Twitch args
parser.add_argument('--twitch_key', type=str, help='Twitch api key')
parser.add_argument('--width', type=int, help='Video frame width', default=1920)
parser.add_argument('--height', type=int, help='Video frame heigh', default=1080)
parser.add_argument('--fps', type=int, help='Video FPS', default=30)

arguments = parser.parse_args()

#load model
wct_model = WCT(checkpoints=arguments.checkpoints, 
                                relu_targets=arguments.relu_targets,
                                vgg_path=arguments.vgg_path, 
                                device=arguments.device,
                                ss_patch_size=arguments.ss_patch_size, 
                                ss_stride=arguments.ss_stride)

#load style image
style = get_img(arguments.style_path)

def changeImage(image):
    return wct_model.predict(image, style, arguments.alpha, arguments.swap5, arguments.ss_alpha, arguments.adain)

camera = cv2.VideoCapture(0)#get camera video

#build stream input
stream = TwitchOutputStream(arguments.twitch_key, width=arguments.width, height=arguments.height, fps=arguments.fps)

while(True):#get data every tick
    ret, frame  = camera.read()#returns if it worked and true or false and the frame

    frame = changeImage(frame)

    #local viewing
    # cv2.imshow('frame',frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    #over twitch
    stream.send_video_frame(frame)

#release camera and close window
camera.release()
cv2.destroyAllWindows()