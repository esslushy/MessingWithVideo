# MessingWithVideo
This is just practice using live video in a machine learning setup

#Requirements
python 3
pip install -r requirements.txt
ffmpeg

# How to run
download these files and unzip them into the root directory keeping the models folder https://www.dropbox.com/s/ssg39coiih5hjzz/models.zip?dl=1 and get this and place it in models https://www.dropbox.com/s/kh8izr3fkvhitfn/vgg_normalised.t7?dl=1
then run:
python GetVideo.py --checkpoints models/relu5_1 models/relu4_1 models/relu3_1 models/relu2_1 models/relu1_1 --relu-targets relu5_1 relu4_1 relu3_1 relu2_1 relu1_1 --style-size 512 --alpha 0.8 --style-path /path/to/styleimgs --twitch_key={key} -- height={height} --width={width}