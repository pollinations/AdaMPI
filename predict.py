from cog import BasePredictor, Path, Input, Path
import os  
from glob import glob
# import PIL image
from PIL import Image
from math import ceil

#MODEL_PATHS = "--smpl_model_folder /smpl_data --AE_path_fname /avatarclip_data/model_VAE_16.pth --codebook_fname /avatarclip_data/codebook.pth"

# INIT_COMMANDS="""pip install git+https://github.com/voodoohop/neural_renderer.git
# mv /avatarclip_data/* /src/AvatarGen/ShapeGen/data/
# mkdir -p /src/smpl_models
# mv /smpl_data /src/smpl_models/smpl"""

class Predictor(BasePredictor):
    def setup(self):
        os.system('mkdir -p /src/adampiweight')
        os.system('mv -v /*.pth /src/adampiweight')
    def predict(self,
            image: Path = Input(description="Image to enlarge"),
    ) -> Path:
        """run python gen_3dphoto.py \
            --img_path images/0810.png \
            --disp_path images/depth/0810.png \
            --width 384 \
            --height 256 \
            --save_path 0810.mp4 \
            --ckpt_path adampiweight/adampi_64p.pth"""

        image_path = image.resolve()

        # Get width and height of image
        im = Image.open(image_path)
        original_width, original_height = im.size

        new_width, new_height = calculate_dimensions(original_width, original_height)
        print("got transformed size",new_width, new_height)
        # resize image (dont use thumbnail)
        im = im.resize((new_width, new_height), Image.LANCZOS)
        # save image
        im.save(image_path)

        os.chdir("/DPT")
        print("image", image_path)
        os.system(f'cp "{image_path}" ./input')
        os.system("python run_monodepth.py")

        depth_map_path = os.path.join("/DPT", glob("./output_monodepth/*.png")[0])
        print("depth_map_path", depth_map_path)
        os.chdir("/src")
        os.system(f'python gen_3dphoto.py --img_path "{image_path}" --disp_path "{depth_map_path}" --width {new_width} --height {new_height} --save_path "./3dphoto.mp4" --ckpt_path adampiweight/adampi_64p.pth')
        
        # use ffmpeg to resize ./3dphoto.mp4 to original size
        os.system(f'ffmpeg -i ./3dphoto.mp4 -vf scale={original_width}:{original_height} ./3dphoto_out.mp4')

        return Path("./3dphoto_out.mp4")
        # print("glob after", glob("./output/coarse_shape/*.obj"))
        # print("returning",filepaths)
        # return Path(filepaths[0])



def calculate_dimensions(original_width, original_height):

        # limit width and height to a maximum of 1024 pixels maintaining the aspect ratio
        limited_width = min(original_width, 640)
        limited_height = limited_width * original_height / original_width

        limited_height = min(limited_height, 640)
        limited_width = limited_height * original_width / original_height

        # resize image to a larger multiple of 128 using ceil
        new_width = ceil(limited_width / 128) * 128
        new_height = ceil(limited_height / 128) * 128

        return new_width, new_height