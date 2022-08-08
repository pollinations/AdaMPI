from cog import BasePredictor, Path, Input, Path
import os  
from glob import glob

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

        os.chdir("/DPT")
        image_path = image.resolve()
        print("image", image_path)
        os.system(f'cp "{image_path}" ./input')
        os.system("python run_monodepth.py")

        depth_map_path = os.path.join("/DPT", glob("./output_monodepth/*.png")[0])
        print("depth_map_path", depth_map_path)
        os.chdir("/src")
        os.system(f'python gen_3dphoto.py --img_path "{image_path}" --disp_path "{depth_map_path}" --width 384 --height 256 --save_path "./3dphoto.mp4" --ckpt_path adampiweight/adampi_64p.pth')
        return Path("./3dphoto.mp4")
        # print("glob after", glob("./output/coarse_shape/*.obj"))
        # print("returning",filepaths)
        # return Path(filepaths[0])

