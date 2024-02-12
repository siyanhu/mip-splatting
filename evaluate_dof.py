import torch
from scene import Scene, SceneDOF
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.loss_utils import ssim
from utils.image_utils import psnr
from utils.loss_utils import l1_loss
from lpipsPyTorch import lpips
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, VirtualPaipelineParams2, get_combined_args
from gaussian_renderer import GaussianModel
from PIL import Image
import numpy as np
import time
import root_file_io as fio


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def current_timestamp(micro_second=False):
    t = time.time()
    if micro_second:
        return int(t * 1000 * 1000)
    else:
        return int(t * 1000)
    
def calculate_resolution(camera):
    orig_w, orig_h = camera.image_width, camera.image_height
    resolution_scale = 1

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
    return resolution
    

def render_set_virtual2(source_path, model_path, name, views, gaussians_na, pipeline, background, kernel_size):

    model_position_combo = model_path.split(fio.sep)
    if len(model_position_combo) < 3:
        return
    model_name = model_position_combo[-2]
    model_setting = model_position_combo[-1]
    session_dir = fio.createPath(fio.sep, [source_path, name, '_'.join([model_name, model_setting])])

    log_path = fio.createPath(fio.sep, [session_dir, 'render_log_' + str(current_timestamp()) + '.txtt'])
    render_path = fio.createPath(fio.sep, [session_dir, 'render'])
    makedirs(render_path, exist_ok=True)

    device = torch.device('cuda')

    psnr_value = 0
    l1_loss_value = 1
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        gt_image_name = view.image_name

        start_time = current_timestamp()
        rendering = render(view, gaussians_na, pipeline, background, kernel_size=kernel_size)["render"]
        after_time = current_timestamp()
        # gt = view.original_image[0:3, :, :]
        gt_original_image_path = os.path.join(source_path, 'images', gt_image_name)
        gt_image = Image.open(gt_original_image_path)

        resolution = calculate_resolution(view)
        resized_image_rgb = PILtoTorch(gt_image, resolution)
        gt_image = resized_image_rgb[:3, ...]

        save_path = fio.createPath(fio.sep, [render_path], gt_image_name)
        (savedir, savename, saveext) = fio.get_filename_components(save_path)
        fio.ensure_dir(savedir)
        torchvision.utils.save_image(rendering, os.path.join(render_path, gt_image_name))
        
        gt_image_gpu = gt_image.to(device)
        # rendering_gpu = rendering.to(device)
        psnr_value += psnr(rendering, gt_image_gpu).mean().double()
        psnr_log_value = psnr_value
        if idx > 0:
            psnr_log_value = psnr_log_value / idx
        log_str = "\n[INDEX {}] Rendering: L1LOSS {} PSNR {} TimeElapse {}"\
        .format(gt_image_name, l1_loss_value, psnr_log_value, str(after_time - start_time))
        with open(log_path, 'a+') as f:
            f.write(log_str)
    final = "\n[FINAL PSNR {}, L1_loss {}]".format(psnr_value/len(views), l1_loss_value/len(views))
    with open(log_path, 'a+') as f:
        f.write(final)


def render_sets_virtual2(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
    with torch.no_grad():
        print("Cuda current device: ", torch.cuda.current_device())
        print("Cuda is avail: ", torch.cuda.is_available())
        
        gaussians = GaussianModel(dataset.sh_degree)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        pretrain_source = dataset.model_path
        combo = pretrain_source.split('/')
        pretrain_tag = '_'.join(combo[0:2])

        kernel_size = dataset.kernel_size

        # log_path = fio.createPath(fio.sep, [dataset.source_path, "evaluate", pretrain_tag +  "_{}".format(iteration)])
        # fio.ensure_dir(log_path)
        # log_path = fio.createPath(fio.sep, [log_path], 'render_log.txt')
        # print("Saving log to", log_path)

        # scene_train = Scene(new_dataset, gaussians, load_iteration=iteration, shuffle=False)
        # scene_test = SceneDOF(dataset, gaussians, load_iteration=iteration, shuffle=False)
        # scene_train.test_cameras = scene_test.test_cameras
        scene = SceneDOF(dataset, gaussians, load_iteration=iteration, shuffle=False)
        render_set_virtual2(dataset.source_path, dataset.model_path, "evaluate", scene.getTestCameras(), gaussians, pipeline, background, kernel_size)



if __name__ == "__main__":
    # Set up command line argument parser
    # parser = ArgumentParser(description="Testing script parameters")
    # model = ModelParams(parser, sentinel=True)
    # pipeline = PipelineParams(parser)
    # parser.add_argument("--iteration", default=-1, type=int)
    # parser.add_argument("--quiet", action="store_true")
    # args = get_combined_args(parser)
    # print("Rendering from model " + args.model_path + ' to test ' + args.source_path)

    # # render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    # log_name = args.source_path.replace(fio.sep, '_')
    # log_name = log_name.replace('output', '')
    # log_name = 'render_log_' + log_name + '_' + str(current_timestamp()) + '.txt'

    # render_sets(model.extract(args), args.iteration, pipeline.extract(args))
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path + ' for testing set ' + args.source_path)

    # render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    # log_name = args.source_path.replace(fio.sep, '_')
    # print(args.model_path)
    # log_name = log_name.replace('output', '')
    # log_name = 'render_log_' + log_name + '_' + str(current_timestamp()) + '.txt'
    # print(log_name)

    # Initialize system state (RNG)
    # safe_state(args.quiet)
    # render_sets(model.extract(args), args.iteration, pipeline.extract(args))
    virtual_pipeline = VirtualPaipelineParams2()
    render_sets_virtual2(model.extract(args), args.iteration, virtual_pipeline)