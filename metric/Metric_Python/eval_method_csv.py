import numpy as np
from PIL import Image
from Metric import *
from natsort import natsorted
from tqdm import tqdm
import os
import warnings
import csv

warnings.filterwarnings("ignore")


def evaluation_one(ir_name, vi_name, f_name):
    f_img = Image.open(f_name).convert('L')
    ir_img = Image.open(ir_name).convert('L')
    vi_img = Image.open(vi_name).convert('L')

    f_img_int = np.array(f_img).astype(np.int32)
    f_img_double = np.array(f_img).astype(np.float32)

    ir_img_int = np.array(ir_img).astype(np.int32)
    ir_img_double = np.array(ir_img).astype(np.float32)

    vi_img_int = np.array(vi_img).astype(np.int32)
    vi_img_double = np.array(vi_img).astype(np.float32)

    EN = EN_function(f_img_int)
    MI = MI_function(ir_img_int, vi_img_int, f_img_int, gray_level=256)

    SF = SF_function(f_img_double)
    SD = SD_function(f_img_double)
    AG = AG_function(f_img_double)
    PSNR = PSNR_function(ir_img_double, vi_img_double, f_img_double)
    MSE = MSE_function(ir_img_double, vi_img_double, f_img_double)
    VIF = VIF_function(ir_img_double, vi_img_double, f_img_double)
    CC = CC_function(ir_img_double, vi_img_double, f_img_double)
    SCD = SCD_function(ir_img_double, vi_img_double, f_img_double)
    Qabf = Qabf_function(ir_img_double, vi_img_double, f_img_double)
    Nabf = Nabf_function(ir_img_double, vi_img_double, f_img_double)
    SSIM = SSIM_function(ir_img_double, vi_img_double, f_img_double)
    MS_SSIM = MS_SSIM_function(ir_img_double, vi_img_double, f_img_double)

    return EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM


if __name__ == '__main__':
    dataset_name = 'msrs'

    ir_dir = os.path.join(
        '/mlx_devbox/users/wangminglei.04/playground/image_fusion/dataset/msrs/test', 'ir')
    vi_dir = os.path.join(
        '/mlx_devbox/users/wangminglei.04/playground/image_fusion/dataset/msrs/test', 'vi')

    Methods = ['cddfusion', 'cmfuse', 'control', 'MaeFuse', 'nestfuse', 'occo', 'rfnnest', 'SAGE', 'swinfusion', 'tdfusion']

    for Method in Methods:
        f_dir = os.path.join(
            '/mlx_devbox/users/wangminglei.04/playground/image_fusion/results',
            dataset_name, Method
        )

        save_dir = 'Results/'
        os.makedirs(save_dir, exist_ok=True)

        csv_path = os.path.join(
            save_dir, f'metric_{dataset_name}_{Method}.csv')

        with_mean = True

        # 表头
        header = [
            'filename', 'EN', 'MI', 'SF', 'AG', 'SD',
            'CC', 'SCD', 'VIF', 'MSE', 'PSNR',
            'Qabf', 'Nabf', 'SSIM', 'MS_SSIM'
        ]

        rows = []

        filelist = natsorted(os.listdir(ir_dir))
        eval_bar = tqdm(filelist)

        for _, item in enumerate(eval_bar):
            ir_name = os.path.join(ir_dir, item)
            vi_name = os.path.join(vi_dir, item)
            f_name = os.path.join(f_dir, item)

            EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM = evaluation_one(
                ir_name, vi_name, f_name)

            row = [
                item,
                round(EN, 3), round(MI, 3), round(SF, 3), round(AG, 3), round(SD, 3),
                round(CC, 3), round(SCD, 3), round(VIF, 3), round(MSE, 3), round(PSNR, 3),
                round(Qabf, 3), round(Nabf, 3), round(SSIM, 3), round(MS_SSIM, 3)
            ]

            rows.append(row)
            eval_bar.set_description(f"{Method} | {item}")

        # 计算 mean 和 std（不污染原数据）
        if with_mean and len(rows) > 0:
            data_array = np.array([r[1:] for r in rows], dtype=np.float32)

            mean_row = ['mean'] + list(np.round(np.mean(data_array, axis=0), 3))
            std_row = ['std'] + list(np.round(np.std(data_array, axis=0), 3))

            rows.append(mean_row)
            rows.append(std_row)

        # 写入 CSV
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

        print(f"结果已保存到: {csv_path}")