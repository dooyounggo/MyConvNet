from isp.parameters_isp import *
import cv2
import subsets.subset_functions as sf


if __name__ == '__main__':
    Param = Parameters()
    save_dir = os.path.join(Param.save_dir, 'results_test')
    noisy_dir = os.path.join(save_dir, 'noisy')
    gt_dir = os.path.join(save_dir, 'ground_truth')
    denoised_dir = os.path.join(save_dir, 'denoised')

    images_orig = sorted(os.listdir(Param.test_dir))
    images_noisy = sorted(os.listdir(noisy_dir))
    images_gt = sorted(os.listdir(gt_dir))
    images_denoised = sorted(os.listdir(denoised_dir))
    for orig, no, gt, dn in zip(images_orig, images_noisy, images_gt, images_denoised):
        img_orig = cv2.imread(os.path.join(Param.test_dir, orig))
        shape = img_orig.shape[0:2]

        img_no = cv2.imread(os.path.join(noisy_dir, no))
        os.remove(os.path.join(noisy_dir, no))
        img_no = sf.resize_with_crop_or_pad(img_no, out_size=shape)
        cv2.imwrite(os.path.join(noisy_dir, orig), sf.to_int(img_no), [cv2.IMWRITE_JPEG_QUALITY, 100])

        img_gt = cv2.imread(os.path.join(gt_dir, gt))
        os.remove(os.path.join(gt_dir, gt))
        img_gt = sf.resize_with_crop_or_pad(img_gt, out_size=shape)
        cv2.imwrite(os.path.join(gt_dir, orig), sf.to_int(img_gt), [cv2.IMWRITE_JPEG_QUALITY, 100])

        img_dn = cv2.imread(os.path.join(denoised_dir, dn))
        os.remove(os.path.join(denoised_dir, dn))
        img_dn = sf.resize_with_crop_or_pad(img_dn, out_size=shape)
        cv2.imwrite(os.path.join(denoised_dir, orig), sf.to_int(img_dn), [cv2.IMWRITE_JPEG_QUALITY, 100])
