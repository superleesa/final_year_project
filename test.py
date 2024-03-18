from typing import Callable
import torch
from PIL import Image
from networkx import tensor_product
from utils import metrics

def test_mse():
    # Test case 1
    gt_tensor_1 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    predicted_tensor_1 = torch.tensor([[[1.5, 2.5], [2.5, 3.5]]])
    mse_value_1 = metrics.mse_per_sample(predicted_tensor_1, gt_tensor_1)
    expected_result_1 = torch.tensor([0.625])
    assert torch.allclose(mse_value_1, torch.tensor(expected_result_1)), f"Test case 1 failed: MSE value is incorrect: {mse_value_1}"

    # Test case 2
    gt_tensor_2 = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
    predicted_tensor_2 = torch.tensor([[[2.0, 2.0], [2.0, 2.0]]])
    mse_value_2 = metrics.mse_per_sample(predicted_tensor_2, gt_tensor_2)
    expected_result_2 = torch.tensor([1.0])
    assert torch.allclose(mse_value_2, torch.tensor(expected_result_2)), f"Test case 2 failed: MSE value is incorrect: {mse_value_2}"

    # Test case 3
    gt_tensor_3 = torch.tensor([[[1.0, 2.0, 3.0]]])
    predicted_tensor_3 = torch.tensor([[[3.0, 2.0, 1.0]]])
    mse_value_3 = metrics.mse_per_sample(predicted_tensor_3, gt_tensor_3)
    expected_result_3 = torch.tensor([2.0])
    assert torch.allclose(mse_value_3, torch.tensor(expected_result_3)), f"Test case 3 failed: MSE value is incorrect: {mse_value_3}"


def test_psnr():
    # Test case 1
    gt_tensor_1 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    predicted_tensor_1 = torch.tensor([[[1.5, 2.5], [2.5, 3.5]]])
    mse1 = metrics.mse_per_sample(predicted_tensor_1, gt_tensor_1)
    psnr_value_1 = metrics.psnr_per_sample(mse1)
    expected_result_1 = torch.tensor([22.0403])  # Expected PSNR value for test case 1
    assert torch.allclose(psnr_value_1, torch.tensor(expected_result_1)), f"Test case 1 failed: PSNR value is incorrect: {psnr_value_1}"

    # Test case 2
    gt_tensor_2 = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
    predicted_tensor_2 = torch.tensor([[[2.0, 2.0], [2.0, 2.0]]])
    mse2 = metrics.mse_per_sample(predicted_tensor_2, gt_tensor_2)
    psnr_value_2 = metrics.psnr_per_sample(mse2)
    expected_result_2 = torch.tensor([0.0])  # Expected PSNR value for test case 2
    assert torch.allclose(psnr_value_2, torch.tensor(expected_result_2)), f"Test case 2 failed: PSNR value is incorrect: {psnr_value_2}"

    # Test case 3
    gt_tensor_3 = torch.tensor([[[1.0, 2.0, 3.0]]])
    predicted_tensor_3 = torch.tensor([[[3.0, 2.0, 1.0]]])
    mse3 = metrics.mse_per_sample(predicted_tensor_3, gt_tensor_3)
    psnr_value_3 = metrics.psnr_per_sample(mse3)
    expected_result_3 = torch.tensor([0.0]) # Expected PSNR value for test case 3
    assert torch.allclose(psnr_value_3, torch.tensor(expected_result_3)), f"Test case 3 failed: PSNR value is incorrect: {psnr_value_3}"

    print("All test cases passed")

def test_ssim():
    # Test case 1
    gt_tensor_1 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    predicted_tensor_1 = torch.tensor([[[1.5, 2.5], [2.5, 3.5]]])
    ssim_value_1 = metrics.ssim(predicted_tensor_1, gt_tensor_1)
    expected_result_1 = torch.tensor([0.998855])  # Expected SSIM value for test case 1
    assert torch.allclose(ssim_value_1, torch.tensor(expected_result_1), atol=1e-5), f"Test case 1 failed: SSIM value is incorrect: {ssim_value_1}"

    # Test case 2
    gt_tensor_2 = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
    predicted_tensor_2 = torch.tensor([[[2.0, 2.0], [2.0, 2.0]]])
    ssim_value_2 = metrics.ssim(predicted_tensor_2, gt_tensor_2)
    expected_result_2 = torch.tensor([0.0])  # Expected SSIM value for test case 2
    assert torch.allclose(ssim_value_2, torch.tensor(expected_result_2)), f"Test case 2 failed: SSIM value is incorrect: {ssim_value_2}"

    # Test case 3
    gt_tensor_3 = torch.tensor([[[1.0, 2.0, 3.0]]])
    predicted_tensor_3 = torch.tensor([[[3.0, 2.0, 1.0]]])
    ssim_value_3 = metrics.ssim(predicted_tensor_3, gt_tensor_3)
    expected_result_3 = torch.tensor([0.0])  # Expected SSIM value for test case 3
    assert torch.allclose(ssim_value_3, torch.tensor(expected_result_3)), f"Test case 3 failed: SSIM value is incorrect: {ssim_value_3}"

