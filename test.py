from typing import Callable
import torch
from PIL import Image
from networkx import tensor_product
from utils import metrics

# def test_mse(gt_tensor: torch.tensor, predicted_tensor: torch.tensor):
#     gt_tensor = torch.randn(100, 100)  # Example ground truth tensor
#     predicted_tensor = torch.randn(100, 100)  # Example predicted tensor
#     mse_per_sample
    
#     # print('GT tensor : ', gt_tensor)
#     # print('Predicted tensor : ', predicted_tensor)
#     assert gt_tensor==predicted_tensor

# def mse(gt_tensor: torch.tensor, predicted_tensor: torch.tensor):
#     return metrics.mse_per_sample(predicted_tensor,gt_tensor)

def test_mse():
    # Test case 1
    gt_tensor_1 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    predicted_tensor_1 = torch.tensor([[[1.5, 2.5], [2.5, 3.5]]])
    mse_value_1 = metrics.mse_per_sample(predicted_tensor_1, gt_tensor_1)
    expected_result_1 = 0.625
    assert torch.allclose(mse_value_1, torch.tensor(expected_result_1)), f"Test case 1 failed: MSE value is incorrect: {mse_value_1}"

    # Test case 2
    gt_tensor_2 = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
    predicted_tensor_2 = torch.tensor([[[2.0, 2.0], [2.0, 2.0]]])
    mse_value_2 = metrics.mse_per_sample(predicted_tensor_2, gt_tensor_2)
    expected_result_2 = 1.0
    assert torch.allclose(mse_value_2, torch.tensor(expected_result_2)), f"Test case 2 failed: MSE value is incorrect: {mse_value_2}"

    # Test case 3
    gt_tensor_3 = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])
    predicted_tensor_3 = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    mse_value_3 = metrics.mse_per_sample(predicted_tensor_3, gt_tensor_3)
    expected_result_3 = 0.0
    assert torch.allclose(mse_value_3, torch.tensor(expected_result_3)), f"Test case 3 failed: MSE value is incorrect: {mse_value_3}"

    # Test case 4
    gt_tensor_4 = torch.tensor([[[1.0]]])
    predicted_tensor_4 = torch.tensor([[[2.0]]])
    mse_value_4 = metrics.mse_per_sample(predicted_tensor_4, gt_tensor_4)
    expected_result_4 = 1.0
    assert torch.allclose(mse_value_4, torch.tensor(expected_result_4)), f"Test case 4 failed: MSE value is incorrect: {mse_value_4}"

    # Test case 5
    gt_tensor_5 = torch.tensor([[[1.0, 2.0, 3.0]]])
    predicted_tensor_5 = torch.tensor([[[3.0, 2.0, 1.0]]])
    mse_value_5 = metrics.mse_per_sample(predicted_tensor_5, gt_tensor_5)
    expected_result_5 = 2.0
    assert torch.allclose(mse_value_5, torch.tensor(expected_result_5)), f"Test case 5 failed: MSE value is incorrect: {mse_value_5}"

    print("All test cases passed")

def test_psnr():
    # Test case 1
    gt_tensor_1 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    predicted_tensor_1 = torch.tensor([[[1.5, 2.5], [2.5, 3.5]]])
    mse1 = metrics.mse_per_sample(predicted_tensor_1, gt_tensor_1)
    psnr_value_1 = metrics.pnsr_per_sample(mse1)
    expected_result_1 = 22.0403  # Expected PSNR value for test case 1
    assert torch.allclose(psnr_value_1, torch.tensor(expected_result_1)), f"Test case 1 failed: PSNR value is incorrect: {psnr_value_1}"

    # Test case 2
    gt_tensor_2 = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
    predicted_tensor_2 = torch.tensor([[[2.0, 2.0], [2.0, 2.0]]])
    mse2 = metrics.mse_per_sample(predicted_tensor_2, gt_tensor_2)
    psnr_value_2 = metrics.pnsr_per_sample(mse2)
    expected_result_2 = 0.0  # Expected PSNR value for test case 2
    assert torch.allclose(psnr_value_2, torch.tensor(expected_result_2)), f"Test case 2 failed: PSNR value is incorrect: {psnr_value_2}"

    # Test case 3
    gt_tensor_3 = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    predicted_tensor_3 = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    mse3 = metrics.mse_per_sample(predicted_tensor_3, gt_tensor_3)
    psnr_value_3 = metrics.pnsr_per_sample(mse3)
    expected_result_3 = float('inf')  # Expected PSNR value for test case 3
    assert torch.allclose(psnr_value_3, torch.tensor(expected_result_3)), f"Test case 3 failed: PSNR value is incorrect: {psnr_value_3}"

    # Test case 4
    gt_tensor_4 = torch.tensor([[[1.0]]])
    predicted_tensor_4 = torch.tensor([[[2.0]]])
    mse4 = metrics.mse_per_sample(predicted_tensor_4, gt_tensor_4)
    psnr_value_4 = metrics.pnsr_per_sample(mse4)
    expected_result_4 = 0.0  # Expected PSNR value for test case 4
    assert torch.allclose(psnr_value_4, torch.tensor(expected_result_4)), f"Test case 4 failed: PSNR value is incorrect: {psnr_value_4}"

    # Test case 5
    gt_tensor_5 = torch.tensor([[[1.0, 2.0, 3.0]]])
    predicted_tensor_5 = torch.tensor([[[3.0, 2.0, 1.0]]])
    mse5 = metrics.mse_per_sample(predicted_tensor_5, gt_tensor_5)
    psnr_value_5 = metrics.pnsr_per_sample(mse5)
    expected_result_5 = 0.0  # Expected PSNR value for test case 5
    assert torch.allclose(psnr_value_5, torch.tensor(expected_result_5)), f"Test case 5 failed: PSNR value is incorrect: {psnr_value_5}"

    print("All test cases passed")

def test_ssim():
    # Test case 1
    gt_tensor_1 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    predicted_tensor_1 = torch.tensor([[[1.5, 2.5], [2.5, 3.5]]])
    ssim_value_1 = metrics.ssim(predicted_tensor_1, gt_tensor_1)
    expected_result_1 = 0.998855  # Expected SSIM value for test case 1
    assert torch.allclose(ssim_value_1, torch.tensor(expected_result_1), atol=1e-5), f"Test case 1 failed: SSIM value is incorrect: {ssim_value_1}"

    # Test case 2
    gt_tensor_2 = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
    predicted_tensor_2 = torch.tensor([[[2.0, 2.0], [2.0, 2.0]]])
    ssim_value_2 = metrics.ssim(predicted_tensor_2, gt_tensor_2)
    expected_result_2 = 0.0  # Expected SSIM value for test case 2
    assert torch.allclose(ssim_value_2, torch.tensor(expected_result_2)), f"Test case 2 failed: SSIM value is incorrect: {ssim_value_2}"

    # Test case 3
    gt_tensor_3 = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    predicted_tensor_3 = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    ssim_value_3 = metrics.ssim(predicted_tensor_3, gt_tensor_3)
    expected_result_3 = 1.0  # Expected SSIM value for test case 3
    assert torch.allclose(ssim_value_3, torch.tensor(expected_result_3)), f"Test case 3 failed: SSIM value is incorrect: {ssim_value_3}"

    # Test case 4
    gt_tensor_4 = torch.tensor([[[1.0]]])
    predicted_tensor_4 = torch.tensor([[[2.0]]])
    ssim_value_4 = metrics.ssim(predicted_tensor_4, gt_tensor_4)
    expected_result_4 = 0.0  # Expected SSIM value for test case 4
    assert torch.allclose(ssim_value_4, torch.tensor(expected_result_4)), f"Test case 4 failed: SSIM value is incorrect: {ssim_value_4}"

    # Test case 5
    gt_tensor_5 = torch.tensor([[[1.0, 2.0, 3.0]]])
    predicted_tensor_5 = torch.tensor([[[3.0, 2.0, 1.0]]])
    ssim_value_5 = metrics.ssim(predicted_tensor_5, gt_tensor_5)
    expected_result_5 = 0.0  # Expected SSIM value for test case 5
    assert torch.allclose(ssim_value_5, torch.tensor(expected_result_5)), f"Test case 5 failed: SSIM value is incorrect: {ssim_value_5}"

    print("All test cases passed!")