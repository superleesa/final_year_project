import torch
from utils import metrics

def test_mse():
    # Test case 1
    gt_tensor_1 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    predicted_tensor_1 = torch.tensor([[[1.5, 2.5], [2.5, 3.5]]])
    mse_value_1 = metrics.mse_per_sample(predicted_tensor_1, gt_tensor_1)
    expected_result_1 = torch.tensor([0.25])
    print(mse_value_1, expected_result_1)
    assert torch.allclose(mse_value_1, expected_result_1), f"Test case 1 failed: MSE value is incorrect: {mse_value_1}"


def test_psnr():
    gt_tensor_1 = torch.tensor([[[0.8, 0.9], [1.0, 0.7]]])
    predicted_tensor_1 = torch.tensor([[[0.75, 0.88], [1.02, 0.68]]])
    mse1 = metrics.mse_per_sample(predicted_tensor_1, gt_tensor_1)
    print(mse1)
    psnr_value_1 = metrics.psnr_per_sample(mse1)
    expected_result_1 = torch.tensor([30.45757491])
    assert torch.allclose(psnr_value_1, expected_result_1, atol=0.15), f"Test case 1 failed: PSNR value is incorrect: {psnr_value_1}"

def test_ssim():
    # Test case 1
    gt_tensor_1 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    predicted_tensor_1 = torch.tensor([[[1.5, 2.5], [2.5, 3.5]]])
    ssim_value_1 = metrics.ssim(predicted_tensor_1, gt_tensor_1)
    expected_result_1 = torch.tensor([0.998855])  # Expected SSIM value for test case 1
    assert torch.allclose(ssim_value_1, torch.tensor(expected_result_1), atol=1e-5), f"Test case 1 failed: SSIM value is incorrect: {ssim_value_1}"

