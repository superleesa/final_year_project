from fire import Fire
from metrics import mse_per_sample, psnr_per_sample, ssim_per_sample

def evaluation_script(y_test, trained_model):
    y_pred = trained_model(eval_data)
    mse_value = mse_per_sample(y_pred, y_test)
    psnr_value = psnr_per_sample(mse_value)
    ssim_value = ssim_per_sample(y_pred, y_test)
    
    # Create DataFrame
    metrics_df = pd.DataFrame({
        'MSE': [mse_value],
        'PSNR': [psnr_value],
        'SSIM': [ssim_value]
    })

    # Save DataFrame as CSV
    metrics_df.to_csv('evaluation_results.csv', index=False)

if __name__ = "__main__":
    Fire(evaluation_script)