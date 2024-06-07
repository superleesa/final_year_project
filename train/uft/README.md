# UFT Training

If you have a config yml file under this directory, you can simply run with
```bash
python3 unpaired_train.py
```

If you want to specify dataset_dir, checkpoint_dir, and save_dir, run as follow. The arguments in the command line have higher priority than the yml file.

```bash
python3 unpaired_train.py dataset_dir checkpoint_path save_dir
```