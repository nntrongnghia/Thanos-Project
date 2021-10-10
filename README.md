# Work In Progress - Thanos Project
Real-time hand gesture recognition for PC/robot manipulation

[Project management](https://www.notion.so/tonynguyen/Thanos-Project-fca58bc5858a458b88f05487f23a7515)


# Setup

All commands are ran from Thanos-Project root

```
pip install -r requirements.txt
```
- Add this repo in your PYTHONPATH
- Create `dataset_config.json` with:
    - "ipn": "path/to/ipn/root"

# Training
```
python thanos\trainers\train_on_ipn.py CONFIG_PATH
```
Example
```
python thanos\trainers\train_on_ipn.py thanos\trainers\expe\default_config.py
```