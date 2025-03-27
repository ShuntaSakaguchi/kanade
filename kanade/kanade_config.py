#default device
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_device(new_device):
    """デバイス設定を変更し、パッケージ全体に適用する"""
    global device
    device = new_device
