#test RGB from
from src.models import RGB
import torch

def test_RGB():
    m = RGB.RGB(classes=10)
    images = torch.randn(20, 3, 11, 11)    
    output = m(images)
    assert output.shape == (20,10)