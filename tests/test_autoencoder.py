#test autoencoder
from src.models import autoencoder
from pytorch_lightning import Trainer
from src import generate
from src import utils
import torch
import geopandas as gpd
import pytest

@pytest.fixture()
def annotations(ROOT, tmpdir, rgb_path):
    data_path = "{}/tests/data/crown.shp".format(ROOT)
    gdf = gpd.read_file(data_path)
    gdf["RGB_tile"] = rgb_path
    annotations = generate.generate_crops(
        gdf=gdf, rgb_glob="{}/tests/data/*.tif".format(ROOT),
        convert_h5=False, sensor_glob="{}/tests/data/*.tif".format(ROOT), savedir=tmpdir)
    annotations = annotations.reset_index(drop=True)
    
    return annotations

def test_autoencoder(annotations, config):
    model = autoencoder.autoencoder(train_df=annotations, val_df=annotations, classes=3, config=config, comet_logger=None)
    trainer = Trainer(fast_dev_run=True)
    results = trainer.validate(model)

def test_classifier(annotations, config):
    cm = autoencoder.classifier(classes=3, config=config)
    model = autoencoder.autoencoder(train_df=annotations, val_df=annotations, classes=3, config=config, comet_logger=None)
    img1 = utils.load_image(img_path=annotations.image_path.iloc[0], image_size=config["image_size"])
    img2 = utils.load_image(img_path=annotations.image_path.iloc[0], image_size=config["image_size"])
    batch = torch.stack([img1,img2])
    reconstruction, features = model(batch)
    scores = cm(features)
    assert scores.shape == (2,3)
    
