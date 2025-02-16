import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from data_prep.feature_store.featurize import preprocess

if __name__ == '__main__':
    preprocess()