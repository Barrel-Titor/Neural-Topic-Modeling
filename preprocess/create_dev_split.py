import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

import utils

if __name__ == "__main__":
    indir = Path("../data/20ng")
    outdir = Path("processed-dev")
    outdir = Path(indir, outdir)
    outdir.mkdir(exist_ok=True)

    # copy over the train files
    shutil.copy(Path(indir, "train.jsonlist"), Path(outdir, "train.jsonlist"))
    shutil.copy(Path(indir, "processed/train.npz"), Path(outdir, "train.npz"))
    shutil.copy(Path(indir, "processed/train.ids.json"), Path(outdir, "train.ids.json"))
    shutil.copy(Path(indir, "processed/train.vocab.json"), Path(outdir, "train.vocab.json"))

    # read in test
    test_jsonlist = utils.load_jsonlist(Path(indir, "test.jsonlist"))
    test_counts = utils.load_sparse(Path(indir, "processed/test.npz"))
    test_ids = utils.load_json(Path(indir, "processed/test.ids.json"))

    # split into a dev set
    dev_jsonlist, test_jsonlist, dev_counts, test_counts, dev_ids, test_ids = (
        train_test_split(
            test_jsonlist,
            test_counts,
            test_ids,
            test_size=0.5,
            random_state=11225
        )
    )

    # save
    utils.save_jsonlist(dev_jsonlist, Path(outdir, "dev.jsonlist"))
    utils.save_sparse(dev_counts, Path(outdir, "dev.npz"))
    utils.save_json(dev_ids, Path(outdir, "dev.ids.json"))

    utils.save_jsonlist(test_jsonlist, Path(outdir, "test.jsonlist"))
    utils.save_sparse(test_counts, Path(outdir, "test.npz"))
    utils.save_json(test_ids, Path(outdir, "test.ids.json"))