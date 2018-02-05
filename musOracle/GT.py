import musdb


def GT(track):
    """Ground Truth Signals
    """

    # perform separtion
    estimates = {}
    for name, target in track.targets.items():
        # set accompaniment source
        estimates[name] = target.audio

    return estimates


# initiate musdb
mus = musdb.DB()


mus.run(
    GT,
    estimates_dir='GT',
    subsets='test',
    parallel=True,
    cpus=4
)
