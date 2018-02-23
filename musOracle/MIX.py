import musdb


def MIX(track):
    """Mixture as Estimate
    """

    # perform separtion
    estimates = {}
    for name, target in track.sources.items():
        # set accompaniment source
        estimates[name] = track.audio / len(track.targets)

    estimates['accompaniment'] = estimates['bass'] + \
        estimates['drums'] + estimates['other']
    return estimates


# initiate musdb
mus = musdb.DB()


mus.run(
    MIX,
    estimates_dir='MIX',
    subsets='test',
    parallel=True,
    cpus=4
)
