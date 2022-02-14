import munch


def TrainOptions():
    to = munch.Munch()
    # Iterations
    to.total_iterations = 150001
    # Save checkpoint every x iters
    to.save_latest_freq = 5000

    # Save checkpoint every x epochs
    to.save_epoch_freq = 5

    # Adam settings
    to.beta1 = 0.5

    # gan type
    to.gan_mode = 'lsgan'

    return to
