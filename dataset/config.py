class DefaultConfigs(object):
    #1.string parameters
    dataroot = "single_orthogonal_mix"
    model_name = "Resnet34_DI"
    weights = "./checkpoints/"
    best_models = weights + "best_model/"
    submit = "./submit/"
    logs = "./logs/"
    gpus = "2,3"
    augmen_level = "medium"  # "light","hard","hard2"

    #2.numeric parameters
    epochs = 30
    batch_size = 4
    img_height = 320
    img_weight = 320
    num_classes = 13
    seed = 888
    lr = 1e-4
    lr_decay = 1e-4
    weight_decay = 1e-4
    plot_every = 10

config = DefaultConfigs()
